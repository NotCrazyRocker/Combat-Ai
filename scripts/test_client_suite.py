#!/usr/bin/env python3
"""Offline observation checks and opt-in live checks for the MinecraftAI client."""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from aim_geometry import head_aim_error, target_yaw_error_degrees
from combo_ppo_math import (
    ATTACK_MISS_PENALTY,
    ATTACK_SUCCESS_REWARD,
    COMBO_REWARD_PER_EXTRA_HIT,
    FALSE_ATTACK_PENALTY,
    FORWARD_TOGGLE_PENALTY,
    INCOMING_HIT_PENALTY,
    KNOCKBACK_HIT_REWARD,
    MAX_COMBO_REWARD,
    MISSED_ELIGIBLE_PENALTY,
    PREMATURE_ATTACK_PENALTY,
    MOVEMENT_OBSERVATION_SIZE,
    SPRINT_TOGGLE_PENALTY,
    encode_movement_observation,
    movement_transition_reward,
)
from yaw_ppo_math import (
    encode_pitch_observation,
    encode_yaw_observation,
    encode_yaw_pitch_observation,
    scripted_attack_decision,
    scripted_sword_attack_eligible,
    pitch_transition_reward,
    yaw_transition_reward,
    yaw_pitch_transition_reward,
    yaw_pitch_transition_rewards,
)

SCHEMA = "minecraftai.pvp.v1"
PROTOCOL = 2
ACTION_KEYS = ("forward", "backward", "left", "right", "jump", "sprint", "sneak", "attack", "use")
ACTION_HISTORY_INDEX = {name: index for index, name in enumerate(ACTION_KEYS)}
ACTION_STATE_KEYS = set(ACTION_KEYS) | {"control_enabled"}
PLAYER_FIELDS = {
    "player_x", "player_y", "player_z", "player_y_level", "player_velocity_x", "player_velocity_y",
    "player_velocity_z", "player_horizontal_speed", "camera_yaw", "camera_pitch", "camera_yaw_sin",
    "camera_yaw_cos", "camera_pitch_sin", "camera_pitch_cos", "camera_horizontal_fov_degrees",
    "player_on_ground", "player_sprinting", "player_sneaking", "player_fall_distance", "player_health",
    "player_absorption", "player_food", "attack_strength", "mainhand_is_sword", "ticks_since_hit_enemy",
    "has_hit_enemy", "ticks_since_hit_by_enemy", "has_been_hit", "last_hit_critical", "last_hit_knockback",
    "last_hit_sweep", "last_hit_normal", "last_hit_type",
    "hit_combo", "knockback_combo",
}
TARGET_FIELDS = {
    "target_valid", "target_distance", "target_horizontal_distance", "target_height", "target_health",
    "target_hurt_time", "target_on_ground", "target_sprinting", "target_sneaking", "target_yaw", "target_pitch",
    "target_relative_velocity_x", "target_relative_velocity_y", "target_relative_velocity_z",
    "target_relative_horizontal_speed", "target_on_screen", "target_bbox_projection_valid",
    "crosshair_on_target", "target_aim_projection_valid", "target_aim_dx", "target_aim_dy", "target_dx",
    "target_dy", "target_bbox_left", "target_bbox_top", "target_bbox_right", "target_bbox_bottom",
    "target_bbox_width", "target_bbox_height",
}
NUMERIC_FIELDS = {
    "player_x", "player_y", "player_z", "player_y_level", "player_velocity_x", "player_velocity_y",
    "player_velocity_z", "player_horizontal_speed", "camera_yaw", "camera_pitch", "camera_yaw_sin",
    "camera_yaw_cos", "camera_pitch_sin", "camera_pitch_cos", "camera_horizontal_fov_degrees",
    "player_fall_distance", "player_health", "player_absorption", "player_food", "attack_strength",
    "ticks_since_hit_enemy", "ticks_since_hit_by_enemy", "hit_combo", "knockback_combo",
    "target_distance", "target_horizontal_distance",
    "target_height", "target_health", "target_hurt_time", "target_yaw", "target_pitch",
    "target_relative_velocity_x", "target_relative_velocity_y", "target_relative_velocity_z",
    "target_relative_horizontal_speed", "target_aim_dx", "target_aim_dy", "target_dx", "target_dy",
    "target_bbox_left", "target_bbox_top", "target_bbox_right", "target_bbox_bottom", "target_bbox_width",
    "target_bbox_height",
}


def _finite_json_numbers(value: Any, path: str = "root") -> list[str]:
    errors: list[str] = []
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return errors
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            errors.append(f"{path} is not finite")
        return errors
    if isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(_finite_json_numbers(child, f"{path}[{index}]"))
    elif isinstance(value, dict):
        for key, child in value.items():
            errors.extend(_finite_json_numbers(child, f"{path}.{key}"))
    return errors


def validate_observation(frame: dict[str, Any]) -> list[str]:
    """Return contract violations for one full mod observation."""
    errors: list[str] = []
    if frame.get("type") != "observation":
        errors.append("type must be 'observation'")
    if frame.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA!r}")
    tick = frame.get("tick")
    if isinstance(tick, bool) or not isinstance(tick, int) or tick < 0:
        errors.append("tick must be a nonnegative integer")

    features = frame.get("features")
    if not isinstance(features, dict):
        return errors + ["features must be an object"]
    if "player_valid" not in features:
        errors.append("features.player_valid is missing")
    diagnostics = features.get("remote_control_diagnostics")
    if not isinstance(diagnostics, dict):
        errors.append("features.remote_control_diagnostics must be an object")
    else:
        state = diagnostics.get("action_state")
        clicks = diagnostics.get("consumed_clicks")
        if not isinstance(state, dict) or not isinstance(state.get("control_enabled"), bool):
            errors.append("remote action_state.control_enabled must be a boolean")
        if not isinstance(state, dict) or not ACTION_STATE_KEYS.issubset(state):
            errors.append("remote action_state is missing one or more action keys")
        elif any(not isinstance(state[key], bool) for key in ACTION_STATE_KEYS):
            errors.append("remote action_state values must be booleans")
        if not isinstance(clicks, dict) or not {"attack", "use"}.issubset(clicks):
            errors.append("consumed_clicks must contain attack and use counters")
        elif any(isinstance(clicks[k], bool) or not isinstance(clicks[k], int) or clicks[k] < 0 for k in ("attack", "use")):
            errors.append("consumed click counters must be nonnegative integers")
        for key in ("last_applied_sequence", "last_applied_observation_tick"):
            value = diagnostics.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < -1:
                errors.append(f"remote_control_diagnostics.{key} must be an integer >= -1")

    if features.get("player_valid") is True:
        for key in PLAYER_FIELDS | TARGET_FIELDS:
            if key not in features:
                errors.append(f"features.{key} is missing")
        for key in ("player_on_ground", "player_sprinting", "player_sneaking", "mainhand_is_sword",
                    "has_hit_enemy", "has_been_hit", "last_hit_critical", "last_hit_knockback",
                    "last_hit_sweep", "last_hit_normal", "target_valid", "target_on_screen",
                    "target_bbox_projection_valid", "crosshair_on_target", "target_aim_projection_valid"):
            if key in features and not isinstance(features[key], bool):
                errors.append(f"features.{key} must be a boolean")
        for key in ("camera_yaw", "camera_pitch", "camera_horizontal_fov_degrees", "player_food", "attack_strength"):
            value = features.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(f"features.{key} must be numeric")
        for key in NUMERIC_FIELDS & (PLAYER_FIELDS | TARGET_FIELDS):
            value = features.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(f"features.{key} must be numeric")
        if not isinstance(features.get("last_hit_type"), str):
            errors.append("features.last_hit_type must be a string")
        combo = features.get("hit_combo")
        if isinstance(combo, bool) or not isinstance(combo, int) or combo < 0:
            errors.append("features.hit_combo must be a nonnegative integer")
        knockback_combo = features.get("knockback_combo")
        if isinstance(knockback_combo, bool) or not isinstance(knockback_combo, int) or knockback_combo < 0:
            errors.append("features.knockback_combo must be a nonnegative integer")
        if isinstance(features.get("camera_pitch"), (int, float)) and not -90.0 <= features["camera_pitch"] <= 90.0:
            errors.append("camera_pitch is outside [-90, 90]")
        yaw = features.get("camera_yaw")
        yaw_sin = features.get("camera_yaw_sin")
        yaw_cos = features.get("camera_yaw_cos")
        if isinstance(yaw_sin, (int, float)) and not -1.0 <= yaw_sin <= 1.0:
            errors.append("camera_yaw_sin is outside [-1, 1]")
        if isinstance(yaw_cos, (int, float)) and not -1.0 <= yaw_cos <= 1.0:
            errors.append("camera_yaw_cos is outside [-1, 1]")
        if all(isinstance(value, (int, float)) and not isinstance(value, bool)
               and math.isfinite(float(value))
               for value in (yaw, yaw_sin, yaw_cos)):
            expected_sin = math.sin(math.radians(yaw))
            expected_cos = math.cos(math.radians(yaw))
            if abs(yaw_sin - expected_sin) > 1e-5 or abs(yaw_cos - expected_cos) > 1e-5:
                errors.append("camera_yaw_sin/cos do not match camera_yaw")
        if isinstance(features.get("camera_horizontal_fov_degrees"), (int, float)) and not 0 < features["camera_horizontal_fov_degrees"] < 180:
            errors.append("camera_horizontal_fov_degrees is outside (0, 180)")
        if isinstance(features.get("player_food"), (int, float)) and not 0 <= features["player_food"] <= 20:
            errors.append("player_food is outside [0, 20]")
        if isinstance(features.get("attack_strength"), (int, float)) and not 0 <= features["attack_strength"] <= 1:
            errors.append("attack_strength is outside [0, 1]")

        for key in ("target_bbox_left", "target_bbox_top", "target_bbox_right", "target_bbox_bottom",
                    "target_bbox_width", "target_bbox_height"):
            value = features.get(key)
            if isinstance(value, (int, float)) and not 0 <= value <= 1:
                errors.append(f"features.{key} is outside [0, 1]")
        if features.get("target_on_screen") is True and features.get("target_bbox_projection_valid") is not True:
            errors.append("target_on_screen is true while bbox projection is invalid")
        if features.get("target_bbox_left", 0) > features.get("target_bbox_right", 0):
            errors.append("target_bbox_left exceeds target_bbox_right")
        if features.get("target_bbox_top", 0) > features.get("target_bbox_bottom", 0):
            errors.append("target_bbox_top exceeds target_bbox_bottom")
        if features.get("target_bbox_projection_valid") is True:
            width = features.get("target_bbox_width", 0.0)
            height = features.get("target_bbox_height", 0.0)
            horizontal_span = features.get("target_bbox_right", 0.0) - features.get("target_bbox_left", 0.0)
            vertical_span = features.get("target_bbox_bottom", 0.0) - features.get("target_bbox_top", 0.0)
            if all(isinstance(value, (int, float)) and not isinstance(value, bool)
                   for value in (width, height, horizontal_span, vertical_span)):
                if abs(width - horizontal_span) > 1e-5:
                    errors.append("target_bbox_width does not match its projected edges")
                if abs(height - vertical_span) > 1e-5:
                    errors.append("target_bbox_height does not match its projected edges")
        if features.get("target_aim_projection_valid") is True:
            for key in ("target_aim_dx", "target_aim_dy"):
                if isinstance(features.get(key), bool) or not isinstance(features.get(key), (int, float)):
                    errors.append(f"features.{key} must be numeric when aim projection is valid")
        if features.get("crosshair_on_target") is True and features.get("target_valid") is not True:
            errors.append("crosshair_on_target is true without a valid target")

    history = frame.get("action_history")
    if not isinstance(history, list) or len(history) != 20:
        errors.append("action_history must contain exactly 20 rows")
    else:
        for row_index, row in enumerate(history):
            if not isinstance(row, list) or len(row) != 11:
                errors.append(f"action_history[{row_index}] must contain 11 entries")
                continue
            for value_index, value in enumerate(row[:9]):
                if value not in (0, 1, False, True):
                    errors.append(f"action_history[{row_index}][{value_index}] is not a boolean flag")
            for value_index, value in enumerate(row[9:], 9):
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    errors.append(f"action_history[{row_index}][{value_index}] is not a finite number")
    errors.extend(_finite_json_numbers(frame))
    return errors


def sample_observation() -> dict[str, Any]:
    features: dict[str, Any] = {key: 0.0 for key in PLAYER_FIELDS | TARGET_FIELDS}
    features.update({key: False for key in (
        "player_on_ground", "player_sprinting", "player_sneaking", "mainhand_is_sword", "has_hit_enemy",
        "has_been_hit", "last_hit_critical", "last_hit_knockback",
        "last_hit_sweep", "last_hit_normal",
        "target_valid", "target_on_screen", "target_bbox_projection_valid", "crosshair_on_target",
        "target_aim_projection_valid",
    )})
    features.update({
        "player_valid": True, "camera_horizontal_fov_degrees": 100.0, "player_food": 20,
        "camera_yaw_cos": 1.0,
        "attack_strength": 1.0, "ticks_since_hit_enemy": -1, "ticks_since_hit_by_enemy": -1,
        "last_hit_type": "none", "target_hurt_time": 0, "hit_combo": 0, "knockback_combo": 0,
    })
    features["remote_control_diagnostics"] = {
        "action_state": {key: False for key in ACTION_STATE_KEYS},
        "consumed_clicks": {"attack": 0, "use": 0},
        "last_applied_sequence": -1,
        "last_applied_observation_tick": -1,
    }
    return {
        "type": "observation", "schema": SCHEMA, "tick": 1, "features": features,
        "action_history": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0] for _ in range(20)],
    }


class ObservationContractTests(unittest.TestCase):
    def test_player_unavailable_frame_still_has_diagnostics(self) -> None:
        frame = sample_observation()
        diagnostics = frame["features"]["remote_control_diagnostics"]
        frame["features"] = {"player_valid": False, "remote_control_diagnostics": diagnostics}
        self.assertEqual(validate_observation(frame), [])

    def test_empty_target_projection_is_valid_observation(self) -> None:
        self.assertEqual(validate_observation(sample_observation()), [])

    def test_rejects_nan_and_invalid_camera_pitch(self) -> None:
        frame = sample_observation()
        frame["features"]["camera_pitch"] = float("nan")
        frame["features"]["player_y"] = 91.0
        self.assertTrue(any("camera_pitch is outside" in error or "not finite" in error
                            for error in validate_observation(frame)))

    def test_accepts_unwrapped_yaw_when_its_trigonometric_features_match(self) -> None:
        frame = sample_observation()
        yaw = 721.25
        frame["features"].update(
            camera_yaw=yaw,
            camera_yaw_sin=math.sin(math.radians(yaw)),
            camera_yaw_cos=math.cos(math.radians(yaw)),
        )
        self.assertEqual(validate_observation(frame), [])

    def test_rejects_inconsistent_yaw_trigonometric_features(self) -> None:
        frame = sample_observation()
        frame["features"].update(camera_yaw=721.25, camera_yaw_sin=0.0, camera_yaw_cos=1.0)
        self.assertTrue(any("camera_yaw_sin/cos" in error for error in validate_observation(frame)))

    def test_head_point_at_reticle_has_zero_vertical_error(self) -> None:
        frame = {"features": {
            "target_valid": True,
            "target_aim_projection_valid": True,
            "target_aim_dy": 0.5,
        }}
        self.assertEqual(head_aim_error(frame), 0.0)

    def test_head_point_error_sign_points_toward_screen_center(self) -> None:
        above = {"features": {"target_valid": True, "target_aim_projection_valid": True,
                              "target_aim_dy": 0.2}}
        below = {"features": {"target_valid": True, "target_aim_projection_valid": True,
                              "target_aim_dy": 0.8}}
        self.assertAlmostEqual(head_aim_error(above), -0.3)
        self.assertAlmostEqual(head_aim_error(below), 0.3)

    def test_head_aim_error_ignores_invalid_projection(self) -> None:
        frame = {"features": {"target_valid": True, "target_aim_projection_valid": False,
                               "target_aim_dy": 0.5}}
        self.assertIsNone(head_aim_error(frame))

    def test_head_aim_error_rejects_missing_or_nonfinite_projection(self) -> None:
        missing = {"features": {"target_valid": True, "target_aim_projection_valid": True}}
        nonfinite = {"features": {"target_valid": True, "target_aim_projection_valid": True,
                                   "target_aim_dy": float("nan")}}
        self.assertIsNone(head_aim_error(missing))
        self.assertIsNone(head_aim_error(nonfinite))

    def test_yaw_error_does_not_default_missing_offset_to_center(self) -> None:
        frame = sample_observation()
        frame["features"].update(target_valid=True, target_bbox_projection_valid=True,
                                 camera_horizontal_fov_degrees=90.0)
        frame["features"].pop("target_dx")
        self.assertIsNone(target_yaw_error_degrees(frame))

    def test_yaw_error_is_zero_at_center_and_signed_by_screen_side(self) -> None:
        frame = sample_observation()
        frame["features"].update(target_valid=True, target_bbox_projection_valid=True,
                                 camera_horizontal_fov_degrees=90.0)
        frame["features"]["target_dx"] = 0.0
        self.assertAlmostEqual(target_yaw_error_degrees(frame), 0.0)
        frame["features"]["target_dx"] = 0.5
        self.assertAlmostEqual(target_yaw_error_degrees(frame), 45.0)
        frame["features"]["target_dx"] = -0.5
        self.assertAlmostEqual(target_yaw_error_degrees(frame), -45.0)

    def test_yaw_error_is_unavailable_when_projection_is_invalid(self) -> None:
        frame = sample_observation()
        frame["features"].update(target_valid=True, target_bbox_projection_valid=False)
        self.assertIsNone(target_yaw_error_degrees(frame))

    def test_yaw_observation_has_fixed_finite_size_and_only_yaw_history(self) -> None:
        frame = sample_observation()
        frame["features"].update(target_valid=True, target_bbox_projection_valid=True,
                                 target_on_screen=True, target_dx=0.25, target_distance=4.0,
                                 target_bbox_width=0.2, camera_horizontal_fov_degrees=90.0)
        state = encode_yaw_observation(frame, previous_error_degrees=20.0,
                                       previous_action=0.2, prior_action=-0.1)
        self.assertEqual(len(state), 9)
        self.assertTrue(all(math.isfinite(value) for value in state))
        self.assertAlmostEqual(state[7], 0.2)

    def test_joint_observation_preserves_yaw_inputs_and_appends_pitch_state(self) -> None:
        frame = sample_observation()
        frame["features"].update(
            target_valid=True, target_bbox_projection_valid=True, target_on_screen=True,
            target_aim_projection_valid=True, target_aim_dy=0.6,
            target_dx=0.2, target_distance=4.0, target_bbox_width=0.2,
            camera_horizontal_fov_degrees=90.0,
        )
        yaw_state = encode_yaw_observation(frame, 20.0, 0.2, -0.1)
        joint_state = encode_yaw_pitch_observation(frame, 20.0, 0.2, -0.1, 0.05, -0.3, 0.1)
        self.assertEqual(len(joint_state), 14)
        self.assertEqual(joint_state[:9], yaw_state)
        self.assertAlmostEqual(joint_state[9], 1.0)
        self.assertAlmostEqual(joint_state[10], 0.2)
        self.assertAlmostEqual(joint_state[11], 0.5)
        self.assertAlmostEqual(joint_state[12], -0.3)
        self.assertTrue(all(math.isfinite(value) for value in joint_state))

    def test_pitch_only_observation_encodes_vertical_error_and_history(self) -> None:
        frame = sample_observation()
        frame["features"].update(
            target_valid=True, target_on_screen=True, target_aim_projection_valid=True,
            target_aim_dy=0.6, target_bbox_width=0.4, target_distance=8.0,
        )
        state = encode_pitch_observation(frame, 0.05, 0.2, -0.1)
        self.assertEqual(len(state), 9)
        self.assertEqual(state[0:2], (1.0, 1.0))
        self.assertAlmostEqual(state[2], 0.2)
        self.assertAlmostEqual(state[3], 0.5)
        self.assertAlmostEqual(state[5], 0.4)
        self.assertAlmostEqual(state[6], 0.25)
        self.assertAlmostEqual(state[7], 0.2)
        self.assertTrue(all(math.isfinite(value) for value in state))

    def test_pitch_only_reward_uses_head_error_not_yaw_or_crosshair(self) -> None:
        before = sample_observation()
        improved = sample_observation()
        worsened = sample_observation()
        for frame, aim_y, dx, crosshair in (
            (before, 0.7, -0.4, False),
            (improved, 0.6, 0.4, True),
            (worsened, 0.8, 0.0, True),
        ):
            frame["features"].update(
                target_valid=True, target_aim_projection_valid=True, target_aim_dy=aim_y,
                target_bbox_projection_valid=True, target_dx=dx,
                camera_horizontal_fov_degrees=90.0, crosshair_on_target=crosshair,
            )
        reward_improved = pitch_transition_reward(before, improved, 0.0, 0.0)
        reward_worsened = pitch_transition_reward(before, worsened, 0.0, 0.0)
        self.assertGreater(reward_improved, reward_worsened)

        yaw_and_hit_changed_only = sample_observation()
        yaw_and_hit_changed_only["features"].update(
            before["features"], target_dx=0.4, crosshair_on_target=True,
        )
        stationary_pitch_reward = pitch_transition_reward(before, before, 0.0, 0.0)
        changed_other_features_reward = pitch_transition_reward(
            before, yaw_and_hit_changed_only, 0.0, 0.0
        )
        self.assertAlmostEqual(changed_other_features_reward, stationary_pitch_reward)

    def test_yaw_reward_prefers_centering_and_quiet_actions(self) -> None:
        before = sample_observation()
        after_better = sample_observation()
        after_worse = sample_observation()
        for frame in (before, after_better, after_worse):
            frame["features"].update(target_valid=True, target_bbox_projection_valid=True,
                                     camera_horizontal_fov_degrees=90.0, target_on_screen=True)
        before["features"]["target_dx"] = 0.2
        after_better["features"]["target_dx"] = 0.1
        after_worse["features"]["target_dx"] = 0.3
        reward_better = yaw_transition_reward(before, after_better, 0.1, 0.1)
        reward_worse = yaw_transition_reward(before, after_worse, 0.1, 0.1)
        reward_jitter = yaw_transition_reward(after_better, after_better, 0.7, 0.0)
        reward_still = yaw_transition_reward(after_better, after_better, 0.0, 0.0)
        self.assertGreater(reward_better, reward_worse)
        self.assertGreater(reward_still, reward_jitter)

    def test_yaw_reward_adds_confirmed_crosshair_hit_bonus(self) -> None:
        before = sample_observation()
        on_target = sample_observation()
        off_target = sample_observation()
        for frame in (before, on_target, off_target):
            frame["features"].update(target_valid=True, target_bbox_projection_valid=True,
                                     camera_horizontal_fov_degrees=90.0, target_on_screen=True,
                                     target_dx=0.0)
        on_target["features"]["crosshair_on_target"] = True
        off_target["features"]["crosshair_on_target"] = False
        reward_hit = yaw_transition_reward(before, on_target, 0.0, 0.0)
        reward_miss = yaw_transition_reward(before, off_target, 0.0, 0.0)
        self.assertAlmostEqual(reward_hit - reward_miss, 0.08)

    def test_joint_pitch_reward_favors_moving_projected_head_toward_reticle(self) -> None:
        before = sample_observation()
        improved = sample_observation()
        worsened = sample_observation()
        for frame, aim_y in ((before, 0.70), (improved, 0.60), (worsened, 0.80)):
            frame["features"].update(
                target_valid=True, target_bbox_projection_valid=True, target_on_screen=True,
                target_aim_projection_valid=True, target_aim_dy=aim_y,
                target_dx=0.0, camera_horizontal_fov_degrees=90.0,
            )
        reward_improved = yaw_pitch_transition_reward(before, improved, 0.0, 0.0, 0.0, 0.0)
        reward_worsened = yaw_pitch_transition_reward(before, worsened, 0.0, 0.0, 0.0, 0.0)
        self.assertGreater(reward_improved, reward_worsened)

    def test_joint_pitch_reward_prefers_head_center_over_body_hit_plateau(self) -> None:
        def stationary_reward(aim_y: float) -> float:
            frame = sample_observation()
            frame["features"].update(
                target_valid=True, target_bbox_projection_valid=True, target_on_screen=True,
                target_aim_projection_valid=True, target_aim_dy=aim_y,
                target_dx=0.0, camera_horizontal_fov_degrees=90.0,
                crosshair_on_target=True,
            )
            return yaw_pitch_transition_reward(frame, frame, 0.0, 0.0, 0.0, 0.0)

        centered = stationary_reward(0.5)
        body_hit_but_head_off_center = stationary_reward(0.65)
        self.assertGreater(centered - body_hit_but_head_off_center, 0.15)

    def test_joint_rewards_keep_yaw_and_pitch_credit_separate(self) -> None:
        before = sample_observation()
        yaw_improved = sample_observation()
        yaw_worsened = sample_observation()
        for frame in (before, yaw_improved, yaw_worsened):
            frame["features"].update(
                target_valid=True, target_bbox_projection_valid=True, target_on_screen=True,
                target_aim_projection_valid=True, target_aim_dy=0.6,
                camera_horizontal_fov_degrees=90.0,
            )
        before["features"]["target_dx"] = 0.2
        yaw_improved["features"]["target_dx"] = 0.1
        yaw_worsened["features"]["target_dx"] = 0.3

        yaw_good, pitch_good = yaw_pitch_transition_rewards(before, yaw_improved, 0.1, 0.0, 0.0, 0.0)
        yaw_bad, pitch_bad = yaw_pitch_transition_rewards(before, yaw_worsened, 0.1, 0.0, 0.0, 0.0)
        self.assertGreater(yaw_good, yaw_bad)
        self.assertAlmostEqual(pitch_good, pitch_bad)

        pitch_improved = sample_observation()
        pitch_improved["features"].update(before["features"], target_aim_dy=0.5)
        yaw_from_pitch_only, pitch_from_pitch_only = yaw_pitch_transition_rewards(
            before, pitch_improved, 0.0, 0.0, 0.0, 0.0
        )
        yaw_without_pitch, pitch_without_pitch = yaw_pitch_transition_rewards(
            before, before, 0.0, 0.0, 0.0, 0.0
        )
        self.assertAlmostEqual(yaw_from_pitch_only, yaw_without_pitch)
        self.assertGreater(pitch_from_pitch_only, pitch_without_pitch)

    def test_joint_pitch_reward_bounds_extreme_finite_projection_errors(self) -> None:
        def progress_reward(before_y: float) -> float:
            before = sample_observation()
            after = sample_observation()
            for frame, aim_y in ((before, before_y), (after, 0.5)):
                frame["features"].update(
                    target_valid=True, target_bbox_projection_valid=True, target_on_screen=True,
                    target_aim_projection_valid=True, target_aim_dy=aim_y,
                    target_dx=0.0, camera_horizontal_fov_degrees=90.0,
                )
            return yaw_pitch_transition_reward(before, after, 0.0, 0.0, 0.0, 0.0)

        self.assertAlmostEqual(progress_reward(1.5), progress_reward(100.5))

    def test_scripted_attack_does_not_credit_knockback_to_pitch_action(self) -> None:
        before = sample_observation()
        after = sample_observation()
        for frame, aim_y in ((before, 0.8), (after, 0.5)):
            frame["features"].update(
                target_valid=True, target_bbox_projection_valid=True, target_on_screen=True,
                target_aim_projection_valid=True, target_aim_dy=aim_y,
                target_dx=0.0, camera_horizontal_fov_degrees=90.0,
            )
        _, normal_pitch_reward = yaw_pitch_transition_rewards(before, after, 0.0, 0.0, 0.0, 0.0)
        _, attack_pitch_reward = yaw_pitch_transition_rewards(
            before, after, 0.0, 0.0, 0.0, 0.0, scripted_attack=True
        )
        self.assertGreater(normal_pitch_reward, attack_pitch_reward)
        self.assertAlmostEqual(attack_pitch_reward, -0.008 * 0.0**2)

    def test_joint_pitch_reward_does_not_bonus_new_head_projection(self) -> None:
        before = sample_observation()
        remains_invalid = sample_observation()
        becomes_valid = sample_observation()
        for frame in (before, remains_invalid, becomes_valid):
            frame["features"].update(
                target_valid=True, target_bbox_projection_valid=True, target_on_screen=True,
                target_dx=0.0, camera_horizontal_fov_degrees=90.0,
            )
        before["features"]["target_aim_projection_valid"] = False
        remains_invalid["features"]["target_aim_projection_valid"] = False
        becomes_valid["features"].update(target_aim_projection_valid=True, target_aim_dy=0.4)
        reward_without_rediscovery = yaw_pitch_transition_reward(
            before, remains_invalid, 0.0, 0.0, 0.0, 0.0
        )
        reward_on_rediscovery = yaw_pitch_transition_reward(
            before, becomes_valid, 0.0, 0.0, 0.0, 0.0
        )
        self.assertAlmostEqual(reward_on_rediscovery, reward_without_rediscovery)

    def test_scripted_attack_requires_target_sword_full_cooldown_and_crosshair(self) -> None:
        frame = sample_observation()
        features = frame["features"]
        features.update(target_valid=True, mainhand_is_sword=True,
                        attack_strength=1.0, crosshair_on_target=True)
        self.assertTrue(scripted_sword_attack_eligible(frame))
        for key, value in (("target_valid", False), ("mainhand_is_sword", False),
                           ("attack_strength", 0.99), ("crosshair_on_target", False)):
            features.update(target_valid=True, mainhand_is_sword=True,
                            attack_strength=1.0, crosshair_on_target=True)
            features[key] = value
            self.assertFalse(scripted_sword_attack_eligible(frame), key)

    def test_scripted_attack_latches_one_pulse_until_gate_clears(self) -> None:
        eligible = sample_observation()
        eligible["features"].update(target_valid=True, mainhand_is_sword=True,
                                    attack_strength=1.0, crosshair_on_target=True)
        ineligible = sample_observation()
        first_press, latched = scripted_attack_decision(eligible, False)
        repeated_press, still_latched = scripted_attack_decision(eligible, latched)
        reset_press, reset_latch = scripted_attack_decision(ineligible, still_latched)
        next_press, next_latch = scripted_attack_decision(eligible, reset_latch)
        self.assertEqual((first_press, latched), (True, True))
        self.assertEqual((repeated_press, still_latched), (False, True))
        self.assertEqual((reset_press, reset_latch), (False, False))
        self.assertEqual((next_press, next_latch), (True, True))

    def test_yaw_reward_marks_target_discovery_and_penalizes_overshoot_loss(self) -> None:
        hidden = sample_observation()
        found = sample_observation()
        lost_behind = sample_observation()
        hidden["features"].update(target_valid=True, target_bbox_projection_valid=False)
        found["features"].update(target_valid=True, target_bbox_projection_valid=True,
                                 camera_horizontal_fov_degrees=90.0, target_dx=2.0, target_on_screen=False)
        lost_behind["features"].update(target_valid=True, target_bbox_projection_valid=False)
        discovery_reward = yaw_transition_reward(hidden, found, 0.0, 0.0)
        overshoot_reward = yaw_transition_reward(found, lost_behind, 0.0, 0.0)
        self.assertGreater(discovery_reward, 0.0)
        self.assertLess(overshoot_reward, 0.0)

    def test_combo_movement_observation_has_expected_finite_layout(self) -> None:
        frame = sample_observation()
        frame["features"].update(
            target_valid=True,
            target_on_screen=True,
            crosshair_on_target=True,
            target_dx=0.1,
            target_dy=-0.2,
            target_distance=8.0,
            target_horizontal_distance=7.0,
            target_relative_horizontal_speed=0.4,
            player_horizontal_speed=0.15,
            player_health=18.0,
            player_food=16,
            attack_strength=0.5,
            mainhand_is_sword=True,
            hit_combo=5,
            knockback_combo=3,
        )
        observation = encode_movement_observation(frame, True, 10, False, 2, 3)
        self.assertEqual(len(observation), MOVEMENT_OBSERVATION_SIZE)
        self.assertTrue(all(math.isfinite(value) for value in observation))
        self.assertEqual(observation[-7:], (1.0, 0.25, 0.0, 0.05, 0.5, 0.0, 0.3))

    def test_combo_movement_penalizes_key_toggles(self) -> None:
        before = sample_observation()
        after = sample_observation()
        reward, knockback_hit, incoming_hit, offset, hit_reward, *_ = movement_transition_reward(
            before, after, True, True, False, False, False, False
        )
        self.assertAlmostEqual(reward, -FORWARD_TOGGLE_PENALTY - SPRINT_TOGGLE_PENALTY)
        self.assertFalse(knockback_hit)
        self.assertFalse(incoming_hit)
        self.assertIsNone(offset)
        self.assertEqual(hit_reward, 0.0)

    def test_combo_reward_uses_existing_knockback_hit_classification(self) -> None:
        before = sample_observation()
        after = sample_observation()
        before["features"].update(has_hit_enemy=True, ticks_since_hit_enemy=1)
        after["features"].update(
            has_hit_enemy=True,
            ticks_since_hit_enemy=0,
            last_hit_knockback=True,
            hit_combo=1,
            knockback_combo=1,
        )
        after["action_history"][-1][7] = 1  # attack
        reward, knockback_hit, incoming_hit, offset, hit_reward, success, miss, *_ = movement_transition_reward(
            before, after, True, True, True, True, True, True
        )
        self.assertAlmostEqual(reward, KNOCKBACK_HIT_REWARD + ATTACK_SUCCESS_REWARD)
        self.assertTrue(knockback_hit)
        self.assertFalse(incoming_hit)
        self.assertEqual(offset, 0)
        self.assertEqual(hit_reward, KNOCKBACK_HIT_REWARD + ATTACK_SUCCESS_REWARD)
        self.assertTrue(success)
        self.assertFalse(miss)

    def test_non_knockback_hits_do_not_earn_combo_reward(self) -> None:
        before = sample_observation()
        before["features"].update(has_hit_enemy=True, ticks_since_hit_enemy=1)
        rewards = []
        for combo_count in (1, 2, 5, 20):
            after = sample_observation()
            after["features"].update(
                has_hit_enemy=True, ticks_since_hit_enemy=0, hit_combo=combo_count
            )
            after["action_history"][-1][7] = 1
            reward, knockback_hit, _, offset, hit_reward, success, *_ = movement_transition_reward(
                before, after, False, False, True, True, False, False
            )
            self.assertFalse(knockback_hit)
            self.assertEqual(offset, 0)
            self.assertEqual(reward, hit_reward)
            self.assertTrue(success)
            rewards.append(hit_reward)
            before = after
            before["features"]["ticks_since_hit_enemy"] = 1
        self.assertEqual(rewards, [ATTACK_SUCCESS_REWARD] * 4)

    def test_only_knockback_hits_build_the_rewarded_combo(self) -> None:
        before = sample_observation()
        before["features"].update(has_hit_enemy=True, ticks_since_hit_enemy=1)
        rewards = []
        prior_knockback_combo = 0
        for combo_count in (1, 2, 3, 4, 5, 6):
            after = sample_observation()
            after["features"].update(
                has_hit_enemy=True, ticks_since_hit_enemy=0, hit_combo=combo_count,
                last_hit_knockback=True, knockback_combo=combo_count,
            )
            after["action_history"][-1][7] = 1
            reward, knockback_hit, _, offset, hit_reward, *result_flags = movement_transition_reward(
                before, after, False, False, True, True, False, False,
                prior_knockback_combo=prior_knockback_combo,
            )
            self.assertTrue(knockback_hit)
            self.assertEqual(offset, 0)
            self.assertEqual(reward, hit_reward)
            rewards.append(hit_reward)
            prior_knockback_combo = result_flags[-1]
            before = after
            before["features"]["ticks_since_hit_enemy"] = 1
        self.assertEqual(rewards[0], ATTACK_SUCCESS_REWARD + KNOCKBACK_HIT_REWARD)
        self.assertEqual(
            rewards[1], ATTACK_SUCCESS_REWARD + KNOCKBACK_HIT_REWARD + COMBO_REWARD_PER_EXTRA_HIT
        )
        self.assertEqual(rewards[-1], ATTACK_SUCCESS_REWARD + KNOCKBACK_HIT_REWARD + MAX_COMBO_REWARD)

    def test_combo_reward_penalizes_each_new_incoming_hit_once(self) -> None:
        before = sample_observation()
        hit = sample_observation()
        later = sample_observation()
        before["features"].update(has_been_hit=True, ticks_since_hit_by_enemy=1)
        hit["features"].update(has_been_hit=True, ticks_since_hit_by_enemy=0)
        later["features"].update(has_been_hit=True, ticks_since_hit_by_enemy=1)
        hit_reward, _, first_hit, _, _, *_ = movement_transition_reward(
            before, hit, False, False, False, False, False, False
        )
        later_reward, _, repeated_hit, _, _, *_ = movement_transition_reward(
            hit, later, False, False, False, False, False, False
        )
        self.assertEqual(hit_reward, -INCOMING_HIT_PENALTY)
        self.assertTrue(first_hit)
        self.assertEqual(later_reward, 0.0)
        self.assertFalse(repeated_hit)

    def test_learned_attack_rewards_success_and_penalizes_misses(self) -> None:
        before = sample_observation()
        after = sample_observation()
        eligible_click = movement_transition_reward(
            before, after, False, False, True, True, False, False
        )
        self.assertEqual(eligible_click[0], -ATTACK_MISS_PENALTY)
        self.assertTrue(eligible_click[6])

        missed_ground_truth = movement_transition_reward(
            before, after, False, False, False, True, False, False
        )
        self.assertEqual(missed_ground_truth[0], -MISSED_ELIGIBLE_PENALTY)
        self.assertTrue(missed_ground_truth[8])

        false_click = movement_transition_reward(
            before, after, False, False, True, False, False, False
        )
        self.assertEqual(false_click[0], -FALSE_ATTACK_PENALTY)
        self.assertTrue(false_click[7])

        before["features"].update(
            target_valid=True, mainhand_is_sword=True, crosshair_on_target=True,
            attack_strength=0.5,
        )
        premature_click = movement_transition_reward(
            before, after, False, False, True, False, False, False
        )
        self.assertEqual(premature_click[0], -PREMATURE_ATTACK_PENALTY)
        self.assertTrue(premature_click[9])

    def test_hit_reward_requires_ground_truth_eligible_attack_action(self) -> None:
        before = sample_observation()
        after = sample_observation()
        before["features"].update(
            has_hit_enemy=True, ticks_since_hit_enemy=1,
            target_valid=True, mainhand_is_sword=True, crosshair_on_target=True,
            attack_strength=0.5,
        )
        after["features"].update(
            has_hit_enemy=True, ticks_since_hit_enemy=0, last_hit_knockback=True,
            hit_combo=3, knockback_combo=3,
        )
        after["action_history"][-1][7] = 1
        reward, knockback_hit, _, _, hit_reward, *_ = movement_transition_reward(
            before, after, False, False, True, False, False, False
        )
        self.assertFalse(knockback_hit)
        self.assertEqual(hit_reward, 0.0)
        self.assertEqual(reward, -PREMATURE_ATTACK_PENALTY)

    def test_delayed_hit_uses_the_ground_truth_status_of_its_original_click(self) -> None:
        before = sample_observation()
        after = sample_observation()
        before["features"].update(has_hit_enemy=True, ticks_since_hit_enemy=2)
        after["features"].update(
            has_hit_enemy=True, ticks_since_hit_enemy=1, last_hit_knockback=True,
            hit_combo=4, knockback_combo=4,
        )
        after["action_history"][-2][7] = 1
        reward, knockback_hit, _, offset, hit_reward, *tail = movement_transition_reward(
            before, after, False, False, False, False, False, False,
            prior_attack_context=((True, False),),
            prior_knockback_combo=3,
        )
        self.assertEqual(offset, 1)
        self.assertFalse(knockback_hit)
        self.assertEqual(hit_reward, 0.0)
        self.assertEqual(reward, 0.0)
        self.assertEqual(tail[-1], 0)

    def test_non_knockback_hit_breaks_the_knockback_combo_streak(self) -> None:
        before = sample_observation()
        after = sample_observation()
        before["features"].update(has_hit_enemy=True, ticks_since_hit_enemy=1)
        after["features"].update(has_hit_enemy=True, ticks_since_hit_enemy=0)
        after["action_history"][-1][7] = 1
        result = movement_transition_reward(
            before, after, False, False, True, True, False, False,
            prior_knockback_combo=4,
        )
        self.assertFalse(result[1])
        self.assertEqual(result[0], ATTACK_SUCCESS_REWARD)
        self.assertEqual(result[-1], 0)

    def test_rejects_on_screen_without_projection(self) -> None:
        frame = sample_observation()
        frame["features"].update(target_valid=True, target_on_screen=True,
                                 target_bbox_projection_valid=False)
        self.assertTrue(any("target_on_screen" in error for error in validate_observation(frame)))

    def test_valid_target_geometry(self) -> None:
        frame = sample_observation()
        frame["features"].update({
            "target_valid": True, "target_bbox_projection_valid": True, "target_on_screen": True,
            "target_aim_projection_valid": True, "target_aim_dx": 0.5, "target_aim_dy": 0.4,
            "target_bbox_left": 0.4, "target_bbox_right": 0.6, "target_bbox_top": 0.2,
            "target_bbox_bottom": 0.8, "target_bbox_width": 0.2, "target_bbox_height": 0.6,
            "target_distance": 4.0, "target_horizontal_distance": 3.0, "target_height": 1.8,
            "target_health": 20.0,
        })
        self.assertEqual(validate_observation(frame), [])

    def test_behind_camera_defaults_are_consistent(self) -> None:
        frame = sample_observation()
        frame["features"].update({
            "target_valid": True, "target_bbox_projection_valid": False, "target_on_screen": False,
            "target_aim_projection_valid": False, "target_bbox_left": 0.0, "target_bbox_right": 0.0,
            "target_bbox_top": 0.0, "target_bbox_bottom": 0.0, "target_bbox_width": 0.0,
            "target_bbox_height": 0.0,
        })
        self.assertEqual(validate_observation(frame), [])

    def test_checks_action_history_shape(self) -> None:
        frame = sample_observation()
        frame["action_history"] = []
        self.assertTrue(any("exactly 20" in error for error in validate_observation(frame)))


class LiveClient:
    def __init__(self, host: str, port: int, timeout: float, wait_for_control: bool = False):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.settimeout(timeout)
        self.reader = self.sock.makefile("rb")
        self.sequence = 1
        self.observations: list[dict[str, Any]] = []
        self.acks: dict[int, dict[str, Any]] = {}
        ready = self._read_until(lambda message: message.get("type") == "ready")
        if ready.get("protocol") != PROTOCOL or ready.get("observation_schema") != SCHEMA:
            raise RuntimeError(f"unexpected ready message: {ready}")
        if not ready.get("ai_control_enabled") and wait_for_control:
            print("AI control is OFF; waiting for it to be enabled in Minecraft...", flush=True)
            ready = self._read_until(
                lambda message: message.get("type") == "control_state"
                and bool(message.get("ai_control_enabled"))
            )
        if not ready.get("ai_control_enabled") and ready.get("type") == "ready":
            raise RuntimeError("AI control is OFF; press K in the client before running live tests")

    def close(self) -> None:
        try:
            self.reader.close()
        finally:
            self.sock.close()

    def _read_message(self) -> dict[str, Any]:
        line = self.reader.readline(65537)
        if not line:
            raise ConnectionError("MinecraftAI closed the test connection")
        if len(line) > 65536 or not line.endswith(b"\n"):
            raise ValueError("observation protocol line exceeded 65536 bytes")
        message = json.loads(line)
        if not isinstance(message, dict):
            raise ValueError("protocol message must be a JSON object")
        if message.get("type") == "observation":
            self.observations.append(message)
        elif message.get("type") == "ack":
            self.acks[int(message.get("seq", -1))] = message
        elif message.get("type") == "control_state" and not message.get("ai_control_enabled"):
            raise RuntimeError("AI control was switched OFF during the live test")
        return message

    def _read_until(self, predicate: Any) -> dict[str, Any]:
        while True:
            message = self._read_message()
            if predicate(message):
                return message

    def next_observation(self, after_tick: int = -1, applied_sequence: int | None = None) -> dict[str, Any]:
        deadline = time.monotonic() + self.sock.gettimeout()
        while True:
            eligible = [
                frame for frame in self.observations
                if int(frame.get("tick", -1)) > after_tick
                and (applied_sequence is None or int(frame.get("features", {}).get(
                    "remote_control_diagnostics", {}
                ).get("last_applied_sequence", -1)) >= applied_sequence)
            ]
            if eligible:
                latest = max(eligible, key=lambda frame: int(frame["tick"]))
                self.observations = [frame for frame in self.observations if int(frame.get("tick", -1)) > int(latest["tick"])]
                return latest
            if time.monotonic() >= deadline:
                raise TimeoutError(f"no observation newer than tick {after_tick}")
            self._read_message()

    def send_action(self, frame: dict[str, Any], **overrides: Any) -> dict[str, Any]:
        action: dict[str, Any] = {key: False for key in ACTION_KEYS}
        action.update({"yaw_delta": 0.0, "pitch_delta": 0.0})
        action.update(overrides)
        command = {"type": "control", "seq": self.sequence, "tick": int(frame["tick"]), **action}
        current_sequence = self.sequence
        self.sequence += 1
        self.sock.sendall((json.dumps(command, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8"))
        deadline = time.monotonic() + self.sock.gettimeout()
        while current_sequence not in self.acks:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"no acknowledgement for sequence {current_sequence}")
            self._read_message()
        ack = self.acks.pop(current_sequence)
        if not ack.get("accepted"):
            raise RuntimeError(f"action rejected: {ack.get('error')}")
        return self.next_observation(after_tick=int(frame["tick"]), applied_sequence=current_sequence)


def _check_live_frame(frame: dict[str, Any], label: str) -> None:
    errors = validate_observation(frame)
    if errors:
        raise AssertionError(f"{label}: " + "; ".join(errors))


def _wrap_degrees(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def run_live(args: argparse.Namespace) -> int:
    client = LiveClient(args.host, args.port, args.timeout)
    latest: dict[str, Any] | None = None
    initial_yaw = initial_pitch = 0.0
    results: list[str] = []
    try:
        latest = client.next_observation()
        _check_live_frame(latest, "initial frame")
        initial_yaw = float(latest["features"]["camera_yaw"])
        initial_pitch = float(latest["features"]["camera_pitch"])
        results.append("PASS initial observation schema, feature types/ranges, projection invariants, and 20-row action history")

        for sample_index in range(args.samples - 1):
            latest = client.next_observation(int(latest["tick"]))
            _check_live_frame(latest, f"observation sample {sample_index + 2}")
        results.append(f"PASS validated {args.samples} live observations")
        if latest["features"].get("target_valid"):
            results.append(
                "PASS target found; bbox projection valid="
                f"{latest['features']['target_bbox_projection_valid']}, on-screen={latest['features']['target_on_screen']}, "
                f"head-point projection valid={latest['features']['target_aim_projection_valid']}"
            )
        else:
            results.append("NOTE no visible target selected; target geometry is validated as the no-target case")
            if args.require_target:
                raise AssertionError("--require-target was set but no line-of-sight target was selected")

        if args.actions:
            # Camera controls are measured from the actual next observation.
            start_yaw = float(latest["features"]["camera_yaw"])
            latest = client.send_action(latest, yaw_delta=2.0)
            yaw_change = _wrap_degrees(float(latest["features"]["camera_yaw"]) - start_yaw)
            if abs(yaw_change - 2.0) > args.rotation_tolerance:
                raise AssertionError(f"positive yaw delta: expected +2°, observed {yaw_change:+.3f}°")
            latest = client.send_action(latest, yaw_delta=-2.0)
            results.append("PASS positive and negative yaw deltas match camera observations")

            start_pitch = float(latest["features"]["camera_pitch"])
            for delta in (2.0, -2.0):
                expected = max(-90.0, min(90.0, start_pitch + delta))
                latest = client.send_action(latest, pitch_delta=delta)
                observed = float(latest["features"]["camera_pitch"])
                if abs(observed - expected) > args.rotation_tolerance:
                    raise AssertionError(f"pitch delta {delta:+.1f}°: expected {expected:.3f}°, observed {observed:.3f}°")
                start_pitch = observed
            results.append("PASS positive and negative pitch deltas match camera observations and pitch clamp")

            probe_actions = ["forward", "backward", "left", "right", "jump", "sneak", "sprint"]
            if args.allow_interactions:
                probe_actions.extend(("attack", "use"))
            for action_name in probe_actions:
                before = latest
                clicks_before = before["features"]["remote_control_diagnostics"]["consumed_clicks"]
                latest = client.send_action(before, **{action_name: True})
                _check_live_frame(latest, f"{action_name} action")
                state = latest["features"]["remote_control_diagnostics"]["action_state"]
                if state[action_name] is not True:
                    raise AssertionError(f"{action_name} was accepted but not observed as active by the mod")
                history_row = latest["action_history"][-1]
                index = ACTION_HISTORY_INDEX[action_name]
                if history_row[index] not in (1, True):
                    raise AssertionError(f"{action_name} missing from action_history")
                if action_name in ("attack", "use"):
                    clicks_after = latest["features"]["remote_control_diagnostics"]["consumed_clicks"]
                    if clicks_after[action_name] <= clicks_before[action_name]:
                        raise AssertionError(f"{action_name} reached the key state but consumeClick did not register it")
                latest = client.send_action(latest)
                if latest["features"]["remote_control_diagnostics"]["action_state"][action_name] is not False:
                    raise AssertionError(f"{action_name} did not release after a neutral control")
                results.append(f"PASS {action_name} press, action-history entry, and release")

            # Restore the camera to its original orientation even if the probes moved it.
            current_yaw = float(latest["features"]["camera_yaw"])
            yaw_restore = _wrap_degrees(initial_yaw - current_yaw)
            current_pitch = float(latest["features"]["camera_pitch"])
            pitch_restore = max(-90.0, min(90.0, initial_pitch)) - current_pitch
            latest = client.send_action(latest, yaw_delta=yaw_restore, pitch_delta=pitch_restore)
            latest = client.send_action(latest)
            results.append("PASS neutral control released all remote actions; camera orientation restored")
        else:
            results.append("NOTE action probes skipped; rerun with --actions in a safe test world")

        for result in results:
            print(result)
        return 0
    finally:
        if latest is not None:
            try:
                # Best-effort release if a live assertion fails mid-probe.
                latest = client.send_action(latest)
            except (OSError, RuntimeError, TimeoutError, ValueError, KeyError):
                pass
        client.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="connect to a running client and inspect real observations")
    parser.add_argument("--actions", action="store_true", help="probe camera and non-combat key actions; can move/jump the player")
    parser.add_argument("--allow-interactions", action="store_true", help="also test attack/use clicks; may hit entities or use the held item")
    parser.add_argument("--require-target", action="store_true", help="fail live mode when no line-of-sight target is selected")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--samples", type=int, default=50, help="number of live observations to validate")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--rotation-tolerance", type=float, default=0.25)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.samples < 1 or args.timeout <= 0 or args.rotation_tolerance < 0:
        parser.error("samples and timeout must be positive; rotation tolerance cannot be negative")
    if args.allow_interactions and not args.actions:
        parser.error("--allow-interactions requires --actions")
    if not args.live:
        if args.actions or args.allow_interactions or args.require_target:
            parser.error("live-only options require --live")
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(ObservationContractTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1
    try:
        return run_live(args)
    except (OSError, RuntimeError, TimeoutError, ValueError, AssertionError, KeyError, json.JSONDecodeError) as error:
        print(f"FAIL {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
