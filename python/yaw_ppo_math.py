"""Small dependency-free observation and reward helpers for yaw/pitch PPO."""

from __future__ import annotations

import math
from typing import Any

from aim_geometry import head_aim_error, target_yaw_error_degrees

MAX_YAW_DELTA_DEGREES = 10.0
YAW_OBSERVATION_SIZE = 9
PITCH_ONLY_OBSERVATION_SIZE = 9
PITCH_DELTA_OBSERVATION_SIZE = 5
JOINT_OBSERVATION_SIZE = YAW_OBSERVATION_SIZE + PITCH_DELTA_OBSERVATION_SIZE
MAX_PITCH_DELTA_DEGREES = 8.0
PITCH_BOX_PROGRESS_SCALE = 0.12
PITCH_BOX_PROGRESS_LIMIT = 0.20
PITCH_BBOX_OVERLAP_BONUS = 0.10
PITCH_LEVEL_REWARD = PITCH_BBOX_OVERLAP_BONUS
PITCH_LEVEL_SCALE_DEGREES = 10.0
JOINT_CROSSHAIR_BONUS = 0.02


def encode_pitch_observation(
    frame: dict[str, Any],
    previous_pitch_error: float | None,
    previous_pitch_action: float,
    prior_pitch_action: float,
) -> tuple[float, ...]:
    """Encode pitch state using the same 9-slot layout as the yaw backbone."""
    features = frame.get("features", {})
    error = head_aim_error(frame)
    error_rate = 0.0
    if error is not None and previous_pitch_error is not None:
        error_rate = max(-1.0, min(1.0, (error - previous_pitch_error) * 10.0))
    width = float(features.get("target_bbox_width", 0.0))
    distance = float(features.get("target_distance", 0.0))
    try:
        camera_pitch = float(features.get("camera_pitch", 0.0))
    except (TypeError, ValueError):
        camera_pitch = 0.0
    if not math.isfinite(camera_pitch):
        camera_pitch = 0.0
    return (
        float(bool(features.get("target_valid", False))),
        float(error is not None),
        0.0 if error is None else max(-1.0, min(1.0, error / 0.5)),
        error_rate,
        float(bool(features.get("target_on_screen", False))),
        max(0.0, min(1.0, width)),
        max(0.0, min(1.0, distance / 32.0)),
        max(-1.0, min(1.0, previous_pitch_action)),
        max(-1.0, min(1.0, camera_pitch / 90.0)),
    )


def pitch_transition_reward(
    before: dict[str, Any],
    after: dict[str, Any],
    normalized_pitch_action: float,
    previous_pitch_action: float,
) -> float:
    """Reward vertical bounding-box alignment and level pitch."""
    action = max(-1.0, min(1.0, float(normalized_pitch_action)))
    previous = max(-1.0, min(1.0, float(previous_pitch_action)))
    reward = -0.008 * action * action - 0.035 * (action - previous) ** 2
    return reward + _pitch_target_reward(before, after) + _pitch_level_reward(after)


def _vertical_crosshair_box_distance(frame: dict[str, Any]) -> float | None:
    """Return screen-space vertical distance from crosshair center to target box."""
    features = frame.get("features", {})
    if not bool(features.get("target_valid", False)) or not bool(features.get("target_on_screen", False)):
        return None
    try:
        top = float(features["target_bbox_top"])
        bottom = float(features["target_bbox_bottom"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(top) or not math.isfinite(bottom) or not 0.0 <= top <= bottom <= 1.0:
        return None
    return max(top - 0.5, 0.5 - bottom, 0.0)


def _pitch_target_reward(before: dict[str, Any], after: dict[str, Any]) -> float:
    """Shape pitch toward the target box and reward keeping it in the box."""
    reward = 0.0
    before_distance = _vertical_crosshair_box_distance(before)
    after_distance = _vertical_crosshair_box_distance(after)
    if before_distance is not None and after_distance is not None:
        progress = (before_distance - after_distance) / PITCH_BOX_PROGRESS_SCALE
        reward += max(-PITCH_BOX_PROGRESS_LIMIT, min(PITCH_BOX_PROGRESS_LIMIT, progress))
    if after_distance == 0.0:
        reward += PITCH_BBOX_OVERLAP_BONUS
    return reward


def _pitch_level_reward(frame: dict[str, Any]) -> float:
    """Reward every applied tick for keeping camera pitch near the horizon."""
    try:
        pitch = float(frame.get("features", {}).get("camera_pitch"))
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(pitch):
        return 0.0
    return PITCH_LEVEL_REWARD * math.exp(-abs(pitch) / PITCH_LEVEL_SCALE_DEGREES)


def encode_yaw_observation(
    frame: dict[str, Any],
    previous_error_degrees: float | None,
    previous_action: float,
    prior_action: float,
) -> tuple[float, ...]:
    """Build a compact observation with no pitch or non-aim action features."""
    features = frame.get("features", {})
    error = target_yaw_error_degrees(frame)
    error_rate = 0.0
    if error is not None and previous_error_degrees is not None:
        error_rate = max(-1.0, min(1.0, (error - previous_error_degrees) / MAX_YAW_DELTA_DEGREES))
    width = float(features.get("target_bbox_width", 0.0))
    distance = float(features.get("target_distance", 0.0))
    return (
        float(bool(features.get("target_valid", False))),
        float(error is not None),
        0.0 if error is None else max(-1.0, min(1.0, error / 90.0)),
        error_rate,
        float(bool(features.get("target_on_screen", False))),
        max(0.0, min(1.0, width)),
        max(0.0, min(1.0, distance / 32.0)),
        max(-1.0, min(1.0, previous_action)),
        max(-1.0, min(1.0, previous_action - prior_action)) / 2.0,
    )


def encode_yaw_pitch_observation(
    frame: dict[str, Any],
    previous_yaw_error_degrees: float | None,
    previous_yaw_action: float,
    prior_yaw_action: float,
    previous_pitch_error: float | None,
    previous_pitch_action: float,
    prior_pitch_action: float,
) -> tuple[float, ...]:
    """Keep the trained yaw inputs intact and append head-pitch error/history."""
    yaw_features = encode_yaw_observation(
        frame, previous_yaw_error_degrees, previous_yaw_action, prior_yaw_action
    )
    pitch_error = head_aim_error(frame)
    pitch_error_rate = 0.0
    if pitch_error is not None and previous_pitch_error is not None:
        pitch_error_rate = max(-1.0, min(1.0, (pitch_error - previous_pitch_error) * 10.0))
    return yaw_features + (
        float(pitch_error is not None),
        0.0 if pitch_error is None else max(-1.0, min(1.0, pitch_error / 0.5)),
        pitch_error_rate,
        max(-1.0, min(1.0, previous_pitch_action)),
        max(-1.0, min(1.0, previous_pitch_action - prior_pitch_action)) / 2.0,
    )


def yaw_transition_reward(
    before: dict[str, Any],
    after: dict[str, Any],
    normalized_action: float,
    previous_action: float,
    *,
    crosshair_bonus: float = 0.08,
) -> float:
    """Reward horizontal centering progress and discourage jitter/large inputs."""
    action = max(-1.0, min(1.0, float(normalized_action)))
    previous = max(-1.0, min(1.0, float(previous_action)))
    reward = -0.008 * action * action - 0.035 * (action - previous) ** 2

    before_error = target_yaw_error_degrees(before)
    after_error = target_yaw_error_degrees(after)
    if after_error is None:
        if before_error is not None and bool(after.get("features", {}).get("target_valid", False)):
            # The visible target crossed behind the camera. A modest penalty
            # teaches the controller not to lose it through overshoot.
            reward -= 0.08
        return reward

    if before_error is not None:
        reward += 1.5 * (abs(before_error) - abs(after_error)) / MAX_YAW_DELTA_DEGREES
    else:
        # A fresh, projectable target is a useful discovery event, even when
        # its center is still outside the viewport and yields no progress yet.
        reward += 0.05
    # An absolute centering term supplies a gradient even when actions make
    # only tiny progress, and gives a small positive target for stable aim.
    reward += 0.06 * math.exp(-abs(after_error) / 2.0)
    if bool(after.get("features", {}).get("target_on_screen", False)):
        reward += 0.01
    if bool(after.get("features", {}).get("crosshair_on_target", False)) and crosshair_bonus > 0.0:
        # Reward a confirmed reticle hit, independent of the projected box.
        # Keep this modest so horizontal centering remains the main signal.
        reward += crosshair_bonus
    return reward


def scripted_sword_attack_eligible(frame: dict[str, Any]) -> bool:
    """Whether the trainer may trigger its one-shot scripted sword attack."""
    features = frame.get("features", {})
    try:
        attack_strength = float(features.get("attack_strength", 0.0))
    except (TypeError, ValueError):
        return False
    return (
        bool(features.get("target_valid", False))
        and bool(features.get("mainhand_is_sword", False))
        and math.isfinite(attack_strength)
        and attack_strength >= 1.0
        and bool(features.get("crosshair_on_target", False))
    )


def scripted_attack_decision(
    frame: dict[str, Any], attack_latched: bool
) -> tuple[bool, bool]:
    """Return (send_attack_pulse, new_latch_state), firing once per eligibility."""
    eligible = scripted_sword_attack_eligible(frame)
    return eligible and not attack_latched, eligible


def yaw_pitch_transition_rewards(
    before: dict[str, Any],
    after: dict[str, Any],
    normalized_yaw_action: float,
    previous_yaw_action: float,
    normalized_pitch_action: float,
    previous_pitch_action: float,
    *,
    scripted_attack: bool = False,
) -> tuple[float, float]:
    """Return independent yaw and pitch rewards for factorized policy updates."""
    yaw_reward = yaw_transition_reward(
        before, after, normalized_yaw_action, previous_yaw_action,
        crosshair_bonus=JOINT_CROSSHAIR_BONUS,
    )
    pitch_action = max(-1.0, min(1.0, float(normalized_pitch_action)))
    previous_pitch = max(-1.0, min(1.0, float(previous_pitch_action)))
    pitch_reward = -0.008 * pitch_action * pitch_action
    pitch_reward -= 0.035 * (pitch_action - previous_pitch) ** 2

    pitch_reward += _pitch_target_reward(before, after)
    pitch_reward += _pitch_level_reward(after)
    return yaw_reward, pitch_reward


def yaw_pitch_transition_reward(
    before: dict[str, Any],
    after: dict[str, Any],
    normalized_yaw_action: float,
    previous_yaw_action: float,
    normalized_pitch_action: float,
    previous_pitch_action: float,
) -> float:
    """Return the combined reward for diagnostics and backwards compatibility."""
    return sum(yaw_pitch_transition_rewards(
        before, after, normalized_yaw_action, previous_yaw_action,
        normalized_pitch_action, previous_pitch_action,
    ))
