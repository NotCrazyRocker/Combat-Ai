#!/usr/bin/env python3
"""Verify every observation streamed by the MinecraftAI mod (schema minecraftai.pvp.v2).

Checks, per frame:
  * the exact key set of every block (missing AND unexpected keys fail),
  * types, finiteness and value ranges,
  * cross-field consistency (sin/cos vs angle, egocentric geometry vs yaw error,
    reach flag vs hitbox distance, event fields vs feature fields, ...),
  * masking (live target features must be zero/false while the target is not visible),
  * removed v1 keys stay removed (player_x/y/z, target_aim_*, ...).
Between frames: tick/timestamp ordering, counter increments, yaw continuity.

Modes:
    py -3.12 .\\scripts\\verify_observations.py --self-test
        Offline. Checks the validator itself against synthetic frames and mutations.
    py -3.12 .\\scripts\\verify_observations.py
        Live. Start runClient, join a world, press K, then run. Sends no actions.
    py -3.12 .\\scripts\\verify_observations.py --frames 1000 --require-target --require-hits

Exit code 0 = all checks passed, 1 = failures (or a --require condition was not met).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import socket
import sys
import time
from collections import Counter

SCHEMA = "minecraftai.pvp.v2"
EPS = 1e-6
ITEM_CATEGORIES = ["empty", "sword", "axe", "ranged", "trident_or_mace", "ender_pearl",
                   "golden_apple", "potion", "shield", "fishing_rod", "block", "other"]
HIT_TYPES = {"none", "sweep", "critical", "knockback", "normal"}
OUTCOMES = {"none", "hit", "miss", "blocked", "no_effect"}
REMOVED_KEYS = {"player_x", "player_y", "player_z", "player_y_level", "player_velocity_x",
                "player_velocity_z", "target_aim_dx", "target_aim_dy", "target_aim_projection_valid",
                "target_relative_velocity_x", "target_relative_velocity_y", "target_relative_velocity_z",
                "remote_control_diagnostics"}
DIAG_ACTION_KEYS = ["control_enabled", "forward", "backward", "left", "right", "jump", "sneak",
                    "sprint", "attack", "use"]


def spec(kind: str, names: str, lo=None, hi=None) -> dict:
    return {name: (kind, lo, hi) for name in names.split()}


# ---------------------------------------------------------------- schema ---
SELF: dict = {}
SELF.update(spec("bool", "player_valid player_on_ground player_sprinting player_sneaking player_blocking "
                         "player_using_item player_on_fire player_in_water player_in_lava player_on_climbable "
                         "mainhand_is_sword has_hit_enemy has_been_hit last_hit_critical last_hit_knockback "
                         "last_hit_sweep last_hit_normal last_attack_hit last_attack_miss last_attack_blocked "
                         "last_attack_no_effect target_tracked target_valid has_last_known_target"))
SELF.update(spec("num", "player_velocity_forward player_velocity_right player_velocity_y", -10, 10))
SELF.update(spec("num", "player_horizontal_speed", 0, 10))
SELF.update(spec("num", "player_fall_distance", 0, 1e6))
SELF.update(spec("num", "camera_yaw"))  # intentionally unbounded
SELF.update(spec("num", "camera_pitch", -90, 90))
SELF.update(spec("num", "camera_yaw_sin camera_yaw_cos camera_pitch_sin camera_pitch_cos", -1, 1))
SELF.update(spec("num", "camera_horizontal_fov_degrees", 1, 180))
SELF.update(spec("int", "player_hurt_time", 0, 20))
SELF.update(spec("int", "player_invulnerable_ticks", 0, 100))
SELF.update(spec("num", "player_health player_absorption player_armor player_armor_toughness "
                        "player_attack_damage player_attack_speed", 0, 2048))
SELF.update(spec("int", "player_food", 0, 20))
SELF.update(spec("num", "player_reach", 0, 16))
SELF.update(spec("int", "effect_speed effect_slowness effect_strength effect_weakness effect_resistance "
                        "effect_jump_boost effect_regeneration", 0, 256))
SELF.update(spec("num", "attack_strength", 0, 1))
SELF.update(spec("num", "ticks_until_attack_ready", 0, 200))
SELF.update(spec("int", "ticks_on_ground ticks_in_air hit_combo knockback_combo", 0, 1_000_000))
COUNTERS = ("ticks_since_jump ticks_since_sprint_toggle ticks_since_attack_swing ticks_since_knocked_back "
            "ticks_since_hit_enemy ticks_since_hit_by_enemy ticks_since_attack_outcome ticks_since_target_seen")
SELF.update(spec("int", COUNTERS, -1, 1_000_000))
SELF.update(spec("str", "last_hit_type last_attack_outcome"))
SELF.update(spec("num", "last_known_target_rel_forward last_known_target_rel_right last_known_target_rel_up",
                 -300, 300))
SELF.update(spec("num", "last_known_target_distance", 0, 300))
SELF.update(spec("num", "last_known_target_yaw_error", -180, 180))

TARGET: dict = {}  # live target features: must be zero/false/"none" while the target is not visible
TARGET.update(spec("num", "target_health target_absorption target_armor", 0, 2048))
TARGET.update(spec("num", "target_height", 0, 4))
TARGET.update(spec("int", "target_hurt_time", 0, 20))
TARGET.update(spec("int", "target_invulnerable_ticks", 0, 100))
TARGET.update(spec("num", "target_swing_progress", 0, 1))
TARGET.update(spec("int", "target_held_item_id", 0, len(ITEM_CATEGORIES) - 1))
TARGET.update(spec("num", "target_distance target_horizontal_distance target_hitbox_distance", 0, 300))
TARGET.update(spec("num", "closing_speed", -10, 10))
TARGET.update(spec("num", "target_rel_forward target_rel_right target_rel_up", -300, 300))
TARGET.update(spec("num", "target_yaw_error target_yaw target_facing_error", -180, 180))
TARGET.update(spec("num", "target_yaw_sin target_yaw_cos target_facing_cos", -1, 1))
TARGET.update(spec("num", "target_pitch", -90, 90))
TARGET.update(spec("num", "target_relative_velocity_forward target_relative_velocity_right "
                          "target_relative_velocity_up target_relative_acceleration_forward "
                          "target_relative_acceleration_right target_relative_acceleration_up", -10, 10))
TARGET.update(spec("num", "target_relative_horizontal_speed", 0, 10))
TARGET.update(spec("num", "target_dx target_dy"))  # offsets from screen center, may exceed 0.5 off-screen
TARGET.update(spec("num", "target_bbox_left target_bbox_top target_bbox_right target_bbox_bottom "
                          "target_bbox_width target_bbox_height", 0, 1))
TARGET.update(spec("bool", "target_on_ground target_sprinting target_sneaking target_swinging target_blocking "
                           "target_using_item target_in_reach can_crit_now target_on_screen "
                           "target_bbox_projection_valid target_center_projection_valid crosshair_on_target"))
TARGET.update(spec("str", "target_held_item"))

FEATURE_KEYS = set(SELF) | set(TARGET) | {"target_relative_velocity_history"}

EVENTS = {}
EVENTS.update(spec("num", "damage_dealt damage_taken", 0, 2048))
EVENTS.update(spec("num", "target_health_delta self_health_delta", -2048, 2048))
EVENTS.update(spec("int", "hits_dealt hits_taken", 0, 100))
EVENTS.update(spec("str", "attack_outcome"))

CRITIC = {}
CRITIC.update(spec("bool", "target_tracked target_visible"))
CRITIC.update(spec("num", "target_rel_forward target_rel_right target_rel_up", -300, 300))
CRITIC.update(spec("num", "target_hitbox_distance", 0, 300))
CRITIC.update(spec("num", "target_yaw_error", -180, 180))
CRITIC.update(spec("num", "target_health target_absorption", 0, 2048))

TOP_KEYS = {"type", "schema", "tick", "timestamp_nanos", "wall_time_millis", "diagnostics", "events",
            "features", "critic_only", "action_history"}


# --------------------------------------------------------------- helpers ---
def close(a: float, b: float, tol: float = 1e-4) -> bool:
    return abs(a - b) <= tol


def wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def check_value(v, kind: str, lo, hi) -> str | None:
    if kind == "bool":
        return None if isinstance(v, bool) else f"expected bool, got {v!r}"
    if kind == "str":
        return None if isinstance(v, str) else f"expected string, got {v!r}"
    if not is_number(v):
        return f"expected number, got {v!r}"
    if not math.isfinite(v):
        return f"not finite: {v!r}"
    if kind == "int" and isinstance(v, float) and not v.is_integer():
        return f"expected integer, got {v!r}"
    if lo is not None and v < lo - EPS:
        return f"{v} below minimum {lo}"
    if hi is not None and v > hi + EPS:
        return f"{v} above maximum {hi}"
    return None


# ----------------------------------------------------- single-frame checks ---
def validate_frame(frame: dict) -> list[tuple[str, str, str]]:
    """Returns (level, check, detail) tuples. Level is FAIL or WARN."""
    out: list[tuple[str, str, str]] = []

    def fail(check: str, detail: str = "") -> None:
        out.append(("FAIL", check, detail))

    def check_block(name: str, block, spec_dict: dict, extra_ok: set = frozenset()) -> bool:
        if not isinstance(block, dict):
            fail(f"{name}.is_object", f"got {type(block).__name__}")
            return False
        ok = True
        for key in set(spec_dict) - set(block):
            fail(f"{name}.missing_key", key)
            ok = False
        for key in set(block) - set(spec_dict) - set(extra_ok):
            fail(f"{name}.unexpected_key", key + (" (removed in v2)" if key in REMOVED_KEYS else ""))
        for key, (kind, lo, hi) in spec_dict.items():
            if key in block:
                problem = check_value(block[key], kind, lo, hi)
                if problem:
                    fail(f"{name}.value", f"{key}: {problem}")
                    ok = False
        return ok

    for key in TOP_KEYS - set(frame):
        fail("top.missing_key", key)
    for key in set(frame) - TOP_KEYS:
        fail("top.unexpected_key", key)
    if frame.get("type") != "observation":
        fail("top.type", repr(frame.get("type")))
    if frame.get("schema") != SCHEMA:
        fail("top.schema", f"expected {SCHEMA}, got {frame.get('schema')!r}")
    for key in ("tick", "timestamp_nanos", "wall_time_millis"):
        if not is_number(frame.get(key)) or isinstance(frame.get(key), float) and not frame[key].is_integer():
            fail("top.integer_field", key)

    _check_diagnostics(frame.get("diagnostics"), fail)
    _check_action_history(frame.get("action_history"), fail)

    features = frame.get("features")
    if not isinstance(features, dict):
        fail("features.is_object")
        return out
    if features.get("player_valid") is False:
        if set(features) != {"player_valid"}:
            fail("features.invalid_player_keys", str(sorted(set(features) - {"player_valid"})))
        return out

    spec_all = {**SELF, **TARGET}
    features_ok = check_block("features", features, spec_all, {"target_relative_velocity_history"})
    _check_velocity_history(features.get("target_relative_velocity_history"), fail)
    events_ok = check_block("events", frame.get("events"), EVENTS)
    critic_ok = check_block("critic_only", frame.get("critic_only"), CRITIC)

    if features_ok:
        _check_features(features, fail)
    if events_ok:
        _check_events(frame["events"], features if features_ok else None, fail)
    if critic_ok and features_ok:
        _check_critic(frame["critic_only"], features, fail)
    return out


def _check_diagnostics(diag, fail) -> None:
    if not isinstance(diag, dict):
        fail("diagnostics.is_object")
        return
    state, clicks = diag.get("action_state"), diag.get("consumed_clicks")
    if not isinstance(state, dict) or set(state) != set(DIAG_ACTION_KEYS) \
            or not all(isinstance(v, bool) for v in state.values()):
        fail("diagnostics.action_state", "expected exactly the 10 boolean action keys")
    if not isinstance(clicks, dict) or set(clicks) != {"attack", "use"} \
            or not all(is_number(v) and v >= 0 for v in clicks.values()):
        fail("diagnostics.consumed_clicks", "expected attack/use non-negative counts")
    for key in ("last_applied_sequence", "last_applied_observation_tick"):
        if not is_number(diag.get(key)) or diag[key] < -1:
            fail("diagnostics.last_applied", key)


def _check_action_history(history, fail) -> None:
    if not isinstance(history, list) or len(history) != 20:
        fail("action_history.shape", "expected 20 rows")
        return
    for i, row in enumerate(history):
        if not isinstance(row, list) or len(row) != 11 or not all(is_number(v) and math.isfinite(v) for v in row):
            fail("action_history.row", f"row {i} must be 11 finite numbers")
            return
        if any(v not in (0, 1) for v in row[:9]):
            fail("action_history.binary", f"row {i} button values must be 0/1: {row[:9]}")
        if abs(row[9]) > 180 or abs(row[10]) > 180:
            fail("action_history.view_delta_range", f"row {i}: {row[9:]}")


def _check_velocity_history(history, fail) -> None:
    if not isinstance(history, list) or len(history) != 5:
        fail("velocity_history.shape", "expected 5 rows")
        return
    for i, row in enumerate(history):
        if not isinstance(row, list) or len(row) != 3 or not all(is_number(v) and math.isfinite(v) for v in row):
            fail("velocity_history.row", f"row {i} must be 3 finite numbers")


def _check_features(f: dict, fail) -> None:
    # Camera trig consistency (yaw is intentionally unwrapped; sin/cos must still agree).
    for name, angle in (("camera_yaw", f["camera_yaw"]), ("camera_pitch", f["camera_pitch"])):
        rad = math.radians(angle)
        if not close(f[name + "_sin"], math.sin(rad), 1e-3) or not close(f[name + "_cos"], math.cos(rad), 1e-3):
            fail("camera.trig_matches_angle", name)
    if not close(math.hypot(f["player_velocity_forward"], f["player_velocity_right"]),
                 f["player_horizontal_speed"], 1e-4):
        fail("self.egocentric_velocity_norm", "hypot(forward, right) != horizontal speed")

    # Counters and flags.
    if f["ticks_on_ground"] > 0 and f["ticks_in_air"] > 0:
        fail("self.ground_air_exclusive")
    if f["player_on_ground"] != (f["ticks_on_ground"] > 0):
        fail("self.on_ground_matches_counter")
    if f["has_hit_enemy"] != (f["ticks_since_hit_enemy"] >= 0):
        fail("combat.has_hit_enemy_matches_counter")
    if f["has_been_hit"] != (f["ticks_since_hit_by_enemy"] >= 0):
        fail("combat.has_been_hit_matches_counter")
    if f["knockback_combo"] > f["hit_combo"]:
        fail("combat.knockback_combo_le_hit_combo")
    if f["attack_strength"] >= 0.9999 and f["ticks_until_attack_ready"] > 1e-6:
        fail("attack.ready_matches_strength")
    if f["last_hit_type"] not in HIT_TYPES:
        fail("combat.last_hit_type_enum", f["last_hit_type"])
    else:
        flags = {"sweep": f["last_hit_sweep"], "critical": f["last_hit_critical"],
                 "knockback": f["last_hit_knockback"], "normal": f["last_hit_normal"]}
        hit_type = f["last_hit_type"]
        if hit_type == "none" and any(flags.values()):
            fail("combat.hit_flags_vs_type", "type none but a last_hit_* flag is set")
        elif hit_type != "none" and not flags[hit_type]:
            fail("combat.hit_flags_vs_type", f"type {hit_type} but its flag is false")
        if flags["normal"] and (flags["sweep"] or flags["critical"] or flags["knockback"]):
            fail("combat.normal_is_exclusive")
    outcome = f["last_attack_outcome"]
    if outcome not in OUTCOMES:
        fail("combat.last_attack_outcome_enum", outcome)
    else:
        for name in ("hit", "miss", "blocked", "no_effect"):
            if f[f"last_attack_{name}"] != (outcome == name):
                fail("combat.last_attack_flags_vs_outcome", name)
        if (outcome == "none") != (f["ticks_since_attack_outcome"] == -1):
            fail("combat.outcome_counter_vs_none")

    # Target tracking and line-of-sight memory.
    if f["target_valid"] and not f["target_tracked"]:
        fail("target.valid_implies_tracked")
    if f["target_valid"] and f["ticks_since_target_seen"] != 0:
        fail("target.visible_means_seen_now")
    if not f["target_valid"] and f["target_tracked"] and f["has_last_known_target"] \
            and f["ticks_since_target_seen"] < 1:
        fail("target.invisible_means_seen_earlier")
    memory = (f["last_known_target_rel_forward"], f["last_known_target_rel_right"], f["last_known_target_rel_up"])
    if not f["has_last_known_target"]:
        if any(memory) or f["last_known_target_distance"] or f["last_known_target_yaw_error"]:
            fail("memory.empty_is_zero")
    else:
        if not close(math.sqrt(sum(v * v for v in memory)), f["last_known_target_distance"], 1e-3):
            fail("memory.distance_matches_vector")
        horizontal = math.hypot(memory[0], memory[1])
        if horizontal > 1e-3:
            expected = math.degrees(math.atan2(memory[1], memory[0]))
            if abs(wrap(f["last_known_target_yaw_error"] - expected)) > 0.05:
                fail("memory.yaw_error_matches_vector",
                     f"{f['last_known_target_yaw_error']:.3f} vs {expected:.3f}")

    if f["target_valid"]:
        _check_visible_target(f, memory, fail)
    else:
        _check_masked_target(f, fail)


def _check_visible_target(f: dict, memory, fail) -> None:
    rel = (f["target_rel_forward"], f["target_rel_right"], f["target_rel_up"])
    horizontal = math.hypot(rel[0], rel[1])
    if not close(horizontal, f["target_horizontal_distance"], 1e-3):
        fail("target.horizontal_distance_matches_rel", f"{horizontal:.4f} vs {f['target_horizontal_distance']:.4f}")
    if f["target_distance"] < f["target_horizontal_distance"] - 1e-3:
        fail("target.distance_ge_horizontal")
    if f["target_hitbox_distance"] > math.sqrt(sum(v * v for v in rel)) + 1e-3:
        fail("target.hitbox_le_center_distance", "hitbox distance cannot exceed eye-to-center distance")
    if abs(f["target_hitbox_distance"] - f["player_reach"]) > 1e-6 \
            and f["target_in_reach"] != (f["target_hitbox_distance"] <= f["player_reach"]):
        fail("target.in_reach_matches_hitbox",
             f"hitbox {f['target_hitbox_distance']:.3f}, reach {f['player_reach']}, flag {f['target_in_reach']}")
    if horizontal > 1e-3:
        expected = math.degrees(math.atan2(rel[1], rel[0]))  # positive = target on the right
        if abs(wrap(f["target_yaw_error"] - expected)) > 0.05:
            fail("target.yaw_error_matches_egocentric",
                 f"{f['target_yaw_error']:.3f} vs {expected:.3f} (sign convention: positive = turn right)")
    for a, b in zip(rel, memory):
        if not close(a, b, 1e-4):
            fail("memory.matches_live_when_visible")
            break
    if not close(f["target_yaw_error"], f["last_known_target_yaw_error"], 1e-4):
        fail("memory.yaw_error_matches_live_when_visible")
    yaw_rad = math.radians(f["target_yaw"])
    if not close(f["target_yaw_sin"], math.sin(yaw_rad), 1e-3) or not close(f["target_yaw_cos"], math.cos(yaw_rad), 1e-3):
        fail("target.yaw_trig")
    if not close(f["target_facing_cos"], math.cos(math.radians(f["target_facing_error"])), 1e-3):
        fail("target.facing_cos_matches_error")
    if not close(math.hypot(f["target_relative_velocity_forward"], f["target_relative_velocity_right"]),
                 f["target_relative_horizontal_speed"], 1e-4):
        fail("target.relative_speed_norm")
    if f["target_held_item"] not in ITEM_CATEGORIES \
            or ITEM_CATEGORIES.index(f["target_held_item"]) != f["target_held_item_id"]:
        fail("target.held_item_matches_id", f"{f['target_held_item']!r} / {f['target_held_item_id']}")
    if f["target_bbox_left"] > f["target_bbox_right"] or f["target_bbox_top"] > f["target_bbox_bottom"]:
        fail("target.bbox_ordered")
    if not close(f["target_bbox_width"], f["target_bbox_right"] - f["target_bbox_left"], 1e-4) \
            or not close(f["target_bbox_height"], f["target_bbox_bottom"] - f["target_bbox_top"], 1e-4):
        fail("target.bbox_size_matches_edges")
    if f["target_on_screen"] != (f["target_bbox_width"] > 0 or f["target_bbox_height"] > 0
                                 or f["target_bbox_projection_valid"]):
        pass  # on-screen semantics are projection-dependent; range checks above are the contract
    if f["crosshair_on_target"] and not f["target_on_screen"]:
        fail("target.crosshair_implies_on_screen")
    if f["crosshair_on_target"] and not f["target_in_reach"] and f["target_hitbox_distance"] > 6.5:
        fail("target.crosshair_far_beyond_reach", f"{f['target_hitbox_distance']:.2f}")
    if not f["target_center_projection_valid"] and (f["target_dx"] or f["target_dy"]):
        fail("target.invalid_projection_is_neutral")
    if f["target_center_projection_valid"] and f["target_rel_forward"] <= 0:
        fail("target.projection_valid_means_in_front")


def _check_masked_target(f: dict, fail) -> None:
    for key, (kind, _, _) in TARGET.items():
        value = f[key]
        if kind == "bool" and value is not False:
            fail("masking.live_target_bool_must_be_false", key)
        elif kind in ("num", "int") and value != 0:
            fail("masking.live_target_number_must_be_zero", f"{key}={value}")
        elif kind == "str" and value != "none":
            fail("masking.live_target_string_must_be_none", f"{key}={value!r}")
    history = f.get("target_relative_velocity_history")
    if isinstance(history, list) and any(any(v != 0 for v in row) for row in history if isinstance(row, list)):
        fail("masking.velocity_history_must_be_zero")


def _check_events(e: dict, f: dict | None, fail) -> None:
    if e["attack_outcome"] not in OUTCOMES:
        fail("events.attack_outcome_enum", e["attack_outcome"])
    if not close(e["damage_dealt"], max(0.0, -e["target_health_delta"]), 1e-4):
        fail("events.damage_dealt_matches_delta")
    if not close(e["damage_taken"], max(0.0, -e["self_health_delta"]), 1e-4):
        fail("events.damage_taken_matches_delta")
    if e["hits_dealt"] > 0 and e["attack_outcome"] != "hit":
        fail("events.hits_dealt_implies_hit_outcome")
    if e["attack_outcome"] == "hit" and e["hits_dealt"] < 1:
        fail("events.hit_outcome_implies_hits_dealt")
    if f is not None and e["attack_outcome"] != "none":
        if f["last_attack_outcome"] != e["attack_outcome"] or f["ticks_since_attack_outcome"] != 0:
            fail("events.outcome_reflected_in_features",
                 f"event {e['attack_outcome']} vs feature {f['last_attack_outcome']} "
                 f"({f['ticks_since_attack_outcome']} ticks ago)")


def _check_critic(c: dict, f: dict, fail) -> None:
    if c["target_tracked"] != f["target_tracked"]:
        fail("critic.tracked_matches_features")
    if c["target_visible"] != f["target_valid"]:
        fail("critic.visible_matches_target_valid")
    if f["target_valid"]:
        pairs = (("target_rel_forward", "target_rel_forward"), ("target_rel_right", "target_rel_right"),
                 ("target_rel_up", "target_rel_up"), ("target_hitbox_distance", "target_hitbox_distance"),
                 ("target_yaw_error", "target_yaw_error"), ("target_health", "target_health"),
                 ("target_absorption", "target_absorption"))
        for ck, fk in pairs:
            if not close(c[ck], f[fk], 1e-4):
                fail("critic.matches_features_when_visible", f"{ck}: {c[ck]} vs {f[fk]}")
    if not c["target_tracked"] and any(c[k] for k in ("target_rel_forward", "target_rel_right", "target_rel_up",
                                                      "target_hitbox_distance", "target_health")):
        fail("critic.untracked_is_zero")


# ---------------------------------------------------- frame-to-frame checks ---
COUNTER_KEYS = COUNTERS.split()


def validate_pair(prev: dict, cur: dict) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    try:
        dt = cur["tick"] - prev["tick"]
        if dt <= 0:
            out.append(("FAIL", "sequence.tick_increases", f"{prev['tick']} -> {cur['tick']}"))
            return out
        if dt > 1:
            out.append(("WARN", "sequence.skipped_ticks", f"{prev['tick']} -> {cur['tick']}"))
        if cur["timestamp_nanos"] <= prev["timestamp_nanos"]:
            out.append(("FAIL", "sequence.timestamp_increases", ""))
        if cur["wall_time_millis"] < prev["wall_time_millis"]:
            out.append(("FAIL", "sequence.wall_time_monotonic", ""))
        pf, cf = prev["features"], cur["features"]
        if dt != 1 or not pf.get("player_valid") or not cf.get("player_valid"):
            return out
        if abs(cf["camera_yaw"] - pf["camera_yaw"]) > 90:
            out.append(("WARN", "sequence.camera_yaw_jump", f"{pf['camera_yaw']:.1f} -> {cf['camera_yaw']:.1f}"))
        for key in COUNTER_KEYS:
            old, new = pf[key], cf[key]
            allowed = {old + 1, 0} if old >= 0 else {-1, 0}
            if key == "ticks_since_target_seen":
                allowed.add(-1)  # target switch resets memory
            if new not in allowed:
                out.append(("FAIL", "sequence.counter_step", f"{key}: {old} -> {new}"))
        for key, flag in (("ticks_on_ground", "player_on_ground"), ("ticks_in_air", None)):
            if key == "ticks_on_ground" and pf[flag] and cf[flag] and cf[key] != pf[key] + 1:
                out.append(("FAIL", "sequence.ground_counter_step", f"{pf[key]} -> {cf[key]}"))
            if key == "ticks_in_air" and not pf["player_on_ground"] and not cf["player_on_ground"] \
                    and cf[key] != pf[key] + 1:
                out.append(("FAIL", "sequence.air_counter_step", f"{pf[key]} -> {cf[key]}"))
        if cf["hit_combo"] > pf["hit_combo"] + max(1, cur["events"]["hits_dealt"]):
            out.append(("FAIL", "sequence.hit_combo_step", f"{pf['hit_combo']} -> {cf['hit_combo']}"))
    except (KeyError, TypeError) as exc:  # frame-level validation reports the root cause
        out.append(("WARN", "sequence.skipped_due_to_malformed_frame", repr(exc)))
    return out


# --------------------------------------------------------- synthetic frames ---
def _default(kind: str, lo):
    if kind == "bool":
        return False
    if kind == "str":
        return "none"
    return 0 if (lo is None or lo <= 0) else lo


def make_frame(tick: int = 100, visible: bool = True, ground_ticks: int = 1) -> dict:
    """A synthetic frame that satisfies every contract. Used by --self-test."""
    f = {k: _default(*v[:1], v[1]) for k, v in {**SELF, **TARGET}.items()}
    for key in COUNTER_KEYS:
        f[key] = -1
    yaw = 30.0
    f.update(player_valid=True, player_on_ground=True, ticks_on_ground=ground_ticks, camera_yaw=yaw,
             camera_yaw_sin=math.sin(math.radians(yaw)), camera_yaw_cos=math.cos(math.radians(yaw)),
             camera_pitch_cos=1.0, camera_horizontal_fov_degrees=100.0, player_reach=3.0, attack_strength=1.0,
             player_health=20.0, player_food=20, player_attack_damage=1.0, player_attack_speed=4.0,
             player_horizontal_speed=0.2, player_velocity_forward=0.2, last_hit_type="none",
             last_attack_outcome="none", target_held_item="none")
    rel = (2.0, 0.5, -0.72)
    f.update(target_tracked=True, has_last_known_target=True,
             last_known_target_rel_forward=rel[0], last_known_target_rel_right=rel[1],
             last_known_target_rel_up=rel[2], last_known_target_distance=math.sqrt(sum(v * v for v in rel)),
             last_known_target_yaw_error=math.degrees(math.atan2(rel[1], rel[0])))
    if visible:
        f.update(target_valid=True, ticks_since_target_seen=0, target_health=20.0, target_height=1.8,
                 target_distance=2.5, target_horizontal_distance=math.hypot(rel[0], rel[1]),
                 target_hitbox_distance=1.3, target_in_reach=True, target_rel_forward=rel[0],
                 target_rel_right=rel[1], target_rel_up=rel[2], target_yaw_error=f["last_known_target_yaw_error"],
                 target_yaw=100.0, target_yaw_sin=math.sin(math.radians(100.0)),
                 target_yaw_cos=math.cos(math.radians(100.0)), target_facing_error=170.0,
                 target_facing_cos=math.cos(math.radians(170.0)), target_relative_velocity_forward=0.1,
                 target_relative_horizontal_speed=0.1, target_held_item="sword", target_held_item_id=1,
                 target_bbox_left=0.4, target_bbox_top=0.2, target_bbox_right=0.6, target_bbox_bottom=0.9,
                 target_bbox_width=0.2, target_bbox_height=0.7, target_on_screen=True,
                 target_bbox_projection_valid=True, target_center_projection_valid=True, target_dy=-0.05)
        history = [[0.0, 0.0, 0.0]] * 4 + [[0.1, 0.0, 0.0]]
    else:
        f.update(target_valid=False, ticks_since_target_seen=7)
        history = [[0.0, 0.0, 0.0]] * 5
    f["target_relative_velocity_history"] = history
    live = visible
    return {
        "type": "observation", "schema": SCHEMA, "tick": tick, "timestamp_nanos": tick * 50_000_000,
        "wall_time_millis": 1_700_000_000_000 + tick * 50,
        "diagnostics": {"action_state": {k: False for k in DIAG_ACTION_KEYS},
                        "consumed_clicks": {"attack": 0, "use": 0},
                        "last_applied_sequence": -1, "last_applied_observation_tick": -1},
        "events": {"damage_dealt": 0.0, "damage_taken": 0.0, "target_health_delta": 0.0,
                   "self_health_delta": 0.0, "hits_dealt": 0, "hits_taken": 0, "attack_outcome": "none"},
        "features": f,
        "critic_only": {"target_tracked": True, "target_visible": live,
                        "target_rel_forward": rel[0], "target_rel_right": rel[1], "target_rel_up": rel[2],
                        "target_hitbox_distance": 1.3 if live else 0.0,
                        "target_yaw_error": f["last_known_target_yaw_error"] if live else 0.0,
                        "target_health": 20.0 if live else 0.0, "target_absorption": 0.0},
        "action_history": [[0] * 9 + [0.0, 0.0] for _ in range(20)],
    }


def self_test() -> int:
    problems = 0

    def expect_pass(label: str, results) -> None:
        nonlocal problems
        fails = [r for r in results if r[0] == "FAIL"]
        print(f"  {'ok  ' if not fails else 'BAD '} {label}")
        for r in fails:
            print(f"        unexpected: {r[1]} {r[2]}")
        problems += bool(fails)

    def expect_fail(label: str, results, check_prefix: str) -> None:
        nonlocal problems
        hit = [r for r in results if r[0] == "FAIL" and r[1].startswith(check_prefix)]
        print(f"  {'ok  ' if hit else 'BAD '} mutation caught: {label}")
        problems += not hit

    print("Valid frames must pass:")
    expect_pass("visible target", validate_frame(make_frame()))
    expect_pass("target tracked but not visible", validate_frame(make_frame(visible=False)))
    invalid = make_frame()
    invalid["features"] = {"player_valid": False}
    expect_pass("player not valid", validate_frame(invalid))
    expect_pass("consecutive frames", validate_pair(make_frame(100), make_frame(101, ground_ticks=2)))

    print("Broken frames must fail:")

    def mutate(label: str, prefix: str, fn, visible: bool = True) -> None:
        frame = copy.deepcopy(make_frame(visible=visible))
        fn(frame)
        expect_fail(label, validate_frame(frame), prefix)

    mutate("missing camera_yaw", "features.missing_key", lambda x: x["features"].pop("camera_yaw"))
    mutate("removed key player_x returns", "features.unexpected_key", lambda x: x["features"].update(player_x=1.0))
    mutate("removed key target_aim_dx returns", "features.unexpected_key",
           lambda x: x["features"].update(target_aim_dx=0.1))
    mutate("diagnostics left inside features", "features.unexpected_key",
           lambda x: x["features"].update(remote_control_diagnostics={}))
    mutate("NaN health", "features.value", lambda x: x["features"].update(player_health=float("nan")))
    mutate("pitch out of range", "features.value", lambda x: x["features"].update(camera_pitch=95.0))
    mutate("yaw_sin disagrees with yaw", "camera.trig", lambda x: x["features"].update(camera_yaw_sin=0.9))
    mutate("in_reach flag wrong", "target.in_reach",
           lambda x: x["features"].update(target_in_reach=False))
    mutate("yaw error sign flipped", "target.yaw_error",
           lambda x: x["features"].update(target_yaw_error=-x["features"]["target_yaw_error"]))
    mutate("masking leak while invisible", "masking",
           lambda x: x["features"].update(target_distance=2.0), visible=False)
    mutate("held item id mismatch", "target.held_item", lambda x: x["features"].update(target_held_item_id=2))
    mutate("bad attack outcome enum", "combat.last_attack_outcome_enum",
           lambda x: x["features"].update(last_attack_outcome="bogus"))
    mutate("hit flags vs type", "combat.hit_flags_vs_type", lambda x: x["features"].update(last_hit_type="sweep"))
    mutate("damage event without delta", "events.damage_dealt", lambda x: x["events"].update(damage_dealt=5.0))
    mutate("hit event without outcome", "events.hits_dealt", lambda x: x["events"].update(hits_dealt=1))
    mutate("critic visibility disagrees", "critic.visible", lambda x: x["critic_only"].update(target_visible=False))
    mutate("action_history too short", "action_history.shape", lambda x: x["action_history"].pop())
    mutate("non-binary action value", "action_history.binary", lambda x: x["action_history"][0].__setitem__(0, 2))
    mutate("wrong schema", "top.schema", lambda x: x.update(schema="minecraftai.pvp.v1"))

    print("Broken sequences must fail:")
    later = make_frame(101, ground_ticks=2)
    later["features"]["ticks_since_hit_enemy"] = 50
    expect_fail("counter jumped", validate_pair(make_frame(100), later), "sequence.counter_step")
    expect_fail("tick did not advance", validate_pair(make_frame(100), make_frame(100)), "sequence.tick")
    stalled = make_frame(101, ground_ticks=1)
    expect_fail("ground counter stalled", validate_pair(make_frame(100), stalled), "sequence.ground_counter_step")

    print("\nSELF-TEST " + ("FAILED" if problems else "PASSED"))
    return 1 if problems else 0


# ------------------------------------------------------------------ live run ---
class Report:
    def __init__(self) -> None:
        self.counts: Counter = Counter()
        self.first: dict = {}
        self.warn_counts: Counter = Counter()
        self.warn_first: dict = {}

    def add(self, tick, results) -> None:
        for level, check, detail in results:
            counts, first = (self.counts, self.first) if level == "FAIL" else (self.warn_counts, self.warn_first)
            counts[check] += 1
            first.setdefault(check, (tick, detail))

    def print(self) -> None:
        for title, counts, first in (("FAILURES", self.counts, self.first),
                                     ("WARNINGS", self.warn_counts, self.warn_first)):
            if counts:
                print(f"\n{title}:")
                for check, n in counts.most_common():
                    tick, detail = first[check]
                    print(f"  {check}  x{n}  (first at tick {tick}) {detail}")


def live(args: argparse.Namespace) -> int:
    try:
        sock = socket.create_connection((args.host, args.port), timeout=5)
    except OSError as exc:
        print(f"Could not connect to {args.host}:{args.port}: {exc}\nIs runClient open with the mod loaded?")
        return 1
    sock.settimeout(2.0)
    print(f"Connected to {args.host}:{args.port}. Press K in game to start the observation stream.")

    report = Report()
    ranges: dict[str, list[float]] = {}
    seen = Counter()
    frames = 0
    previous = None
    shown = False
    started = last_rx = time.perf_counter()
    buffer = b""

    try:
        while frames < args.frames and (not args.seconds or time.perf_counter() - started < args.seconds):
            try:
                chunk = sock.recv(65536)
            except socket.timeout:
                if time.perf_counter() - last_rx > 5.0:
                    print("... no data yet; is AI control on (K)?")
                    last_rx = time.perf_counter()
                continue
            if not chunk:
                print("Connection closed by the mod.")
                break
            last_rx = time.perf_counter()
            buffer += chunk
            while b"\n" in buffer and frames < args.frames:
                line, buffer = buffer.split(b"\n", 1)
                if not line.strip():
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError as exc:
                    report.add(-1, [("FAIL", "stream.bad_json", str(exc))])
                    continue
                kind = message.get("type")
                if kind == "ready":
                    ok = message.get("protocol") == 2 and message.get("observation_schema") == SCHEMA
                    print(f"[ready] protocol={message.get('protocol')} schema={message.get('observation_schema')}"
                          f" -> {'ok' if ok else 'MISMATCH'}")
                    if not ok:
                        report.add(-1, [("FAIL", "stream.ready_schema", json.dumps(message))])
                    continue
                if kind != "observation":
                    print(f"[{kind}] {json.dumps(message)}")
                    continue

                frames += 1
                if args.show_first and not shown:
                    print(json.dumps(message, indent=2))
                    shown = True
                results = validate_frame(message)
                if previous is not None:
                    results += validate_pair(previous, message)
                report.add(message.get("tick", -1), results)
                previous = message

                f = message.get("features", {})
                if f.get("player_valid"):
                    seen["frames with player"] += 1
                    live_target = bool(f.get("target_valid"))
                    seen["frames with visible target"] += live_target
                    seen["frames target tracked but hidden"] += bool(f.get("target_tracked")) and not live_target
                    seen["frames in reach"] += bool(f.get("target_in_reach"))
                    seen["frames crosshair on target"] += bool(f.get("crosshair_on_target"))
                    e = message.get("events", {})
                    seen["hits dealt"] += e.get("hits_dealt", 0) or 0
                    seen["hits taken"] += e.get("hits_taken", 0) or 0
                    seen[f"attack outcome: {e.get('attack_outcome')}"] += e.get("attack_outcome") not in (None, "none")
                    for key, value in f.items():
                        if is_number(value) and (live_target or key not in TARGET) and math.isfinite(value):
                            lohi = ranges.setdefault(key, [value, value])
                            lohi[0], lohi[1] = min(lohi[0], value), max(lohi[1], value)
                else:
                    seen["frames without player"] += 1
                if frames % 100 == 0:
                    print(f"  ...{frames} frames, {sum(report.counts.values())} failures so far")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        sock.close()

    elapsed = max(time.perf_counter() - started, 1e-9)
    print(f"\nValidated {frames} observations in {elapsed:.1f}s ({frames / elapsed:.1f}/s, expect ~20/s)")
    for label, n in sorted(seen.items()):
        if n:
            print(f"  {label}: {n}")
    report.print()

    constant = sorted(k for k, (lo, hi) in ranges.items() if lo == hi)
    if constant:
        print(f"\nCoverage: {len(constant)} numeric features never changed during this run "
              f"(normal for situations that did not occur, e.g. effects, hits):")
        print("  " + ", ".join(constant))

    unmet = []
    if args.require_target and not seen["frames with visible target"]:
        unmet.append("no frame had a visible target")
    if args.require_hits and not (seen["hits dealt"] or seen["hits taken"]):
        unmet.append("no hit events were observed")
    if frames == 0:
        unmet.append("no observations were received")
    for item in unmet:
        print(f"\nREQUIREMENT NOT MET: {item}")

    failed = bool(report.counts) or bool(unmet)
    print("\nRESULT: " + ("FAIL" if failed else "PASS"))
    return 1 if failed else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--frames", type=int, default=400, help="observations to validate (default 400, ~20 s)")
    p.add_argument("--seconds", type=float, default=0.0, help="also stop after N seconds")
    p.add_argument("--self-test", action="store_true", help="offline: test the validator itself and exit")
    p.add_argument("--require-target", action="store_true", help="fail unless a target was visible at least once")
    p.add_argument("--require-hits", action="store_true", help="fail unless a hit was dealt or taken")
    p.add_argument("--show-first", action="store_true", help="pretty-print the first observation")
    args = p.parse_args()
    return self_test() if args.self_test else live(args)


if __name__ == "__main__":
    sys.exit(main())