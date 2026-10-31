"""Dependency-free movement observation and event-reward helpers for combo PPO."""

from __future__ import annotations

import math
from typing import Any

MOVEMENT_OBSERVATION_SIZE = 27
MOVEMENT_ACTION_SIZE = 2  # forward, sprint; attacks are scripted
FORWARD_TOGGLE_PENALTY = 0.0  # Temporarily disabled.
SPRINT_TOGGLE_PENALTY = 0.0  # Temporarily disabled.
STRAFE_TOGGLE_PENALTY = 0.0  # A/D strafe is disabled in train_combo_attack_ppo.
KNOCKBACK_HIT_REWARD = 1.0
COMBO_REWARD_PER_EXTRA_HIT = 0.2
MAX_COMBO_REWARD = 1.0
INCOMING_HIT_PENALTY = 0.5
ATTACK_SUCCESS_REWARD = 0.5
ATTACK_MISS_PENALTY = 0.25
FALSE_ATTACK_PENALTY = 0.10
PREMATURE_ATTACK_PENALTY = 0.25
MISSED_ELIGIBLE_PENALTY = 0.02
SHORT_HOLD_TICKS = 3
DAMAGE_DEALT_REWARD_PER_POINT = 0.25
DAMAGE_TAKEN_PENALTY_PER_POINT = 0.35
MAX_DAMAGE_REWARD_POINTS_PER_TICK = 8.0


def _feature_number(features: dict[str, Any], name: str, default: float = 0.0) -> float:
    try:
        value = float(features.get(name, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _elapsed_feature(features: dict[str, Any], name: str) -> float:
    age = _feature_number(features, name, -1.0)
    return -1.0 if age < 0.0 else _clamp(age / 40.0, 0.0, 1.0)


def damage_transition_reward(frame: dict[str, Any]) -> float:
    """Reward target health lost and penalize player health lost this tick."""
    events = frame.get("events", {})
    if not isinstance(events, dict):
        return 0.0
    damage_dealt = _clamp(_feature_number(events, "damage_dealt"), 0.0, MAX_DAMAGE_REWARD_POINTS_PER_TICK)
    damage_taken = _clamp(_feature_number(events, "damage_taken"), 0.0, MAX_DAMAGE_REWARD_POINTS_PER_TICK)
    return (
        damage_dealt * DAMAGE_DEALT_REWARD_PER_POINT
        - damage_taken * DAMAGE_TAKEN_PENALTY_PER_POINT
    )


def encode_movement_observation(
    frame: dict[str, Any],
    previous_forward: bool,
    forward_streak: int,
    previous_sprint: bool,
    sprint_streak: int,
    qualified_knockback_combo: int = 0,
) -> tuple[float, ...]:
    features = frame.get("features", {})
    if not isinstance(features, dict):
        features = {}
    history = frame.get("action_history", [])
    previous_attack = (
        bool(history[-1][7])
        if isinstance(history, list) and history and isinstance(history[-1], (list, tuple)) and len(history[-1]) > 7
        else False
    )
    return (
        float(bool(features.get("target_valid", False))),
        float(bool(features.get("target_on_screen", False))),
        float(bool(features.get("crosshair_on_target", False))),
        _clamp(_feature_number(features, "target_dx") / 0.5, -1.0, 1.0),
        _clamp(_feature_number(features, "target_dy") / 0.5, -1.0, 1.0),
        _clamp(_feature_number(features, "target_distance") / 16.0, 0.0, 1.0),
        _clamp(_feature_number(features, "target_horizontal_distance") / 16.0, 0.0, 1.0),
        _clamp(_feature_number(features, "target_relative_horizontal_speed") / 2.0, 0.0, 1.0),
        float(bool(features.get("target_sprinting", False))),
        _clamp(_feature_number(features, "player_horizontal_speed") / 0.3, 0.0, 1.0),
        float(bool(features.get("player_on_ground", False))),
        float(bool(features.get("player_sprinting", False))),
        _clamp(_feature_number(features, "player_health") / 20.0, 0.0, 1.0),
        _clamp(_feature_number(features, "player_food") / 20.0, 0.0, 1.0),
        _clamp(_feature_number(features, "attack_strength"), 0.0, 1.0),
        float(bool(features.get("mainhand_is_sword", False))),
        float(bool(features.get("has_hit_enemy", False))),
        _elapsed_feature(features, "ticks_since_hit_enemy"),
        float(bool(features.get("has_been_hit", False))),
        _elapsed_feature(features, "ticks_since_hit_by_enemy"),
        float(previous_forward),
        _clamp(forward_streak / 40.0, 0.0, 1.0),
        float(previous_sprint),
        _clamp(sprint_streak / 40.0, 0.0, 1.0),
        _clamp(_feature_number(features, "hit_combo") / 10.0, 0.0, 1.0),
        float(previous_attack),
        _clamp(qualified_knockback_combo / 10.0, 0.0, 1.0),
    )


def _new_combat_event(
    before: dict[str, Any], after: dict[str, Any], flag_name: str, age_name: str
) -> bool:
    before_features = before.get("features", {})
    after_features = after.get("features", {})
    if not isinstance(before_features, dict) or not isinstance(after_features, dict):
        return False
    if not bool(after_features.get(flag_name, False)):
        return False
    after_age = _feature_number(after_features, age_name, -1.0)
    if after_age < 0.0:
        return False
    if not bool(before_features.get(flag_name, False)):
        return True
    before_age = _feature_number(before_features, age_name, -1.0)
    return before_age >= 0.0 and after_age < before_age


def _attack_history_offset(frame: dict[str, Any], event_age: int) -> int | None:
    history = frame.get("action_history", [])
    if not isinstance(history, list):
        return None
    # The damage event can be reported on the attack tick or the following tick.
    candidate_ages = (0, 1) if event_age <= 0 else (event_age, event_age - 1)
    for age in dict.fromkeys(candidate_ages):
        index = len(history) - 1 - age
        if index < 0 or index >= len(history):
            continue
        row = history[index]
        if not isinstance(row, (list, tuple)) or len(row) < 8:
            continue
        if bool(row[7]):
            return age
    return None


def movement_transition_reward(
    before: dict[str, Any],
    after: dict[str, Any],
    forward: bool,
    sprint: bool,
    attack: bool,
    ground_truth_eligible: bool,
    previous_forward: bool,
    previous_sprint: bool,
    prior_attack_context: tuple[tuple[bool, bool], ...] = (),
    prior_knockback_combo: int = 0,
    previous_forward_streak: int = 0,
    previous_sprint_streak: int = 0,
    prior_incoming_combo: int = 0,
    left: bool = False,
    right: bool = False,
    previous_left: bool = False,
    previous_right: bool = False,
    previous_left_streak: int = 0,
    previous_right_streak: int = 0,
) -> tuple[float, bool, bool, int | None, float, bool, bool, bool, bool, bool, int, int]:
    """Reward imitation of eligible attacks and confirmed knockback combat events."""
    reward = 0.0
    before_features = before.get("features", {})
    after_features = after.get("features", {})
    if not isinstance(before_features, dict):
        before_features = {}
    if not isinstance(after_features, dict):
        after_features = {}

    outgoing_hit = _new_combat_event(
        before, after, "has_hit_enemy", "ticks_since_hit_enemy"
    )
    incoming_hit = _new_combat_event(
        before, after, "has_been_hit", "ticks_since_hit_by_enemy"
    )
    event_age = int(_feature_number(after_features, "ticks_since_hit_enemy", -1.0))
    attack_offset = (
        _attack_history_offset(after, event_age)
        if outgoing_hit
        else None
    )
    if attack_offset == 0:
        hit_action_context = (attack, ground_truth_eligible)
    elif attack_offset is not None and 0 < attack_offset <= len(prior_attack_context):
        hit_action_context = prior_attack_context[-attack_offset]
    else:
        hit_action_context = None
    ground_truth_hit = (
        attack_offset is not None
        and hit_action_context is not None
        and hit_action_context[0]
        and hit_action_context[1]
    )
    knockback_hit = ground_truth_hit and bool(after_features.get("last_hit_knockback", False))
    next_knockback_combo = max(0, prior_knockback_combo)
    if incoming_hit:
        next_knockback_combo = 0
    next_incoming_combo = min(10, max(0, prior_incoming_combo) + int(incoming_hit))
    if knockback_hit:
        next_incoming_combo = 0
    if outgoing_hit:
        next_knockback_combo = (
            next_knockback_combo + 1
            if not incoming_hit and knockback_hit
            else 0
        )
    hit_reward = 0.0
    if knockback_hit:
        hit_reward += KNOCKBACK_HIT_REWARD
        hit_reward += min(
            MAX_COMBO_REWARD,
            COMBO_REWARD_PER_EXTRA_HIT * max(0, next_knockback_combo - 1),
        )
    reward += hit_reward

    # Hit packets may arrive in the following observation; ground_truth_hit
    # already verifies the originating attack and its eligibility in history.
    attack_success = ground_truth_hit
    attack_miss = attack and ground_truth_eligible and not attack_success
    premature_attack = (
        attack and not ground_truth_eligible
        and bool(before_features.get("target_valid", False))
        and bool(before_features.get("mainhand_is_sword", False))
        and bool(before_features.get("crosshair_on_target", False))
        and _feature_number(before_features, "attack_strength") < 1.0
    )
    false_attack = attack and not ground_truth_eligible and not premature_attack
    missed_eligible = ground_truth_eligible and not attack
    if attack_success:
        reward += ATTACK_SUCCESS_REWARD
    # Eligible attack misses are penalized by the trainer after its hit-event
    # grace window, so a one-tick-delayed damage observation can still qualify.
    if false_attack:
        reward -= FALSE_ATTACK_PENALTY
    if premature_attack:
        reward -= PREMATURE_ATTACK_PENALTY
    if missed_eligible:
        reward -= MISSED_ELIGIBLE_PENALTY
    if incoming_hit:
        reward -= INCOMING_HIT_PENALTY * next_incoming_combo
    return (
        reward, knockback_hit, incoming_hit, attack_offset, hit_reward,
        attack_success, attack_miss, false_attack, missed_eligible, premature_attack,
        next_knockback_combo, next_incoming_combo,
    )
