#!/usr/bin/env python3
"""Opt-in live test for normal and sprint-knockback attack rewards.

Run in a safe world with a visible, attackable target and AI control enabled:
    py -3.12 scripts/test_combo_hit_timing.py --live --allow-attacks
Each attempt sends one attack only after the ground-rule gate is true.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_client_suite import LiveClient, validate_observation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from combo_ppo_math import movement_transition_reward
from yaw_ppo_math import scripted_sword_attack_eligible


def check_attack_frame(frame: dict[str, Any], label: str) -> None:
    # knockback_combo was added after some deployed client builds; this test
    # validates the raw hit event and does not depend on that aggregate field.
    errors = [error for error in validate_observation(frame) if "knockback_combo" not in error]
    if errors:
        raise AssertionError(f"{label}: " + "; ".join(errors))
    features = frame.get("features", {})
    required = ("target_valid", "crosshair_on_target", "mainhand_is_sword", "attack_strength",
                "has_hit_enemy", "ticks_since_hit_enemy", "last_hit_knockback")
    missing = [name for name in required if name not in features]
    if missing:
        raise AssertionError(f"{label}: missing attack observation fields {missing}")


def detected_hit(before: dict[str, Any], after: dict[str, Any], *, sprint: bool) -> tuple[bool, bool, float]:
    attack = True
    eligible = scripted_sword_attack_eligible(before)
    if not eligible:
        raise AssertionError("test attempted an attack when ground-rule eligibility was false")
    result = movement_transition_reward(
        before, after,
        forward=False, sprint=sprint, attack=attack, ground_truth_eligible=eligible,
        previous_forward=False, previous_sprint=sprint,
        prior_attack_context=((attack, eligible),),
    )
    return result[1], result[5], result[0]


def wait_for_eligible(client: LiveClient, frame: dict[str, Any], timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while not scripted_sword_attack_eligible(frame):
        if time.monotonic() >= deadline:
            raise TimeoutError("no frame met the attack gate: target crosshair, sword, full cooldown")
        frame = client.next_observation(int(frame["tick"]))
    return frame


def run_attempt(client: LiveClient, frame: dict[str, Any], *, sprint: bool, timeout: float) -> tuple[dict[str, Any], bool, bool, float]:
    deadline = time.monotonic() + timeout
    if sprint:
        # Sprint knockback is based on the player's actual sprint state, which
        # requires forward movement in addition to holding the sprint key.
        while not frame["features"].get("player_sprinting", False):
            if time.monotonic() >= deadline:
                raise TimeoutError("player did not enter sprint; check hunger, ground state, and movement")
            frame = client.send_action(frame, forward=True, sprint=True)
    frame = wait_for_eligible(client, frame, max(0.1, deadline - time.monotonic()))
    if sprint and not frame["features"].get("player_sprinting", False):
        raise AssertionError("player stopped sprinting before the eligible attack frame")
    check_attack_frame(frame, "eligible attack frame")
    before_age = frame["features"].get("ticks_since_hit_enemy", -1)
    action = client.send_action(frame, forward=sprint, sprint=sprint, attack=True)
    action = client.send_action(action, forward=sprint, sprint=sprint)  # release LMB, keep sprint state
    deadline = time.monotonic() + timeout
    while True:
        age = action["features"].get("ticks_since_hit_enemy", -1)
        if age >= 0 and (before_age < 0 or age < before_age):
            break
        if time.monotonic() >= deadline:
            raise AssertionError(
                f"no new outgoing-hit event within {timeout:g}s "
                f"(ticks_since_hit_enemy before={before_age}, after={age})"
            )
        action = client.next_observation(int(action["tick"]))
    event_age = int(action["features"].get("ticks_since_hit_enemy", 0))
    offset = min(3, max(1, event_age))
    contexts = [(True, True)] + [(False, False)] * (offset - 1)
    result = movement_transition_reward(
        frame, action, False, sprint, False, False, False, sprint,
        prior_attack_context=tuple(contexts),
    )
    knockback, success, reward = result[1], result[5], result[0]
    print(
        f"sprint={sprint} before_tick={frame['tick']} after_tick={action['tick']} "
        f"hit={action['features'].get('has_hit_enemy')} "
        f"hit_age={action['features'].get('ticks_since_hit_enemy')} "
        f"last_hit_type={action['features'].get('last_hit_type')} "
        f"knockback={knockback} attack_success={success} reward={reward:+.3f}"
    )
    if not action["features"].get("has_hit_enemy") or action["features"].get("ticks_since_hit_enemy", -1) < 0:
        raise AssertionError("attack did not produce an outgoing-hit observation; target may be invulnerable/out of reach")
    return action, knockback, success, reward


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="connect to the running Minecraft client")
    parser.add_argument("--allow-attacks", action="store_true", help="permit two real attacks against the current target")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--attempts", type=int, default=3,
                        help="successful normal and sprint-knockback triggers to verify each (default: 3)")
    args = parser.parse_args()
    if not args.live or not args.allow_attacks:
        parser.error("live attacks are opt-in; pass both --live and --allow-attacks in a safe world")
    if args.attempts < 1:
        parser.error("--attempts must be at least 1")

    client = LiveClient(args.host, args.port, args.timeout, wait_for_control=True)
    latest: dict[str, Any] | None = None
    try:
        latest = client.next_observation()
        check_attack_frame(latest, "initial observation")
        normal_rewards: list[float] = []
        for attempt in range(1, args.attempts + 1):
            latest = client.send_action(latest)
            latest, knockback, success, reward = run_attempt(
                client, latest, sprint=False, timeout=args.timeout
            )
            if knockback or not success or reward <= 0:
                raise AssertionError(f"normal hit attempt {attempt} did not receive the expected normal-hit reward")
            normal_rewards.append(reward)
        sprint_rewards: list[float] = []
        for attempt in range(1, args.attempts + 1):
            latest = client.send_action(latest)
            latest, knockback, success, reward = run_attempt(
                client, latest, sprint=True, timeout=args.timeout
            )
            if not success or not knockback:
                raise AssertionError(f"sprint hit attempt {attempt} was not classified as a successful knockback hit")
            if reward <= normal_rewards[-1]:
                raise AssertionError(f"knockback reward {reward} did not exceed normal-hit reward {normal_rewards[-1]}")
            sprint_rewards.append(reward)
        print(f"PASS {len(normal_rewards)} normal hits and {len(sprint_rewards)} sprint knockback hits")
        return 0
    finally:
        if latest is not None:
            try:
                client.send_action(latest)
            except (OSError, RuntimeError, TimeoutError, ValueError, KeyError):
                pass
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
