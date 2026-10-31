#!/usr/bin/env python3
"""Live integration check for MinecraftAI damage_dealt and damage_taken events.

Run in a safe world with another player as the target. The script sends attacks
only when the target is under the crosshair, a sword is held, and the attack is
fully charged. A second player must hit you during the damage-taken phase.

    py -3 scripts/test_damage_detection.py --live --allow-attacks
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import time
from typing import Any


class DamageTestClient:
    def __init__(self, host: str, port: int, timeout: float):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.settimeout(timeout)
        self.reader = self.sock.makefile("rb")
        self.timeout = timeout
        self.sequence = 1
        self.observations: list[dict[str, Any]] = []
        self.acks: dict[int, dict[str, Any]] = {}
        ready = self._read_message()
        if (ready.get("type") != "ready" or ready.get("protocol") != 2
                or ready.get("observation_schema") != "minecraftai.pvp.v2"):
            self.close()
            raise RuntimeError(f"expected MinecraftAI protocol 2 / pvp.v2, received {ready}")
        if not ready.get("ai_control_enabled"):
            self.close()
            raise RuntimeError("AI control is off; press K in Minecraft first")

    def close(self) -> None:
        try:
            self.reader.close()
        finally:
            self.sock.close()

    def _read_message(self) -> dict[str, Any]:
        line = self.reader.readline(65537)
        if not line:
            raise ConnectionError("MinecraftAI closed the connection")
        if len(line) > 65536 or not line.endswith(b"\n"):
            raise ValueError("MinecraftAI sent an invalid or oversized protocol line")
        message = json.loads(line)
        if not isinstance(message, dict):
            raise ValueError("MinecraftAI protocol message must be an object")
        if message.get("type") == "observation":
            self.observations.append(message)
        elif message.get("type") == "ack":
            self.acks[int(message.get("seq", -1))] = message
        elif message.get("type") == "control_state" and not message.get("ai_control_enabled"):
            raise RuntimeError("AI control was switched off during the test")
        return message

    def next_observation(self, after_tick: int = -1, applied_seq: int | None = None) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout
        while True:
            frames = [
                frame for frame in self.observations
                if int(frame.get("tick", -1)) > after_tick
                and (applied_seq is None or int(frame.get("diagnostics", {}).get(
                    "last_applied_sequence", -1
                )) >= applied_seq)
            ]
            if frames:
                frame = min(frames, key=lambda item: int(item["tick"]))
                self.observations.remove(frame)
                return frame
            if time.monotonic() >= deadline:
                raise TimeoutError(f"no observation newer than tick {after_tick}")
            self._read_message()

    def send(self, frame: dict[str, Any], *, attack: bool = False) -> dict[str, Any]:
        seq = self.sequence
        self.sequence += 1
        command = {
            "type": "control", "seq": seq, "tick": int(frame["tick"]),
            "yaw_delta": 0.0, "pitch_delta": 0.0, "attack": attack,
        }
        self.sock.sendall((json.dumps(command, separators=(",", ":")) + "\n").encode("utf-8"))
        deadline = time.monotonic() + self.timeout
        while seq not in self.acks:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"no acknowledgement for sequence {seq}")
            self._read_message()
        ack = self.acks.pop(seq)
        if not ack.get("accepted"):
            raise RuntimeError(f"control rejected: {ack.get('error')}")
        return self.next_observation(int(frame["tick"]), applied_seq=seq)


def event_amount(frame: dict[str, Any], key: str) -> float:
    events = frame.get("events")
    if not isinstance(events, dict) or key not in events:
        raise AssertionError(f"observation is missing events.{key}")
    value = float(events[key])
    if not math.isfinite(value) or value < 0:
        raise AssertionError(f"events.{key} must be finite and nonnegative, got {value}")
    return value


def eligible_to_attack(frame: dict[str, Any]) -> bool:
    features = frame.get("features", {})
    return all((
        bool(features.get("target_valid")),
        bool(features.get("mainhand_is_sword")),
        bool(features.get("crosshair_on_target")),
        float(features.get("attack_strength", 0.0)) >= 1.0,
    ))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="connect to the running Minecraft client")
    parser.add_argument("--allow-attacks", action="store_true", help="allow controlled attacks against the target")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="maximum seconds for each damage phase (default: 60)")
    args = parser.parse_args()
    if not args.live or not args.allow_attacks:
        parser.error("this live test can cause combat; pass both --live and --allow-attacks in a safe world")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    client = DamageTestClient(args.host, args.port, args.timeout)
    try:
        frame = client.next_observation()
        for field in ("damage_dealt", "damage_taken"):
            event_amount(frame, field)
        print("Phase 1: waiting for an eligible target and attempting real hits...")
        deadline = time.monotonic() + args.timeout
        dealt = 0.0
        while time.monotonic() < deadline and dealt <= 0.0:
            attack = eligible_to_attack(frame)
            frame = client.send(frame, attack=attack)
            dealt = event_amount(frame, "damage_dealt")
        if dealt <= 0.0:
            raise AssertionError("no positive events.damage_dealt observed; target may be invulnerable or attacks may miss")
        print(f"PASS damage dealt detected: {dealt:g} health points")

        print("Phase 2: have the other player hit you once; waiting for events.damage_taken...")
        deadline = time.monotonic() + args.timeout
        taken = 0.0
        while time.monotonic() < deadline and taken <= 0.0:
            frame = client.send(frame)
            taken = event_amount(frame, "damage_taken")
        if taken <= 0.0:
            raise AssertionError("no positive events.damage_taken observed before timeout")
        print(f"PASS damage taken detected: {taken:g} health points")
        print("PASS both damage event fields are detected")
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
