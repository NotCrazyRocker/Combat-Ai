#!/usr/bin/env python3
"""Exercise MinecraftAI's localhost control protocol at 20 control updates per second."""

from __future__ import annotations

import argparse
from collections import deque
import json
import socket
import sys
import time

TICK_SECONDS = 1 / 20
YAW_DELTA_PER_TICK = 12.0
TICKS_PER_REVOLUTION = 30


def read_line(sock: socket.socket) -> dict[str, object]:
    data = bytearray()
    while len(data) <= 4096:
        chunk = sock.recv(1)
        if not chunk:
            raise ConnectionError("MinecraftAI closed the control connection")
        if chunk == b"\n":
            return json.loads(data.decode("utf-8"))
        if chunk != b"\r":
            data.extend(chunk)
    raise ValueError("control response exceeded 4096 bytes")


def send_control(
    sock: socket.socket,
    sequence: int,
    yaw_delta: float,
    pitch_delta: float,
    tick: int,
    observations: deque[dict[str, object]],
) -> int:
    command: dict[str, object] = {"type": "control", "seq": sequence, "tick": tick}
    command.update({"yaw_delta": yaw_delta, "pitch_delta": pitch_delta})
    sock.sendall((json.dumps(command, separators=(",", ":")) + "\n").encode("utf-8"))

    while True:
        response = read_line(sock)
        if response.get("type") == "observation":
            observations.append(response)
            continue
        if response.get("type") != "ack" or response.get("seq") != sequence:
            continue
        if not response.get("accepted"):
            raise RuntimeError(str(response.get("error") or "MinecraftAI rejected the command"))
        break
    return sequence + 1


def next_observation(
    sock: socket.socket,
    observations: deque[dict[str, object]],
    after_tick: int = -1,
    applied_sequence: int | None = None,
) -> dict[str, object]:
    def ready(message: dict[str, object]) -> bool:
        if int(message.get("tick", -1)) <= after_tick:
            return False
        if applied_sequence is None:
            return True
        features = message.get("features", {})
        diagnostics = features.get("remote_control_diagnostics", {}) if isinstance(features, dict) else {}
        return int(diagnostics.get("last_applied_sequence", -1)) >= applied_sequence

    while not any(ready(message) for message in observations):
        message = read_line(sock)
        if message.get("type") == "observation":
            observations.append(message)
        elif message.get("type") == "ack" and not message.get("accepted"):
            raise RuntimeError(str(message.get("error") or "MinecraftAI rejected the command"))
    eligible = [message for message in observations if ready(message)]
    latest = max(eligible, key=lambda message: int(message.get("tick", -1)))
    observations.clear()
    return latest


def describe_observation(observation: dict[str, object]) -> str:
    features = observation.get("features", {})
    if not isinstance(features, dict):
        return f"tick {observation.get('tick')}: malformed features"
    if not features.get("target_valid"):
        return f"tick {observation.get('tick')}: no visible opponent"
    return (
        f"tick {observation.get('tick')}: target dx={float(features.get('target_dx', 0)):+.3f} "
        f"dy={float(features.get('target_dy', 0)):+.3f}, "
        f"crosshair={bool(features.get('crosshair_on_target'))}, "
        f"distance={float(features.get('target_distance', 0)):.2f}, "
        f"last hit={features.get('last_hit_type', 'none')}"
    )


def run(host: str, port: int) -> None:
    sequence = 1
    sock = socket.create_connection((host, port), timeout=3.0)
    sock.settimeout(1.5)
    observations: deque[dict[str, object]] = deque()
    observation: dict[str, object] = {"tick": 0}
    release_sent = False

    try:
        ready = read_line(sock)
        if ready.get("type") != "ready" or ready.get("protocol") != 2:
            raise RuntimeError(f"unsupported MinecraftAI control server response: {ready}")
        if not ready.get("ai_control_enabled"):
            raise RuntimeError("AI control is off. Join a world and press K to enable it, then run this script again.")

        print(f"Connected to MinecraftAI at {host}:{port}; sending at 20 Hz.")
        print("Sending a 360-degree horizontal turn: +12 degrees per tick for 30 ticks.")
        print("Pitch, movement, and attack/use actions stay released.")
        observation = next_observation(sock, observations)
        print(describe_observation(observation))

        next_frame = time.monotonic()
        for step in range(TICKS_PER_REVOLUTION):
            if observation.get("type") != "observation":
                observation = next_observation(sock, observations)
            source_tick = int(observation["tick"])
            sequence = send_control(sock, sequence, YAW_DELTA_PER_TICK, 0.0, source_tick, observations)
            observation = next_observation(sock, observations, source_tick, sequence - 1)
            if (step + 1) % 5 == 0:
                print(f"Turn progress: {(step + 1) * YAW_DELTA_PER_TICK:.0f} degrees")
            next_frame += TICK_SECONDS
            delay = next_frame - time.monotonic()
            if delay > 0:
                time.sleep(delay)

        source_tick = int(observation["tick"])
        sequence = send_control(sock, sequence, 0.0, 0.0, source_tick, observations)
        observation = next_observation(sock, observations, source_tick, sequence - 1)
        release_sent = True
        print("Aiming demo finished; no movement or action buttons were pressed.")
    finally:
        try:
            if sock.fileno() >= 0 and not release_sent:
                send_control(sock, sequence, 0.0, 0.0, int(observation["tick"]), observations)
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
            pass
        sock.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="MinecraftAI's loopback address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="MinecraftAI control port (default: 8765)")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    try:
        run(args.host, args.port)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"Control test failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
