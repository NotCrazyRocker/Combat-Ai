"""Receive per-tick Minecraft observations over localhost TCP and save JSONL records."""

from __future__ import annotations

import argparse
import json
import socket
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: localhost only)")
    parser.add_argument("--port", type=int, default=8765, help="TCP port; must match the mod (default: 8765)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "recordings",
        help="Directory for timestamped .jsonl recordings",
    )
    return parser.parse_args()


def validate_observation(observation: Any) -> None:
    if not isinstance(observation, dict):
        raise ValueError("Each JSON line must be an object")
    required = ("schema_version", "sequence", "client_tick_sequence", "server_tick", "server_tick_available", "player", "opponent", "derived", "events", "world")
    missing = [key for key in required if key not in observation]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")
    if observation["schema_version"] != "1.0.0":
        raise ValueError(f"Unsupported schema_version: {observation['schema_version']!r}")
    if not isinstance(observation["sequence"], int):
        raise ValueError("sequence must be an integer")
    if not isinstance(observation["server_tick_available"], bool):
        raise ValueError("server_tick_available must be a boolean")
    server_tick = observation["server_tick"]
    if observation["server_tick_available"]:
        if not isinstance(server_tick, int) or isinstance(server_tick, bool):
            raise ValueError("server_tick must be an integer when available")
    elif server_tick is not None:
        raise ValueError("server_tick must be null when unavailable")


def handle_client(connection: socket.socket, address: tuple[str, int], output_path: Path) -> None:
    print(f"Minecraft mod connected: {address[0]}:{address[1]}")
    count = 0
    last_status = time.monotonic()
    started = last_status

    with connection, connection.makefile("r", encoding="utf-8", newline="\n") as reader:
        with output_path.open("a", encoding="utf-8", newline="\n") as output:
            for raw_line in reader:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    observation = json.loads(line)
                    validate_observation(observation)
                except (json.JSONDecodeError, ValueError) as error:
                    print(f"Rejected malformed observation: {error}")
                    continue

                output.write(json.dumps(observation, separators=(",", ":"), ensure_ascii=False) + "\n")
                output.flush()
                count += 1

                now = time.monotonic()
                if now - last_status >= 1.0:
                    player = observation.get("player") or {}
                    derived = observation.get("derived") or {}
                    target_distance = derived.get("distance_blocks")
                    distance_text = "none" if target_distance is None else f"{target_distance:.2f} blocks"
                    rate = count / max(now - started, 0.001)
                    server_tick = observation["server_tick"]
                    server_tick_text = str(server_tick) if observation["server_tick_available"] else "unavailable"
                    print(
                        f"records={count} | ~{rate:.1f}/s | seq={observation['sequence']} "
                        f"| client_tick={observation['client_tick_sequence']} | server_tick={server_tick_text} "
                        f"| hp={player.get('health', '?')} | target_distance={distance_text}"
                    )
                    last_status = now

    print(f"Minecraft connection closed; recorded {count} observations in this session.")


def main() -> None:
    args = parse_args()
    if not (1 <= args.port <= 65535):
        raise SystemExit("Port must be between 1 and 65535")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = args.output_dir / f"observations-{timestamp}.jsonl"

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)
        print(f"Listening on {args.host}:{args.port}")
        print(f"Recording to {output_path}")
        print("Start Minecraft with the mod installed. Press Ctrl+C to stop.")
        try:
            while True:
                connection, address = server.accept()
                handle_client(connection, address, output_path)
        except KeyboardInterrupt:
            print("Receiver stopped.")


if __name__ == "__main__":
    main()
