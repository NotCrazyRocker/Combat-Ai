import asyncio
import json
import time
from pathlib import Path

import websockets

URI = "ws://127.0.0.1:8765"
PROTOCOL_VERSION = 1
AGENT_ID = "client_player"

RECORDINGS = Path(__file__).resolve().parent.parent / "recordings"
RECORDINGS.mkdir(parents=True, exist_ok=True)


def scripted_policy(observation, tick):
    """
    Temporary scripted policy.

    This is only a bridge test, not the trained AI.
    It advances and sprints, then sends an attack click periodically.
    """

    buttons = {
        "forward": True,
        "backward": False,
        "left": False,
        "right": False,
        "jump": False,
        "sprint": True,
        "sneak": False,
        "attack": False,
        "use_item": False,
    }

    presses = ["attack"] if tick % 10 == 0 else []

    return {
        "valid_for_ticks": 3,
        "buttons": buttons,
        "press": presses,
    }


async def run():
    recording_path = RECORDINGS / (
        f"session-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    )

    action_sequence = 0
    match_id = None

    print(f"Connecting to {URI}")

    with recording_path.open("a", encoding="utf-8") as recording:
        async with websockets.connect(
            URI,
            ping_interval=20,
            close_timeout=2,
            max_size=2**20,
        ) as websocket:

            print(f"Connected. Recording to {recording_path}")

            async for raw in websocket:
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    print("Received invalid JSON; ignoring.")
                    continue

                message_type = message.get("type")

                if message_type == "hello":
                    if message.get("protocol_version") != PROTOCOL_VERSION:
                        raise RuntimeError("Protocol version mismatch.")

                    match_id = message["match_id"]
                    print(f"Match ID: {match_id}")
                    continue

                if message_type == "rejected":
                    print("Mod rejected message:", message.get("reason"))
                    continue

                if message_type != "observation":
                    continue

                if message.get("protocol_version") != PROTOCOL_VERSION:
                    print("Unsupported observation protocol; ignoring.")
                    continue

                if message.get("agent_id") != AGENT_ID:
                    print("Unexpected agent ID; ignoring.")
                    continue

                if match_id is None:
                    match_id = message.get("match_id")

                if message.get("match_id") != match_id:
                    print("Match ID changed; ignoring observation.")
                    continue

                tick = int(message["tick"])
                observation = message["data"]

                # Record the complete incoming observation.
                recording.write(json.dumps({
                    "record_type": "observation",
                    "received_at": time.time(),
                    "message": message,
                }) + "\n")
                recording.flush()

                action_sequence += 1
                action = scripted_policy(observation, tick)

                outgoing = {
                    "protocol_version": PROTOCOL_VERSION,
                    "type": "action",
                    "agent_id": AGENT_ID,
                    "tick": tick,
                    "sequence": action_sequence,
                    "match_id": match_id,
                    "action": action,
                }

                # Record the action before sending it.
                recording.write(json.dumps({
                    "record_type": "action",
                    "sent_at": time.time(),
                    "message": outgoing,
                }) + "\n")
                recording.flush()

                await websocket.send(json.dumps(outgoing))

                if tick % 100 == 0:
                    print(
                        f"Tick {tick}: action {action_sequence} sent"
                    )


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("Agent stopped.")