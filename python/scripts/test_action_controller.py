import json
import socket
import time

HOST = "127.0.0.1"
PORT = 8766
TICK_SECONDS = 0.05  # Approximately 20 updates per second.

BUTTON_NAMES = (
    "forward",
    "backward",
    "left",
    "right",
    "jump",
    "sprint",
    "sneak",
    "attack",
    "use_item",
)


def complete_buttons(**active):
    """Build the complete desired button state."""
    return {
        name: bool(active.get(name, False))
        for name in BUTTON_NAMES
    }


def send_action(
    writer,
    sequence,
    buttons=None,
    press=None,
    aim=None,
    hotbar_slot=None,
):
    message = {
        "protocol_version": 1,
        "type": "action",
        "sequence": sequence,
        "valid_for_ticks": 3,
        "buttons": buttons or complete_buttons(),
        "press": list(press or []),
    }

    if aim is not None:
        message["aim"] = aim

    if hotbar_slot is not None:
        message["hotbar_slot"] = hotbar_slot

    writer.write(json.dumps(message, separators=(",", ":")) + "\n")
    writer.flush()


def run_segment(
    writer,
    sequence,
    name,
    buttons,
    ticks,
    press=(),
    aim=None,
    hotbar_slot=None,
):
    print(f"Testing: {name}")

    for tick in range(ticks):
        send_action(
            writer,
            sequence,
            buttons=buttons,
            press=press if tick == 0 else (),
            aim=aim,
            hotbar_slot=hotbar_slot if tick == 0 else None,
        )

        sequence += 1
        time.sleep(TICK_SECONDS)

    return sequence


def main():
    print(f"Connecting to {HOST}:{PORT}...")
    print("Join a test world before running actions.")

    sequence = 1

    try:
        with socket.create_connection(
            (HOST, PORT), timeout=5
        ) as sock:
            sock.settimeout(None)

            with sock.makefile(
                "w", encoding="utf-8", newline="\n"
            ) as writer:
                print("Connected. Starting scripted test.")

                neutral = complete_buttons()

                sequence = run_segment(
                    writer,
                    sequence,
                    "Neutral / stop",
                    neutral,
                    10,
                )

                # Verify hotbar selection: slots are numbered 0-8.
                for slot in (0, 1, 2):
                    sequence = run_segment(
                        writer,
                        sequence,
                        f"Select hotbar slot {slot}",
                        neutral,
                        2,
                        hotbar_slot=slot,
                    )

                sequence = run_segment(
                    writer,
                    sequence,
                    "Walk forward",
                    complete_buttons(forward=True),
                    30,
                )

                sequence = run_segment(
                    writer,
                    sequence,
                    "Strafe right",
                    complete_buttons(right=True),
                    20,
                )

                sequence = run_segment(
                    writer,
                    sequence,
                    "Sprint forward",
                    complete_buttons(forward=True, sprint=True),
                    30,
                )

                sequence = run_segment(
                    writer,
                    sequence,
                    "Jump once while moving forward",
                    complete_buttons(forward=True),
                    1,
                    press=("jump",),
                )

                sequence = run_segment(
                    writer,
                    sequence,
                    "Continue walking after jump",
                    complete_buttons(forward=True),
                    15,
                )

                sequence = run_segment(
                    writer,
                    sequence,
                    "Rotate aim gradually",
                    neutral,
                    10,
                    aim={
                        "mode": "delta",
                        "yaw": 3.0,
                        "pitch": -1.0,
                    },
                )

                sequence = run_segment(
                    writer,
                    sequence,
                    "Hold attack",
                    complete_buttons(attack=True),
                    15,
                )

                sequence = run_segment(
                    writer,
                    sequence,
                    "Use held item",
                    complete_buttons(use_item=True),
                    10,
                )

                sequence = run_segment(
                    writer,
                    sequence,
                    "Release all controls",
                    neutral,
                    10,
                )

                print("Scripted test completed.")

    except ConnectionRefusedError:
        print(
            "Connection refused. Start Minecraft with the mod first, "
            "then join a world and rerun this script."
        )
    except OSError as exc:
        print(f"Socket error: {exc}")
    except KeyboardInterrupt:
        print("\nTest interrupted. The action timeout should release keys.")


if __name__ == "__main__":
    main()