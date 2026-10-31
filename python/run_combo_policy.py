#!/usr/bin/env python3
"""Run a frozen combo PPO checkpoint in Minecraft without updating it."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import signal
import sys
from typing import Any

import torch

import train_aim_ppo as aim
import train_combo_attack_ppo as combo
import train_yaw_ppo as transport
from aim_geometry import head_aim_error, target_yaw_error_degrees
from combo_ppo_math import (
    MOVEMENT_ACTION_SIZE,
    MOVEMENT_OBSERVATION_SIZE,
    encode_movement_observation,
    movement_transition_reward,
)
from yaw_ppo_math import (
    MAX_PITCH_DELTA_DEGREES,
    MAX_YAW_DELTA_DEGREES,
    PITCH_ONLY_OBSERVATION_SIZE,
    YAW_OBSERVATION_SIZE,
    encode_pitch_observation,
    encode_yaw_observation,
    scripted_sword_attack_eligible,
)

SCHEMA = "minecraftai.pvp.v1"
TRAINER = "combo_attack_ppo_v1"
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "combo_attack_best.pt"


class FrozenComboPolicy:
    """Deterministic inference wrapper for a saved combo checkpoint."""

    def __init__(self, checkpoint_path: Path, device: torch.device) -> None:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"combo checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise RuntimeError(f"{checkpoint_path} does not contain a checkpoint dictionary")

        saved_movement = checkpoint.get("movement_model", {})
        saved_actor_weight = saved_movement.get("actor_logits.weight")
        saved_actor_bias = saved_movement.get("actor_logits.bias")
        action_count = (
            int(saved_actor_weight.shape[0])
            if isinstance(saved_actor_weight, torch.Tensor) and saved_actor_weight.ndim == 2
            else -1
        )
        legacy_movement = action_count != combo.MOVEMENT_ACTION_COUNT + 1
        movement_shape_valid = (
            isinstance(saved_actor_weight, torch.Tensor)
            and tuple(saved_actor_weight.shape) in (
                (combo.MOVEMENT_ACTION_COUNT + 1, 128),
                (5, 128),
                (MOVEMENT_ACTION_SIZE + 1, 128),
                (MOVEMENT_ACTION_SIZE, 128),
            )
            and isinstance(saved_actor_bias, torch.Tensor)
            and tuple(saved_actor_bias.shape) in (
                (combo.MOVEMENT_ACTION_COUNT + 1,),
                (MOVEMENT_ACTION_SIZE + 1,),
                (MOVEMENT_ACTION_SIZE,),
            )
        )
        if (
            checkpoint.get("trainer") != TRAINER
            or checkpoint.get("observation_schema") != SCHEMA
            or int(checkpoint.get("yaw_observation_size", -1)) != YAW_OBSERVATION_SIZE
            or int(checkpoint.get("pitch_observation_size", -1)) != PITCH_ONLY_OBSERVATION_SIZE
            or int(checkpoint.get("movement_observation_size", -1)) != MOVEMENT_OBSERVATION_SIZE
            or int(checkpoint.get("action_size", -1)) not in (
                combo.MOVEMENT_ACTION_COUNT + 1, 5, MOVEMENT_ACTION_SIZE + 1, MOVEMENT_ACTION_SIZE, 3
            )
            or not movement_shape_valid
            or not math.isclose(float(checkpoint.get("max_yaw_delta_degrees", -1)), MAX_YAW_DELTA_DEGREES)
            or not math.isclose(float(checkpoint.get("max_pitch_delta_degrees", -1)), MAX_PITCH_DELTA_DEGREES)
            or int(checkpoint.get("rollout_steps", -1)) != aim.ROLLOUT_STEPS
        ):
            raise RuntimeError(f"{checkpoint_path} is not a compatible {TRAINER} checkpoint")

        self.yaw_model = aim.AxisActorCritic().to(device)
        self.pitch_model = aim.AxisActorCritic().to(device)
        self.movement_model = combo.MovementActorCritic().to(device)
        try:
            self.yaw_model.load_state_dict(checkpoint["yaw_model"], strict=True)
            self.pitch_model.load_state_dict(checkpoint["pitch_model"], strict=True)
            if legacy_movement:
                migrated = self.movement_model.state_dict()
                saved_actions = int(saved_actor_weight.shape[0])
                for key in migrated:
                    value = saved_movement[key]
                    if key in ("actor_logits.weight", "actor_logits.bias"):
                        migrated[key][:MOVEMENT_ACTION_SIZE].copy_(value[:MOVEMENT_ACTION_SIZE])
                        if saved_actions == MOVEMENT_ACTION_SIZE + 1:
                            migrated[key][combo.MOVEMENT_ACTION_COUNT].copy_(value[MOVEMENT_ACTION_SIZE])
                        elif saved_actions == 5:
                            migrated[key][combo.MOVEMENT_ACTION_COUNT].copy_(value[4])
                    else:
                        migrated[key].copy_(value)
                self.movement_model.load_state_dict(migrated, strict=True)
            else:
                self.movement_model.load_state_dict(saved_movement, strict=True)
        except (KeyError, RuntimeError) as error:
            raise RuntimeError(f"checkpoint model layers do not match the combo policy: {error}") from error

        for model in (self.yaw_model, self.pitch_model, self.movement_model):
            model.eval()
        self.device = device
        self.update_number = int(checkpoint.get("update_number", 0))
        self.total_steps = int(checkpoint.get("total_steps", 0))

    @torch.inference_mode()
    def actions(
        self,
        yaw_observation: tuple[float, ...],
        pitch_observation: tuple[float, ...],
        movement_observation: tuple[float, ...],
    ) -> tuple[float, float, bool, bool, bool, bool]:
        yaw_state = torch.as_tensor(
            yaw_observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        pitch_state = torch.as_tensor(
            pitch_observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        movement_state = torch.as_tensor(
            movement_observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        yaw_distribution, _ = self.yaw_model.distribution_and_value(yaw_state)
        pitch_distribution, _ = self.pitch_model.distribution_and_value(pitch_state)
        movement_distribution, _ = self.movement_model.distribution_and_value(movement_state)

        yaw_action = float(torch.tanh(yaw_distribution.loc).item())
        pitch_action = float(torch.tanh(pitch_distribution.loc).item())
        movement_choices = movement_distribution.probs.squeeze(0) >= 0.5
        return (
            yaw_action,
            pitch_action,
            bool(movement_choices[0].item()),
            bool(movement_choices[1].item()),
            bool(movement_choices[2].item()),
            bool(movement_choices[3].item()),
        )


def run(checkpoint_path: Path, host: str, port: int, log_every: int) -> int:
    transport.STOP_REQUESTED = False
    signal.signal(signal.SIGINT, transport.request_stop)
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = FrozenComboPolicy(checkpoint_path, device)
    print(
        f"Loaded frozen combo policy from {checkpoint_path} "
        f"(update={policy.update_number}, training_steps={policy.total_steps}) on {device}.",
        flush=True,
    )
    print(
        "Inference is deterministic: continuous actions use the Gaussian mean; "
        "movement and attack use their most likely Bernoulli actions. No training or checkpoint writes.",
        flush=True,
    )
    print(
        "The policy can move forward, sprint, jump, and attack. A/D strafe is disabled. "
        "Keep it in a suitable world with the intended target visible.",
        flush=True,
    )

    connection: transport.MinecraftConnection | None = None
    last_tick = -1
    previous_yaw_error: float | None = None
    previous_pitch_error: float | None = None
    previous_yaw_action = prior_yaw_action = 0.0
    previous_pitch_action = prior_pitch_action = 0.0
    previous_forward = previous_sprint = False
    forward_streak = sprint_streak = 0
    attack_context: list[tuple[bool, bool]] = []
    knockback_combo = incoming_combo = 0
    applied_steps = stale_retries = superseded_frames = 0
    try:
        connection = transport.MinecraftConnection(host, port)
        while not transport.STOP_REQUESTED:
            if not connection.enabled:
                previous_yaw_error = previous_pitch_error = None
                previous_yaw_action = prior_yaw_action = 0.0
                previous_pitch_action = prior_pitch_action = 0.0
                previous_forward = previous_sprint = False
                forward_streak = sprint_streak = 0
                attack_context.clear()
                knockback_combo = incoming_combo = 0
                print("Policy paused. Press K in Minecraft to resume.", flush=True)
                connection.wait_until_enabled()
                if transport.STOP_REQUESTED:
                    break

            frame = connection.wait_for_observation(after_tick=last_tick)
            if frame is None:
                continue
            while connection.enabled and not transport.STOP_REQUESTED:
                tick = int(frame.get("tick", -1))
                if tick <= last_tick:
                    frame = connection.wait_for_observation(after_tick=last_tick)
                    if frame is None:
                        break
                    continue
                last_tick = tick

                yaw_observation = encode_yaw_observation(
                    frame, previous_yaw_error, previous_yaw_action, prior_yaw_action
                )
                pitch_observation = encode_pitch_observation(
                    frame, previous_pitch_error, previous_pitch_action, prior_pitch_action
                )
                movement_observation = encode_movement_observation(
                    frame, previous_forward, forward_streak, previous_sprint, sprint_streak, knockback_combo
                )
                yaw_action, pitch_action, forward, sprint, jump, attack = policy.actions(
                    yaw_observation, pitch_observation, movement_observation
                )
                eligible = scripted_sword_attack_eligible(frame)

                fresher_frame = connection.take_newer_observation(after_tick=tick)
                if not connection.enabled:
                    break
                if fresher_frame is not None:
                    frame = fresher_frame
                    superseded_frames += 1
                    continue

                control_result = connection.send_aim(
                    tick,
                    yaw_action * MAX_YAW_DELTA_DEGREES,
                    pitch_action * MAX_PITCH_DELTA_DEGREES,
                    attack=attack,
                    forward=forward,
                    sprint=sprint,
                    jump=jump,
                )
                if control_result == "stale":
                    stale_retries += 1
                    previous_yaw_error = previous_pitch_error = None
                    frame = connection.wait_for_observation(after_tick=tick)
                    if frame is None:
                        break
                    continue
                if control_result == "disabled":
                    break

                next_frame = connection.wait_for_applied_observation(tick, int(control_result))
                if next_frame is None:
                    break

                movement_result = movement_transition_reward(
                    frame,
                    next_frame,
                    forward,
                    sprint,
                    attack,
                    eligible,
                    previous_forward,
                    previous_sprint,
                    tuple(attack_context),
                    knockback_combo,
                    forward_streak,
                    sprint_streak,
                    incoming_combo,
                )
                knockback_combo, incoming_combo = movement_result[-2:]

                attack_context.append((attack, eligible))
                if len(attack_context) > 3:
                    attack_context.pop(0)
                forward_streak = forward_streak + 1 if forward == previous_forward else 1
                sprint_streak = sprint_streak + 1 if sprint == previous_sprint else 1
                previous_forward, previous_sprint = forward, sprint
                prior_yaw_action, previous_yaw_action = previous_yaw_action, yaw_action
                prior_pitch_action, previous_pitch_action = previous_pitch_action, pitch_action
                previous_yaw_error = target_yaw_error_degrees(frame)
                previous_pitch_error = head_aim_error(frame)
                frame = next_frame
                applied_steps += 1

                if applied_steps % log_every == 0:
                    features = frame.get("features", {})
                    print(
                        f"inference_steps={applied_steps} tick={tick} "
                        f"camera_pitch={features.get('camera_pitch', float('nan')):+.2f} "
                        f"yaw_action={yaw_action:+.3f} pitch_action={pitch_action:+.3f} "
                        f"forward={int(forward)} sprint={int(sprint)} attack={int(attack)} "
                        f"stale_retries={stale_retries} superseded_frames={superseded_frames}",
                        flush=True,
                    )

        print(f"Policy stopped after {applied_steps} applied inference steps.", flush=True)
        return 0
    except transport.StopRequested:
        print(f"Policy stopped after {applied_steps} applied inference steps.", flush=True)
        return 0
    except (OSError, ConnectionError, TimeoutError, RuntimeError, ValueError) as error:
        print(f"Frozen combo policy stopped: {error}", file=sys.stderr, flush=True)
        return 1
    finally:
        if connection is not None:
            connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="MinecraftAI loopback address")
    parser.add_argument("--port", type=int, default=8766, help="MinecraftAI control port (default: 8766)")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"frozen combo checkpoint (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="print one inference status line after this many applied actions",
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.log_every < 1:
        parser.error("--log-every must be at least 1")
    try:
        return run(args.checkpoint, args.host, args.port, args.log_every)
    except ImportError as error:
        print(
            "Training dependencies are missing. Install them with: "
            "py -3.12 -m pip install -r .\\python\\requirements.txt",
            file=sys.stderr,
        )
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
