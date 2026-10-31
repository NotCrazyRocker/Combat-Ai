#!/usr/bin/env python3
"""Train MinecraftAI's camera-leveling pitch policy with PPO."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

import train_yaw_ppo as transport
from aim_geometry import head_aim_error
from yaw_ppo_math import (
    MAX_PITCH_DELTA_DEGREES,
    PITCH_ONLY_OBSERVATION_SIZE,
    encode_pitch_observation,
    pitch_transition_reward,
)

SCHEMA = "minecraftai.pvp.v1"
ROLLOUT_STEPS = 1024
ACTION_SIZE = 1
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "pitch_ppo_latest.pt"
DEFAULT_YAW_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "yaw_ppo_latest.pt"
PITCH_TRAINER = "vertical_pitch_ppo_v1"


class PitchActorCritic(nn.Module):
    """Single-action PPO model initialized from the yaw checkpoint's trunk."""

    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(PITCH_ONLY_OBSERVATION_SIZE, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(128, ACTION_SIZE)
        self.critic = nn.Linear(128, 1)
        self.log_std = nn.Parameter(torch.tensor([math.log(0.08)], dtype=torch.float32))
        nn.init.zeros_(self.actor_mean.weight)
        nn.init.zeros_(self.actor_mean.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def distribution_and_value(self, observations: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        hidden = self.shared(observations)
        mean = self.actor_mean(hidden)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std), self.critic(hidden).squeeze(-1)

    @staticmethod
    def squashed_log_probability(distribution: Normal, raw_actions: torch.Tensor) -> torch.Tensor:
        squashed = torch.tanh(raw_actions)
        correction = torch.log(1.0 - squashed.pow(2) + 1e-6)
        return (distribution.log_prob(raw_actions) - correction).sum(dim=-1)

    @torch.no_grad()
    def act(self, observation: tuple[float, ...], device: torch.device) -> tuple[float, float, float]:
        state = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        distribution, value = self.distribution_and_value(state)
        raw_action = distribution.sample()
        log_probability = self.squashed_log_probability(distribution, raw_action)
        return (
            float(raw_action.item()),
            float(log_probability.item()),
            float(value.item()),
        )


def request_pitch_stop(_signum: int, _frame: Any) -> None:
    transport.STOP_REQUESTED = True
    print("\nCtrl+C received; saving the latest completed pitch PPO update...", flush=True)


@dataclass
class PitchRollout:
    observations: list[tuple[float, ...]] = field(default_factory=list)
    raw_actions: list[float] = field(default_factory=list)
    log_probabilities: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    signed_errors: list[float] = field(default_factory=list)
    head_projection_valid: list[bool] = field(default_factory=list)
    target_valid: list[bool] = field(default_factory=list)
    on_screen: list[bool] = field(default_factory=list)
    crosshair_hits: list[bool] = field(default_factory=list)
    bbox_overlaps: list[bool] = field(default_factory=list)
    camera_pitches: list[float] = field(default_factory=list)
    pitch_actions_degrees: list[float] = field(default_factory=list)
    corrective_candidates: list[bool] = field(default_factory=list)
    corrective_actions: list[bool] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.rewards)

    def clear(self) -> None:
        self.observations.clear()
        self.raw_actions.clear()
        self.log_probabilities.clear()
        self.values.clear()
        self.rewards.clear()
        self.signed_errors.clear()
        self.head_projection_valid.clear()
        self.target_valid.clear()
        self.on_screen.clear()
        self.crosshair_hits.clear()
        self.bbox_overlaps.clear()
        self.camera_pitches.clear()
        self.pitch_actions_degrees.clear()
        self.corrective_candidates.clear()
        self.corrective_actions.clear()

    def add(
        self,
        observation: tuple[float, ...],
        raw_action: float,
        log_probability: float,
        value: float,
        reward: float,
        before: dict[str, Any],
        normalized_action: float,
    ) -> None:
        self.observations.append(observation)
        self.raw_actions.append(float(raw_action))
        self.log_probabilities.append(float(log_probability))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        features = before.get("features", {})
        self.target_valid.append(bool(features.get("target_valid", False)))
        self.on_screen.append(bool(features.get("target_on_screen", False)))
        self.crosshair_hits.append(bool(features.get("crosshair_on_target", False)))
        try:
            top = float(features.get("target_bbox_top", float("nan")))
            bottom = float(features.get("target_bbox_bottom", float("nan")))
        except (TypeError, ValueError):
            top = bottom = float("nan")
        self.bbox_overlaps.append(
            bool(features.get("target_valid", False))
            and bool(features.get("target_on_screen", False))
            and math.isfinite(top) and math.isfinite(bottom)
            and top <= 0.5 <= bottom
        )
        try:
            camera_pitch = float(features.get("camera_pitch", float("nan")))
        except (TypeError, ValueError):
            camera_pitch = float("nan")
        if math.isfinite(camera_pitch):
            self.camera_pitches.append(camera_pitch)
        error = head_aim_error(before)
        self.head_projection_valid.append(error is not None)
        if error is not None:
            self.signed_errors.append(error)
        self.pitch_actions_degrees.append(normalized_action * MAX_PITCH_DELTA_DEGREES)
        candidate = error is not None and abs(error) > 0.015 and abs(normalized_action) > 0.01
        self.corrective_candidates.append(candidate)
        # Positive Minecraft pitch looks down. A head point below the reticle
        # has positive screen error and needs positive pitch to move it up.
        self.corrective_actions.append(
            candidate and math.copysign(1.0, error) == math.copysign(1.0, normalized_action)
        )


def load_yaw_backbone(path: Path, model: PitchActorCritic, device: torch.device) -> None:
    """Copy only the compatible feature trunk; never import yaw actions or rewards."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("observation_schema") != SCHEMA:
        raise RuntimeError("yaw checkpoint observation schema does not match this mod")
    if (
        checkpoint.get("trainer") != "horizontal_yaw_ppo_v1"
        or int(checkpoint.get("observation_size", -1)) != PITCH_ONLY_OBSERVATION_SIZE
        or int(checkpoint.get("action_size", -1)) != 1
        or float(checkpoint.get("max_yaw_delta_degrees", -1)) != 10.0
    ):
        raise RuntimeError("--init-from must be a compatible horizontal_yaw_ppo_v1 checkpoint")

    saved = checkpoint.get("model", {})
    current = model.state_dict()
    shared_names = {name for name in current if name.startswith("shared.")}
    missing = shared_names - set(saved)
    if missing:
        raise RuntimeError(f"yaw checkpoint is missing shared layers: {sorted(missing)}")
    for name in shared_names:
        if current[name].shape != saved[name].shape:
            raise RuntimeError(f"yaw checkpoint feature layer has incompatible shape: {name}")
        current[name].copy_(saved[name])
    model.load_state_dict(current)
    print(
        f"Initialized pitch feature trunk from {path}; pitch actor mean is zero, "
        "pitch exploration is small, and the yaw checkpoint is unchanged.",
        flush=True,
    )


def save_checkpoint(
    path: Path,
    model: PitchActorCritic,
    optimizer: torch.optim.Optimizer,
    update_number: int,
    total_steps: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    torch.save(
        {
            "format": 1,
            "trainer": PITCH_TRAINER,
            "observation_schema": SCHEMA,
            "observation_size": PITCH_ONLY_OBSERVATION_SIZE,
            "action_size": ACTION_SIZE,
            "max_pitch_delta_degrees": MAX_PITCH_DELTA_DEGREES,
            "rollout_steps": ROLLOUT_STEPS,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "update_number": update_number,
            "total_steps": total_steps,
            "torch_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng_state": np.random.get_state(),
        },
        temporary,
    )
    temporary.replace(path)
    print(f"Saved pitch checkpoint: {path} (update={update_number}, steps={total_steps})", flush=True)


def load_pitch_checkpoint(
    path: Path,
    model: PitchActorCritic,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("trainer") != PITCH_TRAINER or checkpoint.get("observation_schema") != SCHEMA:
        raise RuntimeError("checkpoint is not a compatible pitch-only checkpoint")
    if (
        int(checkpoint.get("observation_size", -1)) != PITCH_ONLY_OBSERVATION_SIZE
        or int(checkpoint.get("action_size", -1)) != ACTION_SIZE
        or float(checkpoint.get("max_pitch_delta_degrees", -1)) != MAX_PITCH_DELTA_DEGREES
        or int(checkpoint.get("rollout_steps", -1)) != ROLLOUT_STEPS
    ):
        raise RuntimeError("pitch checkpoint dimensions or action scale do not match this trainer")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if checkpoint.get("torch_cuda_rng_state") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint["torch_cuda_rng_state"])
    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])
    return int(checkpoint.get("update_number", 0)), int(checkpoint.get("total_steps", 0))


def ppo_update(
    model: PitchActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: PitchRollout,
    bootstrap_value: float,
    device: torch.device,
    entropy_coefficient: float,
) -> dict[str, float]:
    observations = torch.as_tensor(np.asarray(rollout.observations), dtype=torch.float32, device=device)
    raw_actions = torch.as_tensor(rollout.raw_actions, dtype=torch.float32, device=device).unsqueeze(-1)
    old_logp = torch.as_tensor(rollout.log_probabilities, dtype=torch.float32, device=device)
    rewards = np.asarray(rollout.rewards, dtype=np.float32)
    values = np.asarray(rollout.values, dtype=np.float32)

    advantages = np.zeros_like(rewards)
    gae = 0.0
    for index in reversed(range(len(rewards))):
        next_value = bootstrap_value if index == len(rewards) - 1 else values[index + 1]
        delta = rewards[index] + 0.99 * next_value - values[index]
        gae = delta + 0.99 * 0.95 * gae
        advantages[index] = gae
    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    advantage_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=device)
    returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=device)

    metrics: dict[str, list[float]] = {
        "policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_fraction": [],
    }
    sample_count = len(rollout)
    for _ in range(4):
        permutation = torch.randperm(sample_count, device=device)
        for start in range(0, sample_count, 256):
            indices = permutation[start:start + 256]
            distribution, value_prediction = model.distribution_and_value(observations[indices])
            new_logp = model.squashed_log_probability(distribution, raw_actions[indices])
            log_ratio = new_logp - old_logp[indices]
            ratio = log_ratio.exp()
            policy_loss = -torch.minimum(
                ratio * advantage_tensor[indices],
                torch.clamp(ratio, 0.8, 1.2) * advantage_tensor[indices],
            ).mean()
            value_loss = 0.5 * (value_prediction - returns_tensor[indices]).pow(2).mean()
            entropy = -new_logp.mean()
            loss = policy_loss + 0.5 * value_loss - entropy_coefficient * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            with torch.no_grad():
                model.log_std.clamp_(math.log(0.01), math.log(0.30))
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = ((ratio - 1.0).abs() > 0.2).float().mean()
            metrics["policy_loss"].append(float(policy_loss.item()))
            metrics["value_loss"].append(float(value_loss.item()))
            metrics["entropy"].append(float(entropy.item()))
            metrics["approx_kl"].append(float(approx_kl.item()))
            metrics["clip_fraction"].append(float(clip_fraction.item()))
    return {name: float(np.mean(items)) for name, items in metrics.items()}


def log_update(
    update: int,
    total_steps: int,
    rollout: PitchRollout,
    metrics: dict[str, float],
    model: PitchActorCritic,
    entropy_coefficient: float,
    stale_retries: int,
    superseded_observations: int,
) -> None:
    target_valid = 100.0 * sum(rollout.target_valid) / max(1, len(rollout.target_valid))
    on_screen = 100.0 * sum(rollout.on_screen) / max(1, len(rollout.on_screen))
    bbox_overlap = 100.0 * sum(rollout.bbox_overlaps) / max(1, len(rollout.bbox_overlaps))
    crosshair_hit = 100.0 * sum(rollout.crosshair_hits) / max(1, len(rollout.crosshair_hits))
    near_level = 100.0 * sum(abs(pitch) <= 5.0 for pitch in rollout.camera_pitches) / max(1, len(rollout.camera_pitches))
    actions = np.asarray(rollout.pitch_actions_degrees, dtype=np.float32)
    mean_abs_action = float(np.mean(np.abs(actions))) if len(actions) else 0.0
    signed_action = float(np.mean(actions)) if len(actions) else 0.0
    action_std = float(model.log_std.detach().exp().item() * MAX_PITCH_DELTA_DEGREES)
    positive = 100.0 * sum(action > 0.1 for action in actions) / max(1, len(actions))
    negative = 100.0 * sum(action < -0.1 for action in actions) / max(1, len(actions))
    print(
        f"update={update} steps={total_steps} avg_reward={float(np.mean(rollout.rewards)):+.4f} "
        f"target_bbox_vertical_overlap={bbox_overlap:.1f}% crosshair_on_target={crosshair_hit:.1f}% "
        f"mean_camera_pitch_deg={float(np.mean(rollout.camera_pitches)) if rollout.camera_pitches else float('nan'):+.2f} "
        f"pitch_within_5deg={near_level:.1f}% target_valid={target_valid:.1f}% on_screen={on_screen:.1f}% "
        f"mean_abs_pitch_action_deg={mean_abs_action:.3f} mean_signed_pitch_action_deg={signed_action:+.3f} "
        f"pitch_std_deg={action_std:.3f} negative_pitch={negative:.1f}% positive_pitch={positive:.1f}% "
        f"stale_retries={stale_retries} "
        f"superseded_frames={superseded_observations} entropy_coef={entropy_coefficient:.4f} "
        f"policy_loss={metrics['policy_loss']:+.4f} value_loss={metrics['value_loss']:.4f} "
        f"entropy={metrics['entropy']:.4f} approx_kl={metrics['approx_kl']:.5f} "
        f"clip_fraction={metrics['clip_fraction']:.3f}",
        flush=True,
    )


def train(args: argparse.Namespace) -> int:
    transport.STOP_REQUESTED = False
    signal.signal(signal.SIGINT, request_pitch_stop)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}; pitch-only PPO rollout is {ROLLOUT_STEPS} applied steps per update.", flush=True)
    print(
        f"Policy: pitch only (maximum ±{MAX_PITCH_DELTA_DEGREES:.1f}° per tick); yaw delta is always 0, "
        "movement and every other action are disabled; scripted attacks are off.",
        flush=True,
    )
    print(
        "Objective: move the crosshair into the target box, favor confirmed target aim, "
        "and prefer camera pitch near 0° when alignment permits.",
        flush=True,
    )

    model = PitchActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
    update_number = 0
    total_steps = 0
    if not args.no_resume and args.checkpoint.exists():
        update_number, total_steps = load_pitch_checkpoint(args.checkpoint, model, optimizer, device)
        print(f"Resumed pitch-only checkpoint at update {update_number}, step {total_steps}.", flush=True)
    elif args.init_from.exists():
        load_yaw_backbone(args.init_from, model, device)
        print("Starting pitch training at update 0 with a neutral pitch action mean.", flush=True)
    else:
        print("Yaw checkpoint not found; starting a new neutral pitch policy.", flush=True)

    connection: transport.MinecraftConnection | None = None
    rollout = PitchRollout()
    last_tick = -1
    previous_pitch_error: float | None = None
    previous_pitch_action = 0.0
    prior_pitch_action = 0.0
    consecutive_stale = 0
    total_stale_retries = 0
    superseded_observations = 0
    try:
        connection = transport.MinecraftConnection(args.host, args.port)
        while not transport.STOP_REQUESTED:
            if not connection.enabled:
                save_checkpoint(args.checkpoint, model, optimizer, update_number, total_steps)
                rollout.clear()
                previous_pitch_error = None
                previous_pitch_action = 0.0
                prior_pitch_action = 0.0
                print("Training paused. Press K in Minecraft to resume; the pitch policy is preserved.", flush=True)
                connection.wait_until_enabled()
                if transport.STOP_REQUESTED:
                    break
                print("Resuming with a fresh pitch rollout.", flush=True)

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
                observation = encode_pitch_observation(
                    frame, previous_pitch_error, previous_pitch_action, prior_pitch_action
                )
                raw_action, old_log_probability, value = model.act(observation, device)
                normalized_action = float(np.tanh(raw_action))
                pitch_delta = normalized_action * MAX_PITCH_DELTA_DEGREES

                fresher_frame = connection.take_newer_observation(after_tick=tick)
                if not connection.enabled:
                    break
                if fresher_frame is not None:
                    frame = fresher_frame
                    superseded_observations += 1
                    continue

                # The protocol requires both deltas; a literal 0.0 yaw delta
                # guarantees this pitch-only trainer never rotates horizontally.
                control_result = connection.send_aim(
                    tick, yaw_delta=0.0, pitch_delta=pitch_delta, attack=False
                )
                if control_result == "stale":
                    consecutive_stale += 1
                    total_stale_retries += 1
                    if total_stale_retries == 1 or total_stale_retries % 10 == 0:
                        print(
                            f"stale_retry_total={total_stale_retries}; {connection.last_stale_detail}; "
                            "recomputing pitch from a newer observation",
                            flush=True,
                        )
                    frame = connection.wait_for_observation(after_tick=tick)
                    previous_pitch_error = None
                    if frame is None:
                        break
                    if consecutive_stale >= 100:
                        raise RuntimeError("100 consecutive stale actions; check client tick/observation streaming")
                    continue
                if control_result == "disabled":
                    break
                consecutive_stale = 0

                next_frame = connection.wait_for_applied_observation(tick, int(control_result))
                if next_frame is None:
                    break
                reward = pitch_transition_reward(
                    frame, next_frame, normalized_action, previous_pitch_action
                )
                rollout.add(
                    observation, raw_action, old_log_probability, value, reward,
                    frame, normalized_action,
                )
                total_steps += 1
                prior_pitch_action, previous_pitch_action = previous_pitch_action, normalized_action
                previous_pitch_error = head_aim_error(frame)
                frame = next_frame

                if len(rollout) == ROLLOUT_STEPS:
                    next_observation = encode_pitch_observation(
                        frame, previous_pitch_error, previous_pitch_action, prior_pitch_action
                    )
                    with torch.no_grad():
                        bootstrap = float(model.distribution_and_value(
                            torch.as_tensor(next_observation, dtype=torch.float32, device=device).unsqueeze(0)
                        )[1].item())
                    next_update = update_number + 1
                    entropy_coefficient = max(0.001, 0.01 * math.exp(-next_update / 120.0))
                    metrics = ppo_update(model, optimizer, rollout, bootstrap, device, entropy_coefficient)
                    update_number = next_update
                    log_update(
                        update_number, total_steps, rollout, metrics, model, entropy_coefficient,
                        total_stale_retries, superseded_observations,
                    )
                    rollout.clear()
                    if update_number % args.checkpoint_every == 0:
                        save_checkpoint(args.checkpoint, model, optimizer, update_number, total_steps)

        print("Stopping pitch training; saving the latest completed PPO update.", flush=True)
        save_checkpoint(args.checkpoint, model, optimizer, update_number, total_steps)
        return 0
    except transport.StopRequested:
        print("Stop requested; saving the latest completed pitch PPO update.", flush=True)
        save_checkpoint(args.checkpoint, model, optimizer, update_number, total_steps)
        return 0
    except (OSError, ConnectionError, TimeoutError, RuntimeError, ValueError) as error:
        print(f"Pitch training stopped: {error}", file=sys.stderr, flush=True)
        save_checkpoint(args.checkpoint, model, optimizer, update_number, total_steps)
        return 1
    finally:
        if connection is not None:
            connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="MinecraftAI loopback address")
    parser.add_argument("--port", type=int, default=transport.DEFAULT_PORT, help="MinecraftAI control port")
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
        help=f"pitch-only checkpoint (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--init-from", type=Path, default=DEFAULT_YAW_CHECKPOINT,
        help=f"horizontal yaw checkpoint whose shared trunk seeds pitch (default: {DEFAULT_YAW_CHECKPOINT})",
    )
    parser.add_argument("--checkpoint-every", type=int, default=5,
                        help="save after this many completed PPO updates")
    parser.add_argument("--seed", type=int, default=1, help="random seed for a new pitch policy")
    parser.add_argument(
        "--no-resume", action="store_true",
        help="ignore an existing pitch checkpoint and initialize from the yaw checkpoint again",
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be at least 1")
    try:
        return train(args)
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
