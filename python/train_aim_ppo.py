#!/usr/bin/env python3
"""Train the pretrained yaw and pitch policies together against a moving target."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import signal
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

import train_yaw_ppo as transport
from aim_geometry import head_aim_error, target_yaw_error_degrees
from yaw_ppo_math import (
    MAX_PITCH_DELTA_DEGREES,
    MAX_YAW_DELTA_DEGREES,
    PITCH_ONLY_OBSERVATION_SIZE,
    YAW_OBSERVATION_SIZE,
    encode_pitch_observation,
    encode_yaw_observation,
    scripted_attack_decision,
    yaw_pitch_transition_rewards,
)

SCHEMA = "minecraftai.pvp.v1"
ROLLOUT_STEPS = 1024
ACTION_SIZE = 2
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "aim_ppo_latest.pt"
DEFAULT_BEST_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "aim_ppo_best.pt"
DEFAULT_YAW_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "yaw_ppo_latest.pt"
DEFAULT_PITCH_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "pitch_ppo_latest.pt"
TRAINER = "joint_aim_ppo_v1"


class AxisActorCritic(nn.Module):
    """One independently trained aim axis with the legacy 9-input layout."""

    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(PITCH_ONLY_OBSERVATION_SIZE, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(128, 1)
        self.critic = nn.Linear(128, 1)
        self.log_std = nn.Parameter(torch.tensor([math.log(0.08)], dtype=torch.float32))
        nn.init.zeros_(self.actor_mean.weight)
        nn.init.zeros_(self.actor_mean.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def distribution_and_value(self, observations: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        hidden = self.shared(observations)
        mean = self.actor_mean(hidden)
        return Normal(mean, self.log_std.exp().expand_as(mean)), self.critic(hidden).squeeze(-1)

    @staticmethod
    def squashed_log_probability(distribution: Normal, raw_actions: torch.Tensor) -> torch.Tensor:
        squashed = torch.tanh(raw_actions)
        correction = torch.log(1.0 - squashed.square() + 1e-6)
        return (distribution.log_prob(raw_actions) - correction).sum(dim=-1)

    @torch.no_grad()
    def act(
        self, observation: tuple[float, ...], device: torch.device
    ) -> tuple[float, float, float]:
        state = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        distribution, value = self.distribution_and_value(state)
        raw_action = distribution.sample()
        log_probability = self.squashed_log_probability(distribution, raw_action)
        return float(raw_action.item()), float(log_probability.item()), float(value.item())


@dataclass
class AimRollout:
    yaw_observations: list[tuple[float, ...]] = field(default_factory=list)
    pitch_observations: list[tuple[float, ...]] = field(default_factory=list)
    yaw_raw_actions: list[float] = field(default_factory=list)
    pitch_raw_actions: list[float] = field(default_factory=list)
    yaw_log_probabilities: list[float] = field(default_factory=list)
    pitch_log_probabilities: list[float] = field(default_factory=list)
    yaw_values: list[float] = field(default_factory=list)
    pitch_values: list[float] = field(default_factory=list)
    yaw_rewards: list[float] = field(default_factory=list)
    pitch_rewards: list[float] = field(default_factory=list)
    yaw_errors: list[float] = field(default_factory=list)
    pitch_errors: list[float] = field(default_factory=list)
    signed_pitch_errors: list[float] = field(default_factory=list)
    target_valid: list[bool] = field(default_factory=list)
    yaw_projection_valid: list[bool] = field(default_factory=list)
    head_projection_valid: list[bool] = field(default_factory=list)
    on_screen: list[bool] = field(default_factory=list)
    crosshair_hits: list[bool] = field(default_factory=list)
    pitch_box_overlaps: list[bool] = field(default_factory=list)
    camera_pitches: list[float] = field(default_factory=list)
    yaw_actions_degrees: list[float] = field(default_factory=list)
    pitch_actions_degrees: list[float] = field(default_factory=list)
    yaw_correction_candidates: list[bool] = field(default_factory=list)
    yaw_corrective_actions: list[bool] = field(default_factory=list)
    pitch_correction_candidates: list[bool] = field(default_factory=list)
    pitch_corrective_actions: list[bool] = field(default_factory=list)
    scripted_attacks: int = 0

    def __len__(self) -> int:
        return len(self.yaw_rewards)

    def clear(self) -> None:
        for value in vars(self).values():
            if isinstance(value, list):
                value.clear()
        self.scripted_attacks = 0

    def add(
        self,
        yaw_observation: tuple[float, ...],
        pitch_observation: tuple[float, ...],
        yaw_raw_action: float,
        pitch_raw_action: float,
        yaw_log_probability: float,
        pitch_log_probability: float,
        yaw_value: float,
        pitch_value: float,
        yaw_reward: float,
        pitch_reward: float,
        before: dict[str, Any],
        yaw_action: float,
        pitch_action: float,
        scripted_attack: bool,
    ) -> None:
        self.yaw_observations.append(yaw_observation)
        self.pitch_observations.append(pitch_observation)
        self.yaw_raw_actions.append(yaw_raw_action)
        self.pitch_raw_actions.append(pitch_raw_action)
        self.yaw_log_probabilities.append(yaw_log_probability)
        self.pitch_log_probabilities.append(pitch_log_probability)
        self.yaw_values.append(yaw_value)
        self.pitch_values.append(pitch_value)
        self.yaw_rewards.append(yaw_reward)
        self.pitch_rewards.append(pitch_reward)

        features = before.get("features", {})
        self.target_valid.append(bool(features.get("target_valid", False)))
        self.on_screen.append(bool(features.get("target_on_screen", False)))
        self.crosshair_hits.append(bool(features.get("crosshair_on_target", False)))
        try:
            box_top = float(features.get("target_bbox_top", float("nan")))
            box_bottom = float(features.get("target_bbox_bottom", float("nan")))
        except (TypeError, ValueError):
            box_top = box_bottom = float("nan")
        self.pitch_box_overlaps.append(
            bool(features.get("target_valid", False))
            and bool(features.get("target_on_screen", False))
            and math.isfinite(box_top) and math.isfinite(box_bottom)
            and box_top <= 0.5 <= box_bottom
        )
        try:
            camera_pitch = float(features.get("camera_pitch", float("nan")))
        except (TypeError, ValueError):
            camera_pitch = float("nan")
        if math.isfinite(camera_pitch):
            self.camera_pitches.append(camera_pitch)
        yaw_error = target_yaw_error_degrees(before)
        self.yaw_projection_valid.append(yaw_error is not None)
        if yaw_error is not None:
            self.yaw_errors.append(abs(yaw_error))
        pitch_error = head_aim_error(before)
        self.head_projection_valid.append(pitch_error is not None)
        if pitch_error is not None:
            self.pitch_errors.append(abs(pitch_error))
            self.signed_pitch_errors.append(pitch_error)

        self.yaw_actions_degrees.append(yaw_action * MAX_YAW_DELTA_DEGREES)
        self.pitch_actions_degrees.append(pitch_action * MAX_PITCH_DELTA_DEGREES)
        yaw_candidate = yaw_error is not None and abs(yaw_error) > 1.0 and abs(yaw_action) > 0.01
        self.yaw_correction_candidates.append(yaw_candidate)
        self.yaw_corrective_actions.append(
            yaw_candidate and math.copysign(1.0, yaw_error) == math.copysign(1.0, yaw_action)
        )
        pitch_candidate = (
            pitch_error is not None and abs(pitch_error) > 0.015 and abs(pitch_action) > 0.01
        )
        self.pitch_correction_candidates.append(pitch_candidate)
        self.pitch_corrective_actions.append(
            pitch_candidate and math.copysign(1.0, pitch_error) == math.copysign(1.0, pitch_action)
        )
        self.scripted_attacks += int(scripted_attack)


def _validate_seed_checkpoint(
    checkpoint: dict[str, Any],
    *,
    path: Path,
    expected_trainer: str,
    action_scale_key: str,
    action_scale: float,
) -> None:
    if checkpoint.get("observation_schema") != SCHEMA:
        raise RuntimeError(f"{path} uses an observation schema incompatible with this mod")
    if (
        checkpoint.get("trainer") != expected_trainer
        or int(checkpoint.get("observation_size", -1)) != YAW_OBSERVATION_SIZE
        or int(checkpoint.get("action_size", -1)) != 1
        or int(checkpoint.get("rollout_steps", -1)) != ROLLOUT_STEPS
        or not math.isclose(float(checkpoint.get(action_scale_key, -1)), action_scale)
    ):
        raise RuntimeError(f"{path} is not a compatible {expected_trainer} checkpoint")


def initialize_axis(
    path: Path,
    model: AxisActorCritic,
    device: torch.device,
    *,
    expected_trainer: str,
    action_scale_key: str,
    action_scale: float,
    label: str,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"required pretrained {label} checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    _validate_seed_checkpoint(
        checkpoint,
        path=path,
        expected_trainer=expected_trainer,
        action_scale_key=action_scale_key,
        action_scale=action_scale,
    )
    try:
        model.load_state_dict(checkpoint["model"], strict=True)
    except (KeyError, RuntimeError) as error:
        raise RuntimeError(f"{label} checkpoint model layers do not match the expected policy: {error}") from error


def save_checkpoint(
    path: Path,
    yaw_model: AxisActorCritic,
    pitch_model: AxisActorCritic,
    yaw_optimizer: torch.optim.Optimizer,
    pitch_optimizer: torch.optim.Optimizer,
    update_number: int,
    total_steps: int,
    selection_score: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    torch.save(
        {
            "format": 1,
            "trainer": TRAINER,
            "reward_layout": "independent_yaw_pitch",
            "observation_schema": SCHEMA,
            "yaw_observation_size": YAW_OBSERVATION_SIZE,
            "pitch_observation_size": PITCH_ONLY_OBSERVATION_SIZE,
            "action_size": ACTION_SIZE,
            "max_yaw_delta_degrees": MAX_YAW_DELTA_DEGREES,
            "max_pitch_delta_degrees": MAX_PITCH_DELTA_DEGREES,
            "rollout_steps": ROLLOUT_STEPS,
            "yaw_model": yaw_model.state_dict(),
            "pitch_model": pitch_model.state_dict(),
            "yaw_optimizer": yaw_optimizer.state_dict(),
            "pitch_optimizer": pitch_optimizer.state_dict(),
            "update_number": update_number,
            "total_steps": total_steps,
            "selection_score": selection_score,
            "torch_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng_state": np.random.get_state(),
        },
        temporary,
    )
    temporary.replace(path)
    print(f"Saved aim checkpoint: {path} (update={update_number}, steps={total_steps})", flush=True)


def load_best_score(path: Path, device: torch.device) -> float:
    """Read the best rollout score saved with a prior best checkpoint."""
    if not path.is_file():
        return -math.inf
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if checkpoint.get("trainer") != TRAINER:
            return -math.inf
        score = float(checkpoint.get("selection_score", float("-inf")))
        return score if math.isfinite(score) else -math.inf
    except (OSError, RuntimeError, ValueError, TypeError, EOFError) as error:
        print(f"Could not read prior best-checkpoint score from {path}: {error}; a new best will be selected.", flush=True)
        return -math.inf


def load_checkpoint(
    path: Path,
    yaw_model: AxisActorCritic,
    pitch_model: AxisActorCritic,
    yaw_optimizer: torch.optim.Optimizer,
    pitch_optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if (
        checkpoint.get("trainer") != TRAINER
        or checkpoint.get("observation_schema") != SCHEMA
        or int(checkpoint.get("yaw_observation_size", -1)) != YAW_OBSERVATION_SIZE
        or int(checkpoint.get("pitch_observation_size", -1)) != PITCH_ONLY_OBSERVATION_SIZE
        or int(checkpoint.get("action_size", -1)) != ACTION_SIZE
        or not math.isclose(float(checkpoint.get("max_yaw_delta_degrees", -1)), MAX_YAW_DELTA_DEGREES)
        or not math.isclose(float(checkpoint.get("max_pitch_delta_degrees", -1)), MAX_PITCH_DELTA_DEGREES)
        or int(checkpoint.get("rollout_steps", -1)) != ROLLOUT_STEPS
    ):
        raise RuntimeError(f"{path} is not a compatible {TRAINER} checkpoint")
    yaw_model.load_state_dict(checkpoint["yaw_model"], strict=True)
    pitch_model.load_state_dict(checkpoint["pitch_model"], strict=True)
    yaw_optimizer.load_state_dict(checkpoint["yaw_optimizer"])
    pitch_optimizer.load_state_dict(checkpoint["pitch_optimizer"])
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if checkpoint.get("torch_cuda_rng_state") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint["torch_cuda_rng_state"])
    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])
    return int(checkpoint.get("update_number", 0)), int(checkpoint.get("total_steps", 0))


def ppo_update_axis(
    model: AxisActorCritic,
    optimizer: torch.optim.Optimizer,
    observations: list[tuple[float, ...]],
    raw_actions: list[float],
    old_log_probabilities: list[float],
    rewards: list[float],
    values: list[float],
    bootstrap_value: float,
    device: torch.device,
    entropy_coefficient: float,
) -> dict[str, float]:
    observation_array = np.asarray(observations, dtype=np.float32)
    raw_action_array = np.asarray(raw_actions, dtype=np.float32)
    old_log_probability_array = np.asarray(old_log_probabilities, dtype=np.float32)
    obs = torch.as_tensor(observation_array, dtype=torch.float32, device=device)
    actions = torch.as_tensor(raw_action_array, dtype=torch.float32, device=device).unsqueeze(-1)
    old_logp = torch.as_tensor(old_log_probability_array, dtype=torch.float32, device=device)
    reward_array = np.asarray(rewards, dtype=np.float32)
    value_array = np.asarray(values, dtype=np.float32)
    if (
        len(reward_array) != ROLLOUT_STEPS
        or value_array.shape != reward_array.shape
        or actions.shape != (ROLLOUT_STEPS, 1)
        or old_logp.shape != reward_array.shape
        or obs.shape != (ROLLOUT_STEPS, PITCH_ONLY_OBSERVATION_SIZE)
        or not math.isfinite(bootstrap_value)
        or not all(np.isfinite(array).all() for array in (
            observation_array, raw_action_array, old_log_probability_array, reward_array, value_array
        ))
    ):
        raise ValueError("aim PPO rollout contains invalid or non-finite data")

    advantages = np.zeros_like(reward_array)
    gae = 0.0
    for index in reversed(range(ROLLOUT_STEPS)):
        next_value = bootstrap_value if index == ROLLOUT_STEPS - 1 else value_array[index + 1]
        delta = reward_array[index] + 0.99 * next_value - value_array[index]
        gae = delta + 0.99 * 0.95 * gae
        advantages[index] = gae
    returns = advantages + value_array
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    advantage_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=device)
    return_tensor = torch.as_tensor(returns, dtype=torch.float32, device=device)

    metrics: dict[str, list[float]] = {
        "policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_fraction": [],
    }
    for _ in range(4):
        permutation = torch.randperm(ROLLOUT_STEPS, device=device)
        for start in range(0, ROLLOUT_STEPS, 256):
            indices = permutation[start:start + 256]
            distribution, value_prediction = model.distribution_and_value(obs[indices])
            new_logp = model.squashed_log_probability(distribution, actions[indices])
            log_ratio = new_logp - old_logp[indices]
            ratio = log_ratio.exp()
            policy_loss = -torch.minimum(
                ratio * advantage_tensor[indices],
                torch.clamp(ratio, 0.8, 1.2) * advantage_tensor[indices],
            ).mean()
            value_loss = 0.5 * (value_prediction - return_tensor[indices]).square().mean()
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


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def log_update(
    update: int,
    total_steps: int,
    rollout: AimRollout,
    yaw_metrics: dict[str, float],
    pitch_metrics: dict[str, float],
    yaw_model: AxisActorCritic,
    pitch_model: AxisActorCritic,
    entropy_coefficient: float,
    stale_retries: int,
    superseded_frames: int,
) -> None:
    yaw_errors = rollout.yaw_errors
    yaw_center_2 = 100.0 * sum(error <= 2.0 for error in yaw_errors) / max(1, len(yaw_errors))
    yaw_center_5 = 100.0 * sum(error <= 5.0 for error in yaw_errors) / max(1, len(yaw_errors))
    target_pct = 100.0 * sum(rollout.target_valid) / ROLLOUT_STEPS
    yaw_projected_pct = 100.0 * sum(rollout.yaw_projection_valid) / ROLLOUT_STEPS
    on_screen_pct = 100.0 * sum(rollout.on_screen) / ROLLOUT_STEPS
    crosshair_pct = 100.0 * sum(rollout.crosshair_hits) / ROLLOUT_STEPS
    pitch_box_pct = 100.0 * sum(rollout.pitch_box_overlaps) / ROLLOUT_STEPS
    pitch_near_level_pct = 100.0 * sum(abs(pitch) <= 5.0 for pitch in rollout.camera_pitches) / max(1, len(rollout.camera_pitches))
    yaw_actions = np.asarray(rollout.yaw_actions_degrees, dtype=np.float32)
    pitch_actions = np.asarray(rollout.pitch_actions_degrees, dtype=np.float32)
    yaw_corrective = 100.0 * sum(rollout.yaw_corrective_actions) / max(
        1, sum(rollout.yaw_correction_candidates)
    )
    stale_pct = 100.0 * stale_retries / max(1, total_steps + stale_retries)
    total_rewards = np.asarray(rollout.yaw_rewards) + np.asarray(rollout.pitch_rewards)
    print(
        f"update={update} steps={total_steps} avg_reward={float(total_rewards.mean()):+.4f} "
        f"avg_yaw_reward={_mean(rollout.yaw_rewards):+.4f} avg_pitch_reward={_mean(rollout.pitch_rewards):+.4f} "
        f"mean_abs_yaw_error_deg={_mean(yaw_errors):.2f} "
        f"median_yaw_error_deg={float(np.median(yaw_errors)) if yaw_errors else float('nan'):.2f} "
        f"centered_2deg={yaw_center_2:.1f}% centered_5deg={yaw_center_5:.1f}% "
        f"pitch_bbox_vertical_overlap={pitch_box_pct:.1f}% pitch_crosshair_on_target={crosshair_pct:.1f}% "
        f"mean_camera_pitch_deg={_mean(rollout.camera_pitches):+.2f} pitch_within_5deg={pitch_near_level_pct:.1f}% "
        f"target_valid={target_pct:.1f}% yaw_projected={yaw_projected_pct:.1f}% "
        f"on_screen={on_screen_pct:.1f}% "
        f"scripted_attacks={rollout.scripted_attacks} "
        f"mean_abs_yaw_action_deg={float(np.mean(np.abs(yaw_actions))):.2f} "
        f"mean_abs_pitch_action_deg={float(np.mean(np.abs(pitch_actions))):.2f} "
        f"yaw_std_deg={float(yaw_model.log_std.detach().exp().item() * MAX_YAW_DELTA_DEGREES):.3f} "
        f"pitch_std_deg={float(pitch_model.log_std.detach().exp().item() * MAX_PITCH_DELTA_DEGREES):.3f} "
        f"yaw_corrective_direction={yaw_corrective:.1f}% "
        f"stale_retries={stale_retries} ({stale_pct:.3f}%) superseded_frames={superseded_frames} "
        f"entropy_coef={entropy_coefficient:.4f} "
        f"yaw_policy_loss={yaw_metrics['policy_loss']:+.4f} pitch_policy_loss={pitch_metrics['policy_loss']:+.4f} "
        f"yaw_value_loss={yaw_metrics['value_loss']:.4f} pitch_value_loss={pitch_metrics['value_loss']:.4f} "
        f"yaw_approx_kl={yaw_metrics['approx_kl']:.5f} pitch_approx_kl={pitch_metrics['approx_kl']:.5f} "
        f"yaw_clip_fraction={yaw_metrics['clip_fraction']:.3f} "
        f"pitch_clip_fraction={pitch_metrics['clip_fraction']:.3f}",
        flush=True,
    )


def train(args: argparse.Namespace) -> int:
    transport.STOP_REQUESTED = False
    signal.signal(signal.SIGINT, transport.request_stop)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_score = load_best_score(args.best_checkpoint, device)
    print(f"Using {device}; aim PPO rollout size is exactly {ROLLOUT_STEPS} applied steps per update.", flush=True)
    print(
        f"Policy: yaw ±{MAX_YAW_DELTA_DEGREES:.1f}° and pitch ±{MAX_PITCH_DELTA_DEGREES:.1f}° per tick; "
        "both aim axes are learned independently; movement and non-attack actions are disabled. "
        "Scripted sword attack requires a valid target, sword, full cooldown, and crosshair hit.",
        flush=True,
    )
    print(
        "Pitch objective: favor vertical crosshair overlap with the target box; "
        "reward camera pitch near 0° on every applied tick.",
        flush=True,
    )
    if math.isfinite(best_score):
        print(f"Best-checkpoint selection resumes at avg_reward={best_score:+.4f}.", flush=True)
    else:
        print("A best checkpoint will be saved after the first completed rollout.", flush=True)

    yaw_model = AxisActorCritic().to(device)
    pitch_model = AxisActorCritic().to(device)
    yaw_optimizer = torch.optim.Adam(yaw_model.parameters(), lr=3e-4, eps=1e-5)
    pitch_optimizer = torch.optim.Adam(pitch_model.parameters(), lr=3e-4, eps=1e-5)
    update_number = 0
    total_steps = 0
    if not args.no_resume and args.checkpoint.exists():
        update_number, total_steps = load_checkpoint(
            args.checkpoint, yaw_model, pitch_model, yaw_optimizer, pitch_optimizer, device
        )
        print(f"Resumed aim checkpoint at update {update_number}, step {total_steps}.", flush=True)
    else:
        initialize_axis(
            args.init_yaw_from, yaw_model, device,
            expected_trainer="horizontal_yaw_ppo_v1",
            action_scale_key="max_yaw_delta_degrees",
            action_scale=MAX_YAW_DELTA_DEGREES,
            label="yaw",
        )
        initialize_axis(
            args.init_pitch_from, pitch_model, device,
            expected_trainer="vertical_pitch_ppo_v1",
            action_scale_key="max_pitch_delta_degrees",
            action_scale=MAX_PITCH_DELTA_DEGREES,
            label="pitch",
        )
        print(
            f"Initialized yaw from {args.init_yaw_from} and pitch from {args.init_pitch_from}; "
            f"fresh optimizer state; future saves go to {args.checkpoint}.",
            flush=True,
        )

    connection: transport.MinecraftConnection | None = None
    rollout = AimRollout()
    last_tick = -1
    previous_yaw_error: float | None = None
    previous_pitch_error: float | None = None
    previous_yaw_action = prior_yaw_action = 0.0
    previous_pitch_action = prior_pitch_action = 0.0
    attack_latched = False
    consecutive_stale = 0
    total_stale_retries = 0
    superseded_frames = 0
    try:
        connection = transport.MinecraftConnection(args.host, args.port)
        while not transport.STOP_REQUESTED:
            if not connection.enabled:
                save_checkpoint(
                    args.checkpoint, yaw_model, pitch_model,
                    yaw_optimizer, pitch_optimizer, update_number, total_steps,
                )
                rollout.clear()
                previous_yaw_error = previous_pitch_error = None
                previous_yaw_action = prior_yaw_action = 0.0
                previous_pitch_action = prior_pitch_action = 0.0
                attack_latched = False
                print("Training paused. Press K in Minecraft to resume; both policies are preserved.", flush=True)
                connection.wait_until_enabled()
                if transport.STOP_REQUESTED:
                    break
                print("Resuming with a fresh rollout.", flush=True)

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
                yaw_raw_action, yaw_old_logp, yaw_value = yaw_model.act(yaw_observation, device)
                pitch_raw_action, pitch_old_logp, pitch_value = pitch_model.act(pitch_observation, device)
                yaw_action = float(np.tanh(yaw_raw_action))
                pitch_action = float(np.tanh(pitch_raw_action))
                yaw_delta = yaw_action * MAX_YAW_DELTA_DEGREES
                pitch_delta = pitch_action * MAX_PITCH_DELTA_DEGREES
                attack_press, next_attack_latched = scripted_attack_decision(frame, attack_latched)

                fresher_frame = connection.take_newer_observation(after_tick=tick)
                if not connection.enabled:
                    break
                if fresher_frame is not None:
                    frame = fresher_frame
                    superseded_frames += 1
                    continue

                control_result = connection.send_aim(
                    tick, yaw_delta, pitch_delta, attack=attack_press
                )
                if control_result == "stale":
                    consecutive_stale += 1
                    total_stale_retries += 1
                    if total_stale_retries == 1 or total_stale_retries % 10 == 0:
                        print(
                            f"stale_retry_total={total_stale_retries}; "
                            f"{connection.last_stale_detail}; recomputing both axes",
                            flush=True,
                        )
                    frame = connection.wait_for_observation(after_tick=tick)
                    previous_yaw_error = previous_pitch_error = None
                    if frame is None:
                        break
                    if consecutive_stale >= 100:
                        raise RuntimeError("100 consecutive stale actions; check client tick/observation streaming")
                    continue
                if control_result == "disabled":
                    break
                consecutive_stale = 0
                attack_latched = next_attack_latched

                next_frame = connection.wait_for_applied_observation(tick, int(control_result))
                if next_frame is None:
                    break
                yaw_reward, pitch_reward = yaw_pitch_transition_rewards(
                    frame, next_frame, yaw_action, previous_yaw_action,
                    pitch_action, previous_pitch_action,
                    scripted_attack=attack_press,
                )
                rollout.add(
                    yaw_observation, pitch_observation,
                    yaw_raw_action, pitch_raw_action,
                    yaw_old_logp, pitch_old_logp, yaw_value, pitch_value,
                    yaw_reward, pitch_reward, frame, yaw_action, pitch_action, attack_press,
                )
                total_steps += 1
                prior_yaw_action, previous_yaw_action = previous_yaw_action, yaw_action
                prior_pitch_action, previous_pitch_action = previous_pitch_action, pitch_action
                previous_yaw_error = target_yaw_error_degrees(frame)
                previous_pitch_error = head_aim_error(frame)
                frame = next_frame

                if len(rollout) == ROLLOUT_STEPS:
                    next_yaw_observation = encode_yaw_observation(
                        frame, previous_yaw_error, previous_yaw_action, prior_yaw_action
                    )
                    next_pitch_observation = encode_pitch_observation(
                        frame, previous_pitch_error, previous_pitch_action, prior_pitch_action
                    )
                    with torch.no_grad():
                        yaw_bootstrap = float(yaw_model.distribution_and_value(
                            torch.as_tensor(next_yaw_observation, dtype=torch.float32, device=device).unsqueeze(0)
                        )[1].item())
                        pitch_bootstrap = float(pitch_model.distribution_and_value(
                            torch.as_tensor(next_pitch_observation, dtype=torch.float32, device=device).unsqueeze(0)
                        )[1].item())
                    next_update = update_number + 1
                    entropy_coefficient = max(0.001, 0.01 * math.exp(-next_update / 120.0))
                    yaw_metrics = ppo_update_axis(
                        yaw_model, yaw_optimizer, rollout.yaw_observations,
                        rollout.yaw_raw_actions, rollout.yaw_log_probabilities,
                        rollout.yaw_rewards, rollout.yaw_values, yaw_bootstrap,
                        device, entropy_coefficient,
                    )
                    pitch_metrics = ppo_update_axis(
                        pitch_model, pitch_optimizer, rollout.pitch_observations,
                        rollout.pitch_raw_actions, rollout.pitch_log_probabilities,
                        rollout.pitch_rewards, rollout.pitch_values, pitch_bootstrap,
                        device, entropy_coefficient,
                    )
                    update_number = next_update
                    selection_score = float(np.mean(rollout.yaw_rewards) + np.mean(rollout.pitch_rewards))
                    log_update(
                        update_number, total_steps, rollout, yaw_metrics, pitch_metrics,
                        yaw_model, pitch_model, entropy_coefficient,
                        total_stale_retries, superseded_frames,
                    )
                    if math.isfinite(selection_score) and selection_score > best_score:
                        best_score = selection_score
                        save_checkpoint(
                            args.best_checkpoint, yaw_model, pitch_model,
                            yaw_optimizer, pitch_optimizer, update_number, total_steps,
                            selection_score=best_score,
                        )
                        print(
                            f"New best aim checkpoint: avg_reward={best_score:+.4f} "
                            f"(update={update_number}).",
                            flush=True,
                        )
                    rollout.clear()
                    if update_number % args.checkpoint_every == 0:
                        save_checkpoint(
                            args.checkpoint, yaw_model, pitch_model,
                            yaw_optimizer, pitch_optimizer, update_number, total_steps,
                        )

        print("Stopping; saving the latest completed aim PPO update.", flush=True)
        save_checkpoint(
            args.checkpoint, yaw_model, pitch_model,
            yaw_optimizer, pitch_optimizer, update_number, total_steps,
        )
        return 0
    except transport.StopRequested:
        print("Stop requested; saving the latest completed aim PPO update.", flush=True)
        save_checkpoint(
            args.checkpoint, yaw_model, pitch_model,
            yaw_optimizer, pitch_optimizer, update_number, total_steps,
        )
        return 0
    except (OSError, ConnectionError, TimeoutError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"Aim training stopped: {error}", file=sys.stderr, flush=True)
        save_checkpoint(
            args.checkpoint, yaw_model, pitch_model,
            yaw_optimizer, pitch_optimizer, update_number, total_steps,
        )
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
        help=f"combined aim checkpoint to resume/save (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--best-checkpoint", type=Path, default=DEFAULT_BEST_CHECKPOINT,
        help=f"best combined aim checkpoint selected by rollout avg_reward (default: {DEFAULT_BEST_CHECKPOINT})",
    )
    parser.add_argument(
        "--init-yaw-from", type=Path, default=DEFAULT_YAW_CHECKPOINT,
        help=f"pretrained yaw-only checkpoint (default: {DEFAULT_YAW_CHECKPOINT})",
    )
    parser.add_argument(
        "--init-pitch-from", type=Path, default=DEFAULT_PITCH_CHECKPOINT,
        help=f"pretrained pitch-only checkpoint (default: {DEFAULT_PITCH_CHECKPOINT})",
    )
    parser.add_argument("--checkpoint-every", type=int, default=5,
                        help="save after this many completed 1024-step PPO updates")
    parser.add_argument("--seed", type=int, default=1, help="random seed for a new combined trainer")
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore an existing aim checkpoint and reload both pretrained axis policies")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.checkpoint.resolve() == args.best_checkpoint.resolve():
        parser.error("--checkpoint and --best-checkpoint must use different files")
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
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        print(f"Could not start aim training: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
