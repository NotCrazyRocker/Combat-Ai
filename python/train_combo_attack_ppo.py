#!/usr/bin/env python3
"""Continue combo PPO training with learned movement and attack controls."""

# ground truth attack

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
from torch.distributions import Bernoulli

import train_aim_ppo as aim
import train_yaw_ppo as transport
from combo_ppo_math import (
    ATTACK_MISS_PENALTY,
    MOVEMENT_ACTION_SIZE,
    MOVEMENT_OBSERVATION_SIZE,
    DAMAGE_DEALT_REWARD_PER_POINT,
    DAMAGE_TAKEN_PENALTY_PER_POINT,
    MAX_DAMAGE_REWARD_POINTS_PER_TICK,
    KNOCKBACK_HIT_REWARD,
    COMBO_REWARD_PER_EXTRA_HIT,
    MAX_COMBO_REWARD,
    encode_movement_observation,
    damage_transition_reward,
    movement_transition_reward,
)
from aim_geometry import head_aim_error, target_yaw_error_degrees
from yaw_ppo_math import (
    MAX_PITCH_DELTA_DEGREES,
    MAX_YAW_DELTA_DEGREES,
    PITCH_ONLY_OBSERVATION_SIZE,
    YAW_OBSERVATION_SIZE,
    encode_pitch_observation,
    encode_yaw_observation,
    pitch_transition_reward,
    scripted_sword_attack_eligible,
    yaw_transition_reward,
)

SCHEMA = "minecraftai.pvp.v1"
ROLLOUT_STEPS = aim.ROLLOUT_STEPS
COMBO_AIM_REWARD_SCALE = 0.5
TRAINER = "combo_attack_ppo_v1"
ATTACK_MISS_GRACE_TICKS = 2
MOVEMENT_ACTION_COUNT = 3  # forward, sprint, jump; attack is appended
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "combo_attack_latest.pt"
DEFAULT_BEST_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "combo_attack_best.pt"
DEFAULT_RESUME_CHECKPOINT = DEFAULT_BEST_CHECKPOINT
DEFAULT_BEST_AIM_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "aim_ppo_best.pt"
DEFAULT_LATEST_AIM_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "aim_ppo_latest.pt"


class MovementActorCritic(nn.Module):
    """Bernoulli actions for forward, sprint, jump, and learned attacks."""

    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(MOVEMENT_OBSERVATION_SIZE, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.actor_logits = nn.Linear(128, MOVEMENT_ACTION_COUNT + 1)
        self.critic = nn.Linear(128, 1)
        nn.init.zeros_(self.actor_logits.weight)
        nn.init.zeros_(self.actor_logits.bias)
        with torch.no_grad():
            self.actor_logits.bias[MOVEMENT_ACTION_SIZE] = -1.5  # jump starts conservative
            self.actor_logits.bias[MOVEMENT_ACTION_COUNT] = math.log(0.25 / 0.75)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def distribution_and_value(self, observations: torch.Tensor) -> tuple[Bernoulli, torch.Tensor]:
        hidden = self.shared(observations)
        return Bernoulli(logits=self.actor_logits(hidden)), self.critic(hidden).squeeze(-1)

    @torch.no_grad()
    def act(self, observation: tuple[float, ...], device: torch.device) -> tuple[np.ndarray, float, float]:
        state = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        distribution, value = self.distribution_and_value(state)
        actions = distribution.sample()
        log_probability = distribution.log_prob(actions).sum(dim=-1)
        return actions.squeeze(0).cpu().numpy().astype(np.bool_), float(log_probability.item()), float(value.item())


@dataclass
class ComboRollout:
    yaw_observations: list[tuple[float, ...]] = field(default_factory=list)
    pitch_observations: list[tuple[float, ...]] = field(default_factory=list)
    movement_observations: list[tuple[float, ...]] = field(default_factory=list)
    yaw_raw_actions: list[float] = field(default_factory=list)
    pitch_raw_actions: list[float] = field(default_factory=list)
    yaw_log_probabilities: list[float] = field(default_factory=list)
    pitch_log_probabilities: list[float] = field(default_factory=list)
    yaw_values: list[float] = field(default_factory=list)
    pitch_values: list[float] = field(default_factory=list)
    yaw_rewards: list[float] = field(default_factory=list)
    pitch_rewards: list[float] = field(default_factory=list)
    movement_actions: list[tuple[bool, bool, bool, bool]] = field(default_factory=list)
    movement_log_probabilities: list[float] = field(default_factory=list)
    movement_values: list[float] = field(default_factory=list)
    movement_rewards: list[float] = field(default_factory=list)
    knockback_hits: int = 0
    incoming_hits: int = 0

    def __len__(self) -> int:
        return len(self.yaw_rewards)

    def clear(self) -> None:
        for value in vars(self).values():
            if isinstance(value, list):
                value.clear()
        self.knockback_hits = 0
        self.incoming_hits = 0


def _validate_aim_checkpoint(checkpoint: dict[str, Any], path: Path) -> None:
    if (
        checkpoint.get("trainer") != aim.TRAINER
        or checkpoint.get("observation_schema") != SCHEMA
        or int(checkpoint.get("yaw_observation_size", -1)) != YAW_OBSERVATION_SIZE
        or int(checkpoint.get("pitch_observation_size", -1)) != PITCH_ONLY_OBSERVATION_SIZE
        or int(checkpoint.get("action_size", -1)) != aim.ACTION_SIZE
        or int(checkpoint.get("rollout_steps", -1)) != ROLLOUT_STEPS
        or not math.isclose(float(checkpoint.get("max_yaw_delta_degrees", -1)), MAX_YAW_DELTA_DEGREES)
        or not math.isclose(float(checkpoint.get("max_pitch_delta_degrees", -1)), MAX_PITCH_DELTA_DEGREES)
    ):
        raise RuntimeError(f"{path} is not a compatible combined aim PPO checkpoint")


def initialize_from_aim(path: Path, yaw_model: aim.AxisActorCritic, pitch_model: aim.AxisActorCritic,
                        device: torch.device) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"required aim checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    _validate_aim_checkpoint(checkpoint, path)
    try:
        yaw_model.load_state_dict(checkpoint["yaw_model"], strict=True)
        pitch_model.load_state_dict(checkpoint["pitch_model"], strict=True)
    except (KeyError, RuntimeError) as error:
        raise RuntimeError(f"aim checkpoint model layers do not match this trainer: {error}") from error


def save_checkpoint(path: Path, yaw_model: aim.AxisActorCritic, pitch_model: aim.AxisActorCritic,
                    movement_model: MovementActorCritic, optimizers: tuple[torch.optim.Optimizer, ...],
                    update: int, total_steps: int, performance_score: float | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    torch.save({
        "format": 1,
        "trainer": TRAINER,
        "observation_schema": SCHEMA,
        "yaw_observation_size": YAW_OBSERVATION_SIZE,
        "pitch_observation_size": PITCH_ONLY_OBSERVATION_SIZE,
        "movement_observation_size": MOVEMENT_OBSERVATION_SIZE,
        "action_size": MOVEMENT_ACTION_COUNT + 1,
        "max_yaw_delta_degrees": MAX_YAW_DELTA_DEGREES,
        "max_pitch_delta_degrees": MAX_PITCH_DELTA_DEGREES,
        "rollout_steps": ROLLOUT_STEPS,
        "yaw_model": yaw_model.state_dict(),
        "pitch_model": pitch_model.state_dict(),
        "movement_model": movement_model.state_dict(),
        "yaw_optimizer": optimizers[0].state_dict(),
        "pitch_optimizer": optimizers[1].state_dict(),
        "movement_optimizer": optimizers[2].state_dict(),
        "update_number": update,
        "total_steps": total_steps,
        "performance_score": performance_score,
        "torch_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng_state": np.random.get_state(),
    }, temporary)
    temporary.replace(path)
    print(f"Saved combo checkpoint: {path} (update={update}, steps={total_steps})", flush=True)


def load_checkpoint(path: Path, yaw_model: aim.AxisActorCritic, pitch_model: aim.AxisActorCritic,
                    movement_model: MovementActorCritic, optimizers: tuple[torch.optim.Optimizer, ...],
                    device: torch.device) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    saved_movement_model = checkpoint.get("movement_model", {})
    saved_actor_weight = saved_movement_model.get("actor_logits.weight")
    saved_actor_bias = saved_movement_model.get("actor_logits.bias")
    saved_action_count = tuple(saved_actor_weight.shape)[0] if isinstance(saved_actor_weight, torch.Tensor) else -1
    legacy_action_count = saved_action_count
    legacy_actor = legacy_action_count != MOVEMENT_ACTION_COUNT + 1
    action_shape_matches = (
        isinstance(saved_actor_weight, torch.Tensor)
        and tuple(saved_actor_weight.shape) in ((MOVEMENT_ACTION_COUNT + 1, 128), (5, 128), (MOVEMENT_ACTION_SIZE + 1, 128), (MOVEMENT_ACTION_SIZE, 128))
        and isinstance(saved_actor_bias, torch.Tensor)
        and tuple(saved_actor_bias.shape) in ((MOVEMENT_ACTION_COUNT + 1,), (5,), (MOVEMENT_ACTION_SIZE + 1,), (MOVEMENT_ACTION_SIZE,))
    )
    if (
        checkpoint.get("trainer") != TRAINER
        or checkpoint.get("observation_schema") != SCHEMA
        or int(checkpoint.get("yaw_observation_size", -1)) != YAW_OBSERVATION_SIZE
        or int(checkpoint.get("pitch_observation_size", -1)) != PITCH_ONLY_OBSERVATION_SIZE
        or int(checkpoint.get("movement_observation_size", -1)) != MOVEMENT_OBSERVATION_SIZE
        or int(checkpoint.get("action_size", -1)) not in (MOVEMENT_ACTION_COUNT + 1, 5, MOVEMENT_ACTION_SIZE + 1, MOVEMENT_ACTION_SIZE, 3)
        or not action_shape_matches
        or not math.isclose(float(checkpoint.get("max_yaw_delta_degrees", -1)), MAX_YAW_DELTA_DEGREES)
        or not math.isclose(float(checkpoint.get("max_pitch_delta_degrees", -1)), MAX_PITCH_DELTA_DEGREES)
        or int(checkpoint.get("rollout_steps", -1)) != ROLLOUT_STEPS
    ):
        raise RuntimeError(f"{path} is not a compatible {TRAINER} checkpoint")
    yaw_model.load_state_dict(checkpoint["yaw_model"], strict=True)
    pitch_model.load_state_dict(checkpoint["pitch_model"], strict=True)
    if legacy_actor:
        migrated = movement_model.state_dict()
        saved = checkpoint["movement_model"]
        for key in migrated:
            if key in ("actor_logits.weight", "actor_logits.bias"):
                if legacy_action_count == MOVEMENT_ACTION_SIZE + 1:
                    migrated[key][:MOVEMENT_ACTION_SIZE].copy_(saved[key][:MOVEMENT_ACTION_SIZE])
                    migrated[key][MOVEMENT_ACTION_COUNT].copy_(saved[key][MOVEMENT_ACTION_SIZE])
                else:
                    migrated[key][:MOVEMENT_ACTION_SIZE].copy_(saved[key][:MOVEMENT_ACTION_SIZE])
            else:
                migrated[key].copy_(saved[key])
        # Old layout: forward, sprint, left, right, attack. Preserve W/sprint
        # and attack; initialize the new jump head conservatively.
        if legacy_action_count == 5:
            migrated["actor_logits.weight"][2].zero_()
            migrated["actor_logits.bias"][2] = -1.5
            migrated["actor_logits.weight"][3].copy_(saved["actor_logits.weight"][4])
            migrated["actor_logits.bias"][3].copy_(saved["actor_logits.bias"][4])
        elif legacy_action_count == 3:
            migrated["actor_logits.weight"][:2].copy_(saved["actor_logits.weight"][:2])
            migrated["actor_logits.bias"][:2].copy_(saved["actor_logits.bias"][:2])
            migrated["actor_logits.weight"][2].zero_()
            migrated["actor_logits.bias"][2] = -1.5
            migrated["actor_logits.weight"][3].copy_(saved["actor_logits.weight"][2])
            migrated["actor_logits.bias"][3].copy_(saved["actor_logits.bias"][2])
        else:
            migrated["actor_logits.weight"][:2].copy_(saved["actor_logits.weight"][:2])
            migrated["actor_logits.bias"][:2].copy_(saved["actor_logits.bias"][:2])
            migrated["actor_logits.weight"][2].zero_()
            migrated["actor_logits.bias"][2] = -1.5
            migrated["actor_logits.weight"][3].zero_()
            migrated["actor_logits.bias"][3] = math.log(0.25 / 0.75)
        movement_model.load_state_dict(migrated, strict=True)
    else:
        movement_model.load_state_dict(checkpoint["movement_model"], strict=True)
    for name, optimizer in zip(("yaw_optimizer", "pitch_optimizer", "movement_optimizer"), optimizers):
        if not (legacy_actor and name == "movement_optimizer"):
            try:
                optimizer.load_state_dict(checkpoint[name])
            except (KeyError, ValueError):
                if name != "movement_optimizer":
                    raise
    if legacy_actor:
        print("Migrated legacy movement actions; jump starts conservatively and learned attack weights are retained.", flush=True)
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if checkpoint.get("torch_cuda_rng_state") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint["torch_cuda_rng_state"])
    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])
    return int(checkpoint.get("update_number", 0)), int(checkpoint.get("total_steps", 0))


def load_source_combo_checkpoint(
    path: Path,
    yaw_model: aim.AxisActorCritic,
    pitch_model: aim.AxisActorCritic,
    movement_model: MovementActorCritic,
    optimizers: tuple[torch.optim.Optimizer, ...],
    device: torch.device,
) -> tuple[int, int]:
    """Load the existing combo checkpoint and initialize new A/D/attack outputs."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    source_actor = checkpoint.get("movement_model", {}).get("actor_logits.weight")
    source_bias = checkpoint.get("movement_model", {}).get("actor_logits.bias")
    if (
        checkpoint.get("trainer") != "combo_ppo_v1"
        or checkpoint.get("observation_schema") != SCHEMA
        or int(checkpoint.get("movement_observation_size", -1)) != MOVEMENT_OBSERVATION_SIZE
        or int(checkpoint.get("action_size", -1)) not in (MOVEMENT_ACTION_SIZE, 3)
        or not isinstance(source_actor, torch.Tensor)
        or tuple(source_actor.shape) != (MOVEMENT_ACTION_SIZE, 128)
        or not isinstance(source_bias, torch.Tensor)
        or tuple(source_bias.shape) != (MOVEMENT_ACTION_SIZE,)
        or int(checkpoint.get("yaw_observation_size", -1)) != YAW_OBSERVATION_SIZE
        or int(checkpoint.get("pitch_observation_size", -1)) != PITCH_ONLY_OBSERVATION_SIZE
        or int(checkpoint.get("rollout_steps", -1)) != ROLLOUT_STEPS
    ):
        raise RuntimeError(f"{path} is not a source combo PPO checkpoint")
    yaw_model.load_state_dict(checkpoint["yaw_model"], strict=True)
    pitch_model.load_state_dict(checkpoint["pitch_model"], strict=True)
    source_state = checkpoint["movement_model"]
    target_state = movement_model.state_dict()
    for key, target_value in target_state.items():
        source_value = source_state.get(key)
        if source_value is None:
            raise RuntimeError(f"{path} is missing movement parameter {key}")
        if key == "actor_logits.weight":
            if tuple(source_value.shape) != (MOVEMENT_ACTION_SIZE, 128):
                raise RuntimeError(f"{path} has an unsupported movement actor shape {tuple(source_value.shape)}")
            target_value[:MOVEMENT_ACTION_SIZE].copy_(source_value)
        elif key == "actor_logits.bias":
            if tuple(source_value.shape) != (MOVEMENT_ACTION_SIZE,):
                raise RuntimeError(f"{path} has an unsupported movement actor shape {tuple(source_value.shape)}")
            target_value[:MOVEMENT_ACTION_SIZE].copy_(source_value)
        else:
            if target_value.shape != source_value.shape:
                raise RuntimeError(f"{path} has an incompatible movement parameter {key}")
            target_value.copy_(source_value)
    target_state["actor_logits.weight"][2:].zero_()
    target_state["actor_logits.bias"][2] = -1.5
    target_state["actor_logits.bias"][3] = math.log(0.25 / 0.75)
    movement_model.load_state_dict(target_state, strict=True)
    if len(checkpoint.get("yaw_optimizer", {})):
        optimizers[0].load_state_dict(checkpoint["yaw_optimizer"])
    if len(checkpoint.get("pitch_optimizer", {})):
        optimizers[1].load_state_dict(checkpoint["pitch_optimizer"])
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if checkpoint.get("torch_cuda_rng_state") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint["torch_cuda_rng_state"])
    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])
    return int(checkpoint.get("update_number", 0)), int(checkpoint.get("total_steps", 0))


def ppo_update_movement(model: MovementActorCritic, optimizer: torch.optim.Optimizer, rollout: ComboRollout,
                        bootstrap_value: float, device: torch.device,
                        entropy_coefficient: float) -> dict[str, float]:
    observations = np.asarray(rollout.movement_observations, dtype=np.float32)
    actions = np.asarray(rollout.movement_actions, dtype=np.float32)
    old_logp_array = np.asarray(rollout.movement_log_probabilities, dtype=np.float32)
    rewards = np.asarray(rollout.movement_rewards, dtype=np.float32)
    values = np.asarray(rollout.movement_values, dtype=np.float32)
    if (observations.shape != (ROLLOUT_STEPS, MOVEMENT_OBSERVATION_SIZE)
            or actions.shape != (ROLLOUT_STEPS, MOVEMENT_ACTION_COUNT + 1)
            or old_logp_array.shape != (ROLLOUT_STEPS,) or rewards.shape != (ROLLOUT_STEPS,)
            or values.shape != (ROLLOUT_STEPS,) or not math.isfinite(bootstrap_value)
            or not all(np.isfinite(array).all() for array in (observations, actions, old_logp_array, rewards, values))):
        raise ValueError(
            "combo PPO rollout contains invalid or non-finite movement data "
            f"(observations={observations.shape}, actions={actions.shape}, "
            f"logp={old_logp_array.shape}, rewards={rewards.shape}, values={values.shape})"
        )
    advantages = np.zeros_like(rewards)
    gae = 0.0
    for index in reversed(range(ROLLOUT_STEPS)):
        next_value = bootstrap_value if index == ROLLOUT_STEPS - 1 else values[index + 1]
        delta = rewards[index] + 0.99 * next_value - values[index]
        gae = delta + 0.99 * 0.95 * gae
        advantages[index] = gae
    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    obs = torch.as_tensor(observations, dtype=torch.float32, device=device)
    action_tensor = torch.as_tensor(actions, dtype=torch.float32, device=device)
    old_logp = torch.as_tensor(old_logp_array, dtype=torch.float32, device=device)
    advantage_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=device)
    return_tensor = torch.as_tensor(returns, dtype=torch.float32, device=device)
    metrics: dict[str, list[float]] = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": []}
    for _ in range(4):
        permutation = torch.randperm(ROLLOUT_STEPS, device=device)
        for start in range(0, ROLLOUT_STEPS, 256):
            indices = permutation[start:start + 256]
            distribution, value_prediction = model.distribution_and_value(obs[indices])
            new_logp = distribution.log_prob(action_tensor[indices]).sum(dim=-1)
            log_ratio = new_logp - old_logp[indices]
            ratio = log_ratio.exp()
            policy_loss = -torch.minimum(ratio * advantage_tensor[indices],
                torch.clamp(ratio, 0.8, 1.2) * advantage_tensor[indices]).mean()
            value_loss = 0.5 * (value_prediction - return_tensor[indices]).square().mean()
            entropy = distribution.entropy().sum(dim=-1).mean()
            loss = policy_loss + 0.5 * value_loss - entropy_coefficient * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
            for key, value in (("policy_loss", policy_loss), ("value_loss", value_loss),
                               ("entropy", entropy), ("approx_kl", approx_kl)):
                metrics[key].append(float(value.item()))
    return {name: float(np.mean(items)) for name, items in metrics.items()}


def train(args: argparse.Namespace) -> int:
    transport.STOP_REQUESTED = False
    signal.signal(signal.SIGINT, transport.request_stop)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yaw_model = aim.AxisActorCritic().to(device)
    pitch_model = aim.AxisActorCritic().to(device)
    movement_model = MovementActorCritic().to(device)
    yaw_optimizer = torch.optim.Adam(yaw_model.parameters(), lr=3e-4, eps=1e-5)
    pitch_optimizer = torch.optim.Adam(pitch_model.parameters(), lr=3e-4, eps=1e-5)
    movement_optimizer = torch.optim.Adam(movement_model.parameters(), lr=3e-4, eps=1e-5)
    optimizers = (yaw_optimizer, pitch_optimizer, movement_optimizer)
    update = total_steps = 0
    best_score = -math.inf
    # Continue from latest when available; otherwise seed the run from best.
    if not args.no_resume and args.checkpoint.exists():
        update, total_steps = load_checkpoint(
            args.checkpoint, yaw_model, pitch_model, movement_model, optimizers, device
        )
        print(f"Continued from latest checkpoint: {args.checkpoint} (update={update}, step={total_steps}).", flush=True)
    elif not args.no_resume and args.best_checkpoint.exists():
        update, total_steps = load_checkpoint(
            args.best_checkpoint, yaw_model, pitch_model, movement_model, optimizers, device
        )
        print(f"Initialized from best checkpoint: {args.best_checkpoint} (update={update}, step={total_steps}).", flush=True)
    elif not args.no_resume and args.resume_from.exists():
        update, total_steps = load_source_combo_checkpoint(
            args.resume_from, yaw_model, pitch_model, movement_model, optimizers, device
        )
        print(f"Initialized from original combo checkpoint: {args.resume_from} (update={update}, step={total_steps}).", flush=True)
    else:
        init_path = args.init_from
        if init_path is None:
            init_path = (
                DEFAULT_BEST_AIM_CHECKPOINT
                if DEFAULT_BEST_AIM_CHECKPOINT.is_file()
                else DEFAULT_LATEST_AIM_CHECKPOINT
            )
        initialize_from_aim(init_path, yaw_model, pitch_model, device)
        print(f"Initialized yaw and pitch from {init_path}; forward and sprint start with fresh policies.", flush=True)
    if args.best_checkpoint.exists():
        best_checkpoint = torch.load(args.best_checkpoint, map_location="cpu", weights_only=False)
        best_score = float(best_checkpoint.get("performance_score", -math.inf))
    print(f"Using {device}; combo PPO uses {ROLLOUT_STEPS} applied steps per update.", flush=True)
    print(
        "Damage shaping: "
        f"+{DAMAGE_DEALT_REWARD_PER_POINT:.2f} per health point dealt, "
        f"-{DAMAGE_TAKEN_PENALTY_PER_POINT:.2f} per health point taken "
        f"(capped at {MAX_DAMAGE_REWARD_POINTS_PER_TICK:.0f} points per tick).",
        flush=True,
    )
    print("Critical hits earn the knockback-hit reward plus a capped streak bonus; incoming hits do not reset the critical streak.", flush=True)
    print("Policy learns yaw, pitch, forward (W), sprint, jump (Space), and attack; A/D strafe is disabled. "
          f"Yaw and pitch aiming rewards are scaled to {COMBO_AIM_REWARD_SCALE:.2f}; "
          "pitch also receives combat rewards, while incoming hits are penalized.", flush=True)

    connection: transport.MinecraftConnection | None = None
    rollout = ComboRollout()
    last_tick = -1
    previous_yaw_error: float | None = None
    previous_pitch_error: float | None = None
    previous_yaw_action = prior_yaw_action = 0.0
    previous_pitch_action = prior_pitch_action = 0.0
    previous_forward = previous_sprint = False
    forward_streak = sprint_streak = 0
    attack_context: list[tuple[bool, bool]] = []
    knockback_combo = 0
    critical_combo = 0
    incoming_combo = 0
    pending_eligible_attacks: list[int] = []
    stale_retries = superseded_frames = 0
    try:
        connection = transport.MinecraftConnection(args.host, args.port)
        while not transport.STOP_REQUESTED:
            if not connection.enabled:
                save_checkpoint(args.checkpoint, yaw_model, pitch_model, movement_model, optimizers, update, total_steps)
                rollout.clear()
                previous_yaw_error = None
                previous_pitch_error = None
                previous_yaw_action = prior_yaw_action = 0.0
                previous_pitch_action = prior_pitch_action = 0.0
                previous_forward = previous_sprint = False
                forward_streak = sprint_streak = 0
                attack_context.clear()
                knockback_combo = 0
                critical_combo = 0
                incoming_combo = 0
                pending_eligible_attacks.clear()
                print("Training paused. Press K in Minecraft to resume.", flush=True)
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
                yaw_observation = encode_yaw_observation(frame, previous_yaw_error, previous_yaw_action, prior_yaw_action)
                pitch_observation = encode_pitch_observation(
                    frame, previous_pitch_error, previous_pitch_action, prior_pitch_action
                )
                movement_observation = encode_movement_observation(
                    frame, previous_forward, forward_streak, previous_sprint, sprint_streak, knockback_combo
                )
                yaw_raw, yaw_logp, yaw_value = yaw_model.act(yaw_observation, device)
                pitch_raw, pitch_logp, pitch_value = pitch_model.act(pitch_observation, device)
                move_action, move_logp, move_value = movement_model.act(movement_observation, device)
                yaw_action = float(np.tanh(yaw_raw))
                pitch_action = float(np.tanh(pitch_raw))
                forward, sprint, jump, attack = (bool(move_action[i]) for i in range(4))
                eligible = scripted_sword_attack_eligible(frame)
                fresher_frame = connection.take_newer_observation(after_tick=tick)
                if not connection.enabled:
                    break
                if fresher_frame is not None:
                    frame = fresher_frame
                    superseded_frames += 1
                    continue
                control_result = connection.send_aim(
                    tick, yaw_action * MAX_YAW_DELTA_DEGREES,
                    pitch_action * MAX_PITCH_DELTA_DEGREES,
                    attack=attack, forward=forward, sprint=sprint, jump=jump,
                )
                if control_result == "stale":
                    stale_retries += 1
                    frame = connection.wait_for_observation(after_tick=tick)
                    previous_yaw_error = None
                    previous_pitch_error = None
                    if frame is None:
                        break
                    if stale_retries % 100 == 0:
                        print(f"stale retries={stale_retries}; recomputing from fresh observations", flush=True)
                    continue
                if control_result == "disabled":
                    break
                next_frame = connection.wait_for_applied_observation(tick, int(control_result))
                if next_frame is None:
                    break

                yaw_reward = COMBO_AIM_REWARD_SCALE * yaw_transition_reward(
                    frame, next_frame, yaw_action, previous_yaw_action
                )
                movement_result = movement_transition_reward(
                    frame, next_frame, forward, sprint, attack, eligible,
                    previous_forward, previous_sprint,
                    tuple(attack_context), knockback_combo, forward_streak, sprint_streak, incoming_combo,
                )
                (movement_reward, knockback_hit, incoming_hit, _attack_offset, _hit_reward,
                 _attack_success, _attack_miss, _false_attack, _missed_eligible,
                 _premature_attack, knockback_combo, incoming_combo) = movement_result
                movement_reward += damage_transition_reward(next_frame)
                critical_hit = _attack_success and bool(next_frame.get("features", {}).get("last_hit_critical", False))
                if _attack_success:
                    if critical_hit:
                        critical_combo += 1
                        movement_reward += KNOCKBACK_HIT_REWARD + min(
                            MAX_COMBO_REWARD,
                            COMBO_REWARD_PER_EXTRA_HIT * (critical_combo - 1),
                        )
                    else:
                        critical_combo = 0
                if attack and eligible:
                    pending_eligible_attacks.append(tick)
                if _attack_offset is not None:
                    source_tick = tick - _attack_offset
                    if source_tick in pending_eligible_attacks:
                        pending_eligible_attacks.remove(source_tick)
                expired_attacks = [
                    attack_tick for attack_tick in pending_eligible_attacks
                    if tick - attack_tick >= ATTACK_MISS_GRACE_TICKS
                ]
                if expired_attacks:
                    movement_reward -= ATTACK_MISS_PENALTY * len(expired_attacks)
                    for attack_tick in expired_attacks:
                        pending_eligible_attacks.remove(attack_tick)
                pitch_aim_reward = COMBO_AIM_REWARD_SCALE * pitch_transition_reward(
                    frame, next_frame, pitch_action, previous_pitch_action
                )
                pitch_reward = movement_reward + pitch_aim_reward
                rollout.yaw_observations.append(yaw_observation)
                rollout.pitch_observations.append(pitch_observation)
                rollout.movement_observations.append(movement_observation)
                rollout.yaw_raw_actions.append(yaw_raw)
                rollout.pitch_raw_actions.append(pitch_raw)
                rollout.yaw_log_probabilities.append(yaw_logp)
                rollout.pitch_log_probabilities.append(pitch_logp)
                rollout.yaw_values.append(yaw_value)
                rollout.pitch_values.append(pitch_value)
                rollout.yaw_rewards.append(yaw_reward)
                rollout.pitch_rewards.append(pitch_reward)
                rollout.movement_actions.append((forward, sprint, jump, attack))
                rollout.movement_log_probabilities.append(move_logp)
                rollout.movement_values.append(move_value)
                rollout.movement_rewards.append(movement_reward)
                rollout.knockback_hits += int(knockback_hit)
                rollout.incoming_hits += int(incoming_hit)
                total_steps += 1
                attack_context.append((attack, eligible))
                if len(attack_context) > 3:
                    attack_context.pop(0)
                if forward == previous_forward:
                    forward_streak += 1
                else:
                    forward_streak = 1
                if sprint == previous_sprint:
                    sprint_streak += 1
                else:
                    sprint_streak = 1
                previous_forward, previous_sprint = forward, sprint
                prior_yaw_action, previous_yaw_action = previous_yaw_action, yaw_action
                prior_pitch_action, previous_pitch_action = previous_pitch_action, pitch_action
                previous_yaw_error = target_yaw_error_degrees(frame)
                previous_pitch_error = head_aim_error(frame)
                frame = next_frame

                if len(rollout) == ROLLOUT_STEPS:
                    next_yaw = encode_yaw_observation(frame, previous_yaw_error, previous_yaw_action, prior_yaw_action)
                    next_pitch = encode_pitch_observation(
                        frame, previous_pitch_error, previous_pitch_action, prior_pitch_action
                    )
                    next_movement = encode_movement_observation(
                        frame, previous_forward, forward_streak, previous_sprint, sprint_streak, knockback_combo
                    )
                    with torch.no_grad():
                        def value_of(model: nn.Module, observation: tuple[float, ...]) -> float:
                            return float(model.distribution_and_value(torch.as_tensor(
                                observation, dtype=torch.float32, device=device).unsqueeze(0))[1].item())
                        yaw_bootstrap = value_of(yaw_model, next_yaw)
                        pitch_bootstrap = value_of(pitch_model, next_pitch)
                        movement_bootstrap = value_of(movement_model, next_movement)
                    next_update = update + 1
                    entropy_coef = max(0.001, 0.01 * math.exp(-next_update / 120.0))
                    yaw_metrics = aim.ppo_update_axis(
                        yaw_model, yaw_optimizer, rollout.yaw_observations, rollout.yaw_raw_actions,
                        rollout.yaw_log_probabilities, rollout.yaw_rewards, rollout.yaw_values,
                        yaw_bootstrap, device, entropy_coef,
                    )
                    pitch_metrics = aim.ppo_update_axis(
                        pitch_model, pitch_optimizer, rollout.pitch_observations, rollout.pitch_raw_actions,
                        rollout.pitch_log_probabilities, rollout.pitch_rewards, rollout.pitch_values,
                        pitch_bootstrap, device, entropy_coef,
                    )
                    movement_metrics = ppo_update_movement(
                        movement_model, movement_optimizer, rollout, movement_bootstrap, device, entropy_coef
                    )
                    update = next_update
                    performance_score = float(np.mean(rollout.movement_rewards))
                    print(
                        f"update={update} steps={total_steps} avg_yaw_reward={np.mean(rollout.yaw_rewards):+.4f} "
                        f"avg_pitch_reward={np.mean(rollout.pitch_rewards):+.4f} "
                        f"avg_combat_reward={np.mean(rollout.movement_rewards):+.4f} "
                        f"knockback_hits={rollout.knockback_hits} incoming_hits={rollout.incoming_hits} "
                        f"forward={100*np.mean([a[0] for a in rollout.movement_actions]):.1f}% "
                        f"sprint={100*np.mean([a[1] for a in rollout.movement_actions]):.1f}% "
                        f"jump={100*np.mean([a[2] for a in rollout.movement_actions]):.1f}% "
                        f"attack={100*np.mean([a[3] for a in rollout.movement_actions]):.1f}% "
                        f"yaw_policy_loss={yaw_metrics['policy_loss']:+.4f} "
                        f"pitch_policy_loss={pitch_metrics['policy_loss']:+.4f} "
                        f"movement_policy_loss={movement_metrics['policy_loss']:+.4f} "
                        f"stale_retries={stale_retries} superseded_frames={superseded_frames}", flush=True,
                    )
                    rollout.clear()
                    if performance_score > best_score:
                        best_score = performance_score
                        save_checkpoint(args.best_checkpoint, yaw_model, pitch_model, movement_model,
                                        optimizers, update, total_steps, performance_score)
                        print(f"New best combo model: avg_combat_reward={best_score:+.4f}", flush=True)
                    if update % args.checkpoint_every == 0:
                        save_checkpoint(args.checkpoint, yaw_model, pitch_model, movement_model,
                                        optimizers, update, total_steps)

        print("Stopping; saving the latest completed combo PPO update.", flush=True)
        save_checkpoint(args.checkpoint, yaw_model, pitch_model, movement_model, optimizers, update, total_steps)
        return 0
    except transport.StopRequested:
        print("Stop requested; saving the latest completed combo PPO update.", flush=True)
        save_checkpoint(args.checkpoint, yaw_model, pitch_model, movement_model, optimizers, update, total_steps)
        return 0
    except (OSError, ConnectionError, TimeoutError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"Combo training stopped: {error}", file=sys.stderr, flush=True)
        save_checkpoint(args.checkpoint, yaw_model, pitch_model, movement_model, optimizers, update, total_steps)
        return 1
    finally:
        if connection is not None:
            connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="MinecraftAI loopback address")
    parser.add_argument("--port", type=int, default=transport.DEFAULT_PORT, help="MinecraftAI control port")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help=f"combo checkpoint to resume/save (default: {DEFAULT_CHECKPOINT})")
    parser.add_argument("--best-checkpoint", type=Path, default=DEFAULT_BEST_CHECKPOINT,
                        help=f"best-performing combo checkpoint (default: {DEFAULT_BEST_CHECKPOINT})")
    parser.add_argument("--resume-from", type=Path, default=DEFAULT_RESUME_CHECKPOINT,
                        help=f"source combo checkpoint used for the initial migration (default: {DEFAULT_RESUME_CHECKPOINT})")
    parser.add_argument("--init-from", type=Path, default=None,
                        help="combined aim checkpoint to initialize from (default: aim_ppo_best.pt, falling back to aim_ppo_latest.pt)")
    parser.add_argument("--checkpoint-every", type=int, default=5,
                        help="save after this many completed 1024-step PPO updates")
    parser.add_argument("--seed", type=int, default=1, help="random seed for a new combo trainer")
    parser.add_argument("--no-resume", action="store_true", help="ignore an existing combo checkpoint")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be at least 1")
    try:
        return train(args)
    except ImportError as error:
        print("Training dependencies are missing. Install them with: "
              "py -3.12 -m pip install -r .\\python\\requirements.txt", file=sys.stderr)
        print(str(error), file=sys.stderr)
        return 1
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        print(f"Could not start combo training: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
