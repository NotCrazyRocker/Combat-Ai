#!/usr/bin/env python3
"""Train a joint horizontal and vertical aiming policy online with PPO."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import select
import signal
import socket
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from aim_geometry import head_aim_error, target_yaw_error_degrees
from yaw_ppo_math import (
    MAX_YAW_DELTA_DEGREES,
    MAX_PITCH_DELTA_DEGREES,
    JOINT_OBSERVATION_SIZE,
    YAW_OBSERVATION_SIZE,
    encode_yaw_pitch_observation,
    scripted_attack_decision,
    yaw_pitch_transition_rewards,
)

PROTOCOL = 2
SCHEMA = "minecraftai.pvp.v1"
WIRE_SCHEMA = "minecraftai.pvp.v2"
ROLLOUT_STEPS = 1024
ACTION_SIZE = 2
DEFAULT_PORT = 8765
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "joint_yaw_pitch_legacy_latest.pt"
DEFAULT_YAW_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "yaw_ppo_latest.pt"
JOINT_TRAINER = "joint_yaw_pitch_ppo_v2"
LEGACY_JOINT_TRAINER = "joint_yaw_pitch_ppo_v1"

STOP_REQUESTED = False


class StopRequested(Exception):
    """Raised when Ctrl+C asks the loop to stop at a safe point."""


def request_stop(_signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\nCtrl+C received; saving the latest completed PPO update...", flush=True)


class MinecraftConnection:
    """Line-delimited JSON client for the mod's full-duplex TCP stream."""

    def __init__(self, host: str, port: int):
        self.socket = socket.create_connection((host, port), timeout=5.0)
        self.socket.setblocking(False)
        self.incoming = bytearray()
        self.enabled = False
        self.latest_observation: dict[str, Any] | None = None
        self.pending_acks: dict[int, dict[str, Any]] = {}
        self.last_stale_detail = ""
        self.sequence = 1

        ready = self.read_message(timeout=10.0)
        if ready.get("type") != "ready" or ready.get("protocol") != PROTOCOL:
            raise RuntimeError(f"unsupported MinecraftAI protocol response: {ready}")
        if ready.get("observation_schema") != WIRE_SCHEMA:
            raise RuntimeError(f"unsupported observation schema: {ready.get('observation_schema')!r}")
        self.enabled = bool(ready.get("ai_control_enabled", False))
        print(f"Connected to MinecraftAI (AI control {'ON' if self.enabled else 'OFF'}).", flush=True)

    def close(self) -> None:
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.socket.close()

    def consume(self, message: dict[str, Any]) -> None:
        kind = message.get("type")
        if kind == "observation" and message.get("schema") == WIRE_SCHEMA:
            # Keep the legacy policy input contract while accepting the expanded v2 frame.
            message = dict(message)
            message["schema"] = SCHEMA
            features = dict(message.get("features", {}))
            features.setdefault("remote_control_diagnostics", message.get("diagnostics", {}))
            message["features"] = features
            tick = int(message.get("tick", -1))
            if self.latest_observation is None or tick > int(self.latest_observation.get("tick", -1)):
                self.latest_observation = message
        elif kind == "control_state":
            enabled = bool(message.get("ai_control_enabled", False))
            if enabled != self.enabled:
                self.enabled = enabled
                print(f"AI control switched {'ON' if enabled else 'OFF'}.", flush=True)
        elif kind == "ack":
            try:
                self.pending_acks[int(message.get("seq", -1))] = message
            except (TypeError, ValueError):
                pass

    def read_message(self, timeout: float | None = None) -> dict[str, Any]:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if STOP_REQUESTED:
                raise StopRequested
            newline = self.incoming.find(b"\n")
            if newline >= 0:
                line = bytes(self.incoming[:newline]).rstrip(b"\r")
                del self.incoming[:newline + 1]
                if not line:
                    continue
                message = json.loads(line.decode("utf-8"))
                if not isinstance(message, dict):
                    raise ValueError("MinecraftAI protocol message must be a JSON object")
                self.consume(message)
                return message

            wait = 0.25
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError("timed out waiting for MinecraftAI")
                wait = min(wait, remaining)
            readable, _, _ = select.select([self.socket], [], [], wait)
            if not readable:
                continue
            chunk = self.socket.recv(65536)
            if not chunk:
                raise ConnectionError("MinecraftAI closed the control connection")
            self.incoming.extend(chunk)
            if len(self.incoming) > 2_000_000:
                raise RuntimeError("MinecraftAI response buffer exceeded 2 MB")

    def drain_available(self) -> None:
        while not STOP_REQUESTED:
            newline = self.incoming.find(b"\n")
            if newline >= 0:
                line = bytes(self.incoming[:newline]).rstrip(b"\r")
                del self.incoming[:newline + 1]
                if line:
                    message = json.loads(line.decode("utf-8"))
                    if not isinstance(message, dict):
                        raise ValueError("MinecraftAI protocol message must be a JSON object")
                    self.consume(message)
                continue
            readable, _, _ = select.select([self.socket], [], [], 0.0)
            if not readable:
                return
            chunk = self.socket.recv(65536)
            if not chunk:
                raise ConnectionError("MinecraftAI closed the control connection")
            self.incoming.extend(chunk)

    def wait_for_observation(self, after_tick: int) -> dict[str, Any] | None:
        while True:
            if not self.enabled:
                return None
            observation = self.latest_observation
            if observation is not None and int(observation.get("tick", -1)) > after_tick:
                self.drain_available()
                if not self.enabled:
                    return None
                observation = self.latest_observation
                if observation is not None and int(observation.get("tick", -1)) > after_tick:
                    self.latest_observation = None
                    return observation
            self.read_message()

    def wait_until_enabled(self) -> None:
        while not self.enabled:
            self.read_message()

    def take_newer_observation(self, after_tick: int) -> dict[str, Any] | None:
        """Refresh the socket buffer just before sending an action.

        If inference took long enough that a newer frame arrived, return it so
        the policy can recompute instead of knowingly sending an old action.
        """
        self.drain_available()
        if not self.enabled:
            return None
        observation = self.latest_observation
        if observation is not None and int(observation.get("tick", -1)) > after_tick:
            self.latest_observation = None
            return observation
        return None

    def wait_for_applied_observation(
        self, after_tick: int, sequence: int, timeout: float = 5.0
    ) -> dict[str, Any] | None:
        deadline = time.monotonic() + timeout
        while True:
            if not self.enabled:
                return None
            observation = self.latest_observation
            if observation is not None and int(observation.get("tick", -1)) > after_tick:
                self.drain_available()
                if not self.enabled:
                    return None
                observation = self.latest_observation
                if observation is not None and int(observation.get("tick", -1)) > after_tick:
                    features = observation.get("features", {})
                    diagnostics = features.get("remote_control_diagnostics", {})
                    applied = int(diagnostics.get("last_applied_sequence", -1))
                    if applied >= sequence:
                        self.latest_observation = None
                        return observation
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError(f"control sequence {sequence} was accepted but not observed as applied")
            self.read_message(timeout=min(0.25, remaining))

    def send_aim(
        self,
        observation_tick: int,
        yaw_delta: float,
        pitch_delta: float,
        attack: bool = False,
        forward: bool = False,
        sprint: bool = False,
        left: bool = False,
        right: bool = False,
        backward: bool = False,
        jump: bool = False,
    ) -> int | str:
        sequence = self.sequence
        self.sequence += 1
        # Each command carries the view deltas and per-tick button states.
        command = {
            "type": "control",
            "seq": sequence,
            "tick": int(observation_tick),
            "yaw_delta": float(yaw_delta),
            "pitch_delta": float(pitch_delta),
            "forward": bool(forward),
            "backward": bool(backward),
            "left": bool(left),
            "right": bool(right),
            "jump": bool(jump),
            "sneak": False,
            "sprint": bool(sprint),
            "attack": bool(attack),
            "use": False,
        }
        if not math.isfinite(yaw_delta) or not math.isfinite(pitch_delta):
            raise ValueError("aim actions must be finite")
        payload = json.dumps(command, separators=(",", ":"), allow_nan=False) + "\n"
        self.socket.sendall(payload.encode("utf-8"))
        while sequence not in self.pending_acks:
            self.read_message(timeout=5.0)
        ack = self.pending_acks.pop(sequence)
        if "ai_control_enabled" in ack and not bool(ack["ai_control_enabled"]):
            self.enabled = False
        if not ack.get("accepted"):
            if not self.enabled:
                return "disabled"
            error = str(ack.get("error") or "")
            if error.startswith("observation tick"):
                latest_tick = ack.get("latest_observation_tick", "unknown")
                self.last_stale_detail = f"sent_tick={observation_tick} latest_tick={latest_tick} reason={error}"
                return "stale"
            raise RuntimeError(f"MinecraftAI rejected control tick {observation_tick}: {ack.get('error')}")
        return sequence if self.enabled else "disabled"


class YawActorCritic(nn.Module):
    """Gaussian actor with separate yaw/pitch values and neutral pitch output."""

    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(JOINT_OBSERVATION_SIZE, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(128, ACTION_SIZE)
        self.critic = nn.Linear(128, ACTION_SIZE)
        self.log_std = nn.Parameter(torch.tensor([math.log(0.06), math.log(0.08)], dtype=torch.float32))
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
    def squashed_log_probabilities(distribution: Normal, raw_actions: torch.Tensor) -> torch.Tensor:
        squashed = torch.tanh(raw_actions)
        correction = torch.log(1.0 - squashed.pow(2) + 1e-6)
        return distribution.log_prob(raw_actions) - correction

    @torch.no_grad()
    def act(
        self, observation: tuple[float, ...], device: torch.device
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        state = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        distribution, value = self.distribution_and_value(state)
        raw = distribution.sample()
        log_probabilities = self.squashed_log_probabilities(distribution, raw)
        return (
            raw.squeeze(0).cpu().numpy(),
            log_probabilities.squeeze(0).cpu().numpy(),
            value.squeeze(0).cpu().numpy(),
        )


@dataclass
class Rollout:
    observations: list[tuple[float, ...]] = field(default_factory=list)
    raw_actions: list[np.ndarray] = field(default_factory=list)
    log_probabilities: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    yaw_rewards: list[float] = field(default_factory=list)
    pitch_rewards: list[float] = field(default_factory=list)
    values: list[np.ndarray] = field(default_factory=list)
    yaw_errors: list[float] = field(default_factory=list)
    pitch_errors: list[float] = field(default_factory=list)
    signed_pitch_errors: list[float] = field(default_factory=list)
    target_valid: list[bool] = field(default_factory=list)
    projections_valid: list[bool] = field(default_factory=list)
    head_projections_valid: list[bool] = field(default_factory=list)
    on_screen: list[bool] = field(default_factory=list)
    yaw_actions: list[float] = field(default_factory=list)
    pitch_actions: list[float] = field(default_factory=list)
    correction_candidates: list[bool] = field(default_factory=list)
    corrective_actions: list[bool] = field(default_factory=list)
    pitch_correction_candidates: list[bool] = field(default_factory=list)
    pitch_corrective_actions: list[bool] = field(default_factory=list)
    crosshair_hits: list[bool] = field(default_factory=list)
    scripted_attacks: int = 0

    def __len__(self) -> int:
        return len(self.rewards)

    def clear(self) -> None:
        self.observations.clear()
        self.raw_actions.clear()
        self.log_probabilities.clear()
        self.rewards.clear()
        self.yaw_rewards.clear()
        self.pitch_rewards.clear()
        self.values.clear()
        self.yaw_errors.clear()
        self.pitch_errors.clear()
        self.signed_pitch_errors.clear()
        self.target_valid.clear()
        self.projections_valid.clear()
        self.head_projections_valid.clear()
        self.on_screen.clear()
        self.yaw_actions.clear()
        self.pitch_actions.clear()
        self.correction_candidates.clear()
        self.corrective_actions.clear()
        self.pitch_correction_candidates.clear()
        self.pitch_corrective_actions.clear()
        self.crosshair_hits.clear()
        self.scripted_attacks = 0

    def add(
        self,
        state: tuple[float, ...],
        raw_action: np.ndarray,
        log_probabilities: np.ndarray,
        value: np.ndarray,
        yaw_reward: float,
        pitch_reward: float,
        before: dict[str, Any],
        after: dict[str, Any],
        normalized_action: np.ndarray,
        scripted_attack: bool,
    ) -> None:
        self.observations.append(state)
        self.raw_actions.append(raw_action)
        self.log_probabilities.append(np.asarray(log_probabilities, dtype=np.float32))
        self.yaw_rewards.append(float(yaw_reward))
        self.pitch_rewards.append(float(pitch_reward))
        self.rewards.append(float(yaw_reward + pitch_reward))
        self.values.append(np.asarray(value, dtype=np.float32))
        features = before.get("features", {})
        error = target_yaw_error_degrees(before)
        self.target_valid.append(bool(features.get("target_valid", False)))
        self.projections_valid.append(error is not None)
        self.on_screen.append(bool(features.get("target_on_screen", False)))
        if error is not None:
            self.yaw_errors.append(abs(error))
        pitch_error = head_aim_error(before)
        self.head_projections_valid.append(pitch_error is not None)
        if pitch_error is not None:
            self.pitch_errors.append(abs(pitch_error))
            self.signed_pitch_errors.append(pitch_error)
        self.yaw_actions.append(float(normalized_action[0]) * MAX_YAW_DELTA_DEGREES)
        self.pitch_actions.append(float(normalized_action[1]) * MAX_PITCH_DELTA_DEGREES)
        self.correction_candidates.append(error is not None and abs(error) > 1.0 and abs(normalized_action[0]) > 0.01)
        self.corrective_actions.append(
            self.correction_candidates[-1]
            and math.copysign(1.0, error) == math.copysign(1.0, float(normalized_action[0]))
        )
        pitch_action = float(normalized_action[1])
        pitch_candidate = (
            pitch_error is not None and abs(pitch_error) > 0.015 and abs(pitch_action) > 0.01
        )
        self.pitch_correction_candidates.append(pitch_candidate)
        # Minecraft positive pitch looks down. If the head point is below the
        # reticle (positive screen error), a positive delta moves it upward on
        # screen; the desired action therefore has the same sign as the error.
        self.pitch_corrective_actions.append(
            pitch_candidate
            and math.copysign(1.0, pitch_error) == math.copysign(1.0, pitch_action)
        )
        self.crosshair_hits.append(bool(after.get("features", {}).get("crosshair_on_target", False)))
        self.scripted_attacks += int(scripted_attack)


def ppo_update(
    model: YawActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: Rollout,
    bootstrap_value: np.ndarray,
    device: torch.device,
    entropy_coefficient: float,
) -> dict[str, float]:
    observations = torch.as_tensor(np.asarray(rollout.observations), dtype=torch.float32, device=device)
    raw_actions = torch.as_tensor(np.asarray(rollout.raw_actions, dtype=np.float32).reshape(-1, ACTION_SIZE),
                                  dtype=torch.float32, device=device)
    old_logp = torch.as_tensor(np.asarray(rollout.log_probabilities), dtype=torch.float32, device=device)
    rewards = np.column_stack((rollout.yaw_rewards, rollout.pitch_rewards)).astype(np.float32)
    values = np.asarray(rollout.values, dtype=np.float32)
    bootstrap = np.asarray(bootstrap_value, dtype=np.float32).reshape(ACTION_SIZE)
    if rewards.shape != (len(rollout), ACTION_SIZE) or values.shape != rewards.shape:
        raise ValueError("rollout must have one yaw/pitch reward and value per applied action")
    if old_logp.shape != raw_actions.shape or bootstrap.shape != (ACTION_SIZE,):
        raise ValueError("rollout action probabilities or bootstrap values have invalid dimensions")

    advantages = np.zeros_like(rewards)
    gae = np.zeros(ACTION_SIZE, dtype=np.float32)
    for index in reversed(range(len(rewards))):
        next_value = bootstrap if index == len(rewards) - 1 else values[index + 1]
        delta = rewards[index] + 0.99 * next_value - values[index]
        gae = delta + 0.99 * 0.95 * gae
        advantages[index] = gae
    returns = advantages + values
    advantages = (advantages - advantages.mean(axis=0, keepdims=True)) / (
        advantages.std(axis=0, keepdims=True) + 1e-8
    )
    advantage_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=device)
    returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=device)

    metrics: dict[str, list[float]] = {
        "policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_fraction": [],
    }
    sample_count = len(rollout)
    minibatch_size = 256
    for _ in range(4):
        permutation = torch.randperm(sample_count, device=device)
        for start in range(0, sample_count, minibatch_size):
            indices = permutation[start:start + minibatch_size]
            distribution, value_prediction = model.distribution_and_value(observations[indices])
            new_logp = model.squashed_log_probabilities(distribution, raw_actions[indices])
            log_ratio = new_logp - old_logp[indices]
            ratio = log_ratio.exp()
            batch_advantages = advantage_tensor[indices]
            clipped_surrogate = torch.minimum(
                ratio * batch_advantages,
                torch.clamp(ratio, 0.8, 1.2) * batch_advantages,
            )
            # Sum the independently clipped axis losses to retain the scale of
            # the former two-action policy while preventing cross-axis credit.
            policy_loss = -clipped_surrogate.mean(dim=0).sum()
            value_loss = 0.5 * (value_prediction - returns_tensor[indices]).pow(2).mean()
            entropy = -new_logp.sum(dim=-1).mean()
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


def save_checkpoint(
    path: Path,
    model: YawActorCritic,
    optimizer: torch.optim.Optimizer,
    update_number: int,
    total_steps: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    torch.save(
        {
            "format": 3,
            "trainer": JOINT_TRAINER,
            "reward_layout": "separate_yaw_pitch",
            "observation_schema": SCHEMA,
            "observation_size": JOINT_OBSERVATION_SIZE,
            "action_size": ACTION_SIZE,
            "max_yaw_delta_degrees": MAX_YAW_DELTA_DEGREES,
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
    print(f"Saved checkpoint: {path} (update={update_number}, steps={total_steps})", flush=True)


def load_checkpoint(
    path: Path,
    model: YawActorCritic,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("observation_schema") != SCHEMA:
        raise RuntimeError("checkpoint observation schema does not match this mod")
    saved_model = checkpoint["model"]
    if checkpoint.get("trainer") == JOINT_TRAINER:
        if int(checkpoint.get("observation_size", -1)) != JOINT_OBSERVATION_SIZE or int(
            checkpoint.get("action_size", -1)
        ) != ACTION_SIZE:
            raise RuntimeError("joint checkpoint policy dimensions do not match this trainer")
        if float(checkpoint.get("max_yaw_delta_degrees", -1)) != MAX_YAW_DELTA_DEGREES or float(
            checkpoint.get("max_pitch_delta_degrees", -1)
        ) != MAX_PITCH_DELTA_DEGREES:
            raise RuntimeError("joint checkpoint aim action scales do not match this trainer")
        model.load_state_dict(saved_model)
        optimizer.load_state_dict(checkpoint["optimizer"])
    elif (
        checkpoint.get("trainer") == LEGACY_JOINT_TRAINER
        and int(checkpoint.get("observation_size", -1)) == JOINT_OBSERVATION_SIZE
        and int(checkpoint.get("action_size", -1)) == ACTION_SIZE
        and float(checkpoint.get("max_yaw_delta_degrees", -1)) == MAX_YAW_DELTA_DEGREES
        and float(checkpoint.get("max_pitch_delta_degrees", -1)) == MAX_PITCH_DELTA_DEGREES
    ):
        # Keep old joint actor weights but replace its single shared critic with
        # separate yaw/pitch value heads. Its Adam moments are not valid for the
        # new reward decomposition, so the optimizer intentionally stays fresh.
        expanded = model.state_dict()
        if set(expanded) != set(saved_model):
            raise RuntimeError("legacy joint checkpoint parameter names differ from this trainer")
        for name, value in saved_model.items():
            if name == "critic.weight":
                if value.shape != (1, expanded[name].shape[1]):
                    raise RuntimeError("legacy joint critic weight dimensions are unexpected")
                expanded[name][0:1].copy_(value)
                expanded[name][1:2].zero_()
            elif name == "critic.bias":
                if value.shape != (1,):
                    raise RuntimeError("legacy joint critic bias dimensions are unexpected")
                expanded[name][0:1].copy_(value)
                expanded[name][1:2].zero_()
            else:
                if expanded[name].shape != value.shape:
                    raise RuntimeError(f"legacy joint checkpoint parameter shape differs: {name}")
                expanded[name].copy_(value)
        model.load_state_dict(expanded)
        print(
            "Migrated legacy joint weights to separate yaw/pitch critics; "
            "the optimizer was reset for independent reward advantages.",
            flush=True,
        )
    elif (
        checkpoint.get("trainer") == "horizontal_yaw_ppo_v1"
        and int(checkpoint.get("observation_size", -1)) == YAW_OBSERVATION_SIZE
        and int(checkpoint.get("action_size", -1)) == 1
        and float(checkpoint.get("max_yaw_delta_degrees", -1)) == MAX_YAW_DELTA_DEGREES
    ):
        # Preserve the trained yaw path exactly. The additional pitch inputs
        # and output are neutral at migration; the changed parameter shapes
        # require fresh Adam state.
        expanded = model.state_dict()
        missing = set(expanded) - set(saved_model)
        unexpected = set(saved_model) - set(expanded)
        if missing or unexpected:
            raise RuntimeError(
                f"yaw checkpoint parameter names differ (missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)})"
            )
        for name, value in saved_model.items():
            if name == "shared.0.weight":
                if value.shape != (expanded[name].shape[0], YAW_OBSERVATION_SIZE):
                    raise RuntimeError("yaw checkpoint first-layer dimensions are unexpected")
                expanded[name][:, :YAW_OBSERVATION_SIZE].copy_(value)
                expanded[name][:, YAW_OBSERVATION_SIZE:].zero_()
            elif name == "actor_mean.weight":
                expanded[name][0:1].copy_(value)
                expanded[name][1:2].zero_()
            elif name == "actor_mean.bias":
                expanded[name][0:1].copy_(value)
                expanded[name][1:2].zero_()
            elif name == "log_std":
                expanded[name][0:1].copy_(value)
                expanded[name][1] = math.log(0.08)
            elif name == "critic.weight":
                if value.shape != (1, expanded[name].shape[1]):
                    raise RuntimeError("yaw checkpoint critic weight dimensions are unexpected")
                expanded[name][0:1].copy_(value)
                expanded[name][1:2].zero_()
            elif name == "critic.bias":
                if value.shape != (1,):
                    raise RuntimeError("yaw checkpoint critic bias dimensions are unexpected")
                expanded[name][0:1].copy_(value)
                expanded[name][1:2].zero_()
            else:
                if expanded[name].shape != value.shape:
                    raise RuntimeError(f"unexpected yaw checkpoint parameter shape: {name}")
                expanded[name].copy_(value)
        model.load_state_dict(expanded)
        print(
            "Migrated the trained yaw checkpoint: yaw weights and 10° action scale retained; "
            "pitch starts with a neutral mean and small exploration (optimizer reset).",
            flush=True,
        )
    else:
        raise RuntimeError("checkpoint is neither a supported yaw checkpoint nor a joint yaw-pitch checkpoint")
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if checkpoint.get("torch_cuda_rng_state") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint["torch_cuda_rng_state"])
    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])
    return int(checkpoint.get("update_number", 0)), int(checkpoint.get("total_steps", 0))


def log_update(update: int, total_steps: int, rollout: Rollout,
               metrics: dict[str, float], model: YawActorCritic, entropy_coefficient: float,
               stale_retries: int, superseded_observations: int) -> None:
    errors = rollout.yaw_errors
    mean_error = float(np.mean(errors)) if errors else float("nan")
    median_error = float(np.median(errors)) if errors else float("nan")
    centered_2 = 100.0 * sum(error <= 2.0 for error in errors) / len(errors) if errors else 0.0
    centered_5 = 100.0 * sum(error <= 5.0 for error in errors) / len(errors) if errors else 0.0
    target_pct = 100.0 * sum(rollout.target_valid) / ROLLOUT_STEPS
    projected_pct = 100.0 * sum(rollout.projections_valid) / ROLLOUT_STEPS
    on_screen_pct = 100.0 * sum(rollout.on_screen) / ROLLOUT_STEPS
    crosshair_pct = 100.0 * sum(rollout.crosshair_hits) / ROLLOUT_STEPS
    pitch_errors = rollout.pitch_errors
    signed_pitch_errors = rollout.signed_pitch_errors
    mean_pitch_error = float(np.mean(pitch_errors)) if pitch_errors else float("nan")
    median_pitch_error = float(np.median(pitch_errors)) if pitch_errors else float("nan")
    mean_signed_pitch_error = float(np.mean(signed_pitch_errors)) if signed_pitch_errors else float("nan")
    pitch_centered_pct = (
        100.0 * sum(error <= 0.015 for error in pitch_errors) / len(pitch_errors)
        if pitch_errors else 0.0
    )
    actions = np.asarray(rollout.yaw_actions, dtype=np.float32)
    pitch_actions = np.asarray(rollout.pitch_actions, dtype=np.float32)
    mean_abs_action = float(np.mean(np.abs(actions))) if len(actions) else 0.0
    mean_abs_pitch_action = float(np.mean(np.abs(pitch_actions))) if len(pitch_actions) else 0.0
    mean_signed_pitch_action = float(np.mean(pitch_actions)) if len(pitch_actions) else 0.0
    action_std = model.log_std.detach().exp().cpu().numpy() * np.asarray(
        [MAX_YAW_DELTA_DEGREES, MAX_PITCH_DELTA_DEGREES], dtype=np.float32
    )
    left_pct = 100.0 * sum(action < -0.1 for action in actions) / ROLLOUT_STEPS
    right_pct = 100.0 * sum(action > 0.1 for action in actions) / ROLLOUT_STEPS
    corrective_pct = (100.0 * sum(rollout.corrective_actions) / max(1, sum(rollout.correction_candidates)))
    pitch_corrective_pct = (
        100.0 * sum(rollout.pitch_corrective_actions) / max(1, sum(rollout.pitch_correction_candidates))
    )
    head_projected_pct = (
        100.0 * sum(rollout.head_projections_valid) / max(1, len(rollout.head_projections_valid))
    )
    stale_pct = 100.0 * stale_retries / max(1, total_steps + stale_retries)
    print(
        f"update={update} steps={total_steps} avg_reward={float(np.mean(rollout.rewards)):+.4f} "
        f"avg_yaw_reward={float(np.mean(rollout.yaw_rewards)):+.4f} "
        f"avg_pitch_reward={float(np.mean(rollout.pitch_rewards)):+.4f} "
        f"entropy={metrics['entropy']:.4f} mean_abs_yaw_error_deg={mean_error:.2f} "
        f"median_yaw_error_deg={median_error:.2f} centered_2deg={centered_2:.1f}% centered_5deg={centered_5:.1f}% "
        f"mean_abs_head_pitch_error_screen={mean_pitch_error:.4f} "
        f"median_head_pitch_error_screen={median_pitch_error:.4f} "
        f"mean_signed_head_pitch_error_screen={mean_signed_pitch_error:+.4f} "
        f"head_projected={head_projected_pct:.1f}% head_centered_1.5pct={pitch_centered_pct:.1f}% "
        f"target_valid={target_pct:.1f}% projected={projected_pct:.1f}% on_screen={on_screen_pct:.1f}% "
        f"crosshair_on_target={crosshair_pct:.1f}% scripted_attacks={rollout.scripted_attacks} "
        f"mean_abs_yaw_action_deg={mean_abs_action:.2f} mean_abs_pitch_action_deg={mean_abs_pitch_action:.2f} "
        f"mean_signed_pitch_action_deg={mean_signed_pitch_action:+.3f} "
        f"policy_std_deg=({action_std[0]:.2f},{action_std[1]:.2f}) "
        f"negative_yaw={left_pct:.1f}% positive_yaw={right_pct:.1f}% corrective_direction={corrective_pct:.1f}% "
        f"pitch_corrective_direction={pitch_corrective_pct:.1f}% "
        f"stale_retries={stale_retries} ({stale_pct:.3f}%) superseded_frames={superseded_observations} "
        f"entropy_coef={entropy_coefficient:.4f} policy_loss={metrics['policy_loss']:+.4f} "
        f"value_loss={metrics['value_loss']:.4f} approx_kl={metrics['approx_kl']:.5f} "
        f"clip_fraction={metrics['clip_fraction']:.3f}",
        flush=True,
    )


def train(args: argparse.Namespace) -> int:
    global STOP_REQUESTED
    signal.signal(signal.SIGINT, request_stop)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # This is a tiny batch-1 network in the latency-sensitive control loop.
    # A single inference thread avoids OpenMP launch/scheduling overhead.
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}; PPO rollout size is exactly {ROLLOUT_STEPS} applied steps per update.", flush=True)
    print(
        f"Policy: joint yaw ±{MAX_YAW_DELTA_DEGREES:.1f}° and pitch ±{MAX_PITCH_DELTA_DEGREES:.1f}° per tick; "
        "yaw and pitch use separate reward advantages; "
        "movement/jump/sprint/sneak/use are disabled; sword attacks are scripted only when "
        "cooldown is full and the crosshair hits the target.", flush=True,
    )

    model = YawActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
    update_number = 0
    total_steps = 0
    if not args.no_resume and args.checkpoint.exists():
        update_number, total_steps = load_checkpoint(args.checkpoint, model, optimizer, device)
        print(f"Resumed {args.checkpoint} at update {update_number}, step {total_steps}.", flush=True)
    elif not args.no_resume and args.init_from.exists():
        if args.checkpoint.resolve() == args.init_from.resolve():
            raise RuntimeError(
                "joint training needs a separate output checkpoint so the yaw-only source is preserved; "
                "choose a different --checkpoint path"
            )
        update_number, total_steps = load_checkpoint(args.init_from, model, optimizer, device)
        print(
            f"Initialized joint yaw-pitch training from {args.init_from} at update {update_number}, "
            f"step {total_steps}; future saves go to {args.checkpoint}.",
            flush=True,
        )
    else:
        print("Starting a fresh, neutral-initialized joint policy.", flush=True)

    connection: MinecraftConnection | None = None
    rollout = Rollout()
    last_tick = -1
    previous_yaw_error: float | None = None
    previous_pitch_error: float | None = None
    previous_yaw_action = 0.0
    prior_yaw_action = 0.0
    previous_pitch_action = 0.0
    prior_pitch_action = 0.0
    attack_latched = False
    consecutive_stale = 0
    total_stale_retries = 0
    superseded_observations = 0
    try:
        connection = MinecraftConnection(args.host, args.port)
        while not STOP_REQUESTED:
            if not connection.enabled:
                save_checkpoint(args.checkpoint, model, optimizer, update_number, total_steps)
                rollout.clear()
                previous_yaw_error = None
                previous_pitch_error = None
                previous_yaw_action = 0.0
                prior_yaw_action = 0.0
                previous_pitch_action = 0.0
                prior_pitch_action = 0.0
                attack_latched = False
                print("Training paused. Press K in Minecraft to resume; this process will keep waiting.", flush=True)
                connection.wait_until_enabled()
                if STOP_REQUESTED:
                    break
                print("Resuming with a fresh rollout; the policy and optimizer were preserved.", flush=True)

            frame = connection.wait_for_observation(after_tick=last_tick)
            if frame is None:
                continue

            while connection.enabled and not STOP_REQUESTED:
                tick = int(frame.get("tick", -1))
                if tick <= last_tick:
                    frame = connection.wait_for_observation(after_tick=last_tick)
                    if frame is None:
                        break
                    continue
                last_tick = tick
                state = encode_yaw_pitch_observation(
                    frame, previous_yaw_error, previous_yaw_action, prior_yaw_action,
                    previous_pitch_error, previous_pitch_action, prior_pitch_action,
                )
                raw_action, old_log_probabilities, value = model.act(state, device)
                normalized_action = np.tanh(raw_action).astype(np.float32)
                yaw_delta = float(normalized_action[0]) * MAX_YAW_DELTA_DEGREES
                pitch_delta = float(normalized_action[1]) * MAX_PITCH_DELTA_DEGREES
                attack_press, next_attack_latched = scripted_attack_decision(frame, attack_latched)
                fresher_frame = connection.take_newer_observation(after_tick=tick)
                if not connection.enabled:
                    break
                if fresher_frame is not None:
                    frame = fresher_frame
                    superseded_observations += 1
                    continue
                control_result = connection.send_aim(tick, yaw_delta, pitch_delta, attack=attack_press)
                if control_result == "stale":
                    consecutive_stale += 1
                    total_stale_retries += 1
                    if total_stale_retries == 1 or total_stale_retries % 10 == 0:
                        print(
                            f"stale_retry_total={total_stale_retries}; "
                            f"{connection.last_stale_detail}; "
                            "recomputing from a newer observation",
                            flush=True,
                        )
                    frame = connection.wait_for_observation(after_tick=tick)
                    previous_yaw_error = None
                    previous_pitch_error = None
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
                    frame, next_frame,
                    float(normalized_action[0]), previous_yaw_action,
                    float(normalized_action[1]), previous_pitch_action,
                    scripted_attack=attack_press,
                )
                rollout.add(state, raw_action, old_log_probabilities, value,
                            yaw_reward, pitch_reward, frame, next_frame,
                            normalized_action, attack_press)
                total_steps += 1

                prior_yaw_action, previous_yaw_action = previous_yaw_action, float(normalized_action[0])
                prior_pitch_action, previous_pitch_action = previous_pitch_action, float(normalized_action[1])
                previous_yaw_error = target_yaw_error_degrees(frame)
                previous_pitch_error = head_aim_error(frame)
                frame = next_frame

                if len(rollout) == ROLLOUT_STEPS:
                    next_state = encode_yaw_pitch_observation(
                        frame, previous_yaw_error, previous_yaw_action, prior_yaw_action,
                        previous_pitch_error, previous_pitch_action, prior_pitch_action,
                    )
                    with torch.no_grad():
                        bootstrap = model.distribution_and_value(
                            torch.as_tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
                        )[1].squeeze(0).cpu().numpy()
                    next_update = update_number + 1
                    entropy_coefficient = max(0.001, 0.01 * math.exp(-next_update / 120.0))
                    metrics = ppo_update(model, optimizer, rollout, bootstrap, device, entropy_coefficient)
                    update_number = next_update
                    log_update(update_number, total_steps, rollout, metrics, model, entropy_coefficient,
                               total_stale_retries, superseded_observations)
                    rollout.clear()
                    if update_number % args.checkpoint_every == 0:
                        save_checkpoint(args.checkpoint, model, optimizer, update_number, total_steps)

        print("Stopping; saving the latest completed PPO update.", flush=True)
        save_checkpoint(args.checkpoint, model, optimizer, update_number, total_steps)
        return 0
    except StopRequested:
        print("Stop requested; saving the latest completed PPO update.", flush=True)
        save_checkpoint(args.checkpoint, model, optimizer, update_number, total_steps)
        return 0
    except (OSError, ConnectionError, TimeoutError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"Training stopped: {error}", file=sys.stderr, flush=True)
        save_checkpoint(args.checkpoint, model, optimizer, update_number, total_steps)
        return 1
    finally:
        if connection is not None:
            connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="MinecraftAI loopback address")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="MinecraftAI control port")
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
        help=f"legacy joint yaw-pitch checkpoint (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--init-from", type=Path, default=DEFAULT_YAW_CHECKPOINT,
        help=f"yaw-only checkpoint used to initialize joint training when no joint checkpoint exists (default: {DEFAULT_YAW_CHECKPOINT})",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=5,
        help="save after this many completed 1024-step PPO updates",
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed for a new policy")
    parser.add_argument("--no-resume", action="store_true", help="ignore any existing checkpoint and start fresh")
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
