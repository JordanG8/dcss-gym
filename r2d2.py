"""Recurrent Q-network and prioritized sequence replay for DCSS.

The policy receives only the same player-visible encoded screen and legal-action
mask as PPO. Memory summarizes observation/action history; it does not expose
game internals.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_rl import D_MODEL, Policy, apply_action_mask


class RecurrentQ(nn.Module):
    """Spatial encoder followed by a GRU and dueling discrete-action head."""

    def __init__(self, n_actions: int):
        super().__init__()
        self.n_actions = n_actions
        self.spatial = Policy(n_actions)
        self.core = nn.GRUCell(D_MODEL + n_actions, D_MODEL)
        self.advantage = nn.Linear(D_MODEL, n_actions)
        self.value = nn.Linear(D_MODEL, 1)
        nn.init.orthogonal_(self.advantage.weight, gain=0.01)
        nn.init.zeros_(self.advantage.bias)
        # Memory is a residual over the spatial encoder. Zero initialization
        # makes a warm-started recurrent policy behaviour-neutral on day one;
        # replay then learns which history should alter the current features.
        for parameter in self.core.parameters():
            nn.init.zeros_(parameter)

    def warm_start_spatial(self, state_dict):
        """Preserve a feed-forward checkpoint's initial action distribution."""
        from train_rl import load_policy_state
        report = load_policy_state(self.spatial, state_dict)
        with torch.no_grad():
            self.advantage.weight.copy_(self.spatial.actor.weight)
            self.advantage.bias.copy_(self.spatial.actor.bias)
            self.value.weight.copy_(self.spatial.critic.weight)
            self.value.bias.copy_(self.spatial.critic.bias)
        return report

    def initial_state(self, batch: int, device=None):
        return torch.zeros(batch, D_MODEL, device=device)

    def step(self, observation, previous_action, hidden, action_mask=None):
        spatial = self.spatial.features(observation)
        previous = torch.zeros(
            observation.shape[0], self.n_actions,
            device=observation.device, dtype=spatial.dtype)
        valid = previous_action >= 0
        if bool(valid.any()):
            previous[valid] = F.one_hot(
                previous_action[valid], self.n_actions).to(spatial.dtype)
        hidden = self.core(torch.cat((spatial, previous), dim=-1), hidden)
        features = spatial + hidden
        advantage = self.advantage(features)
        value = self.value(features)
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        if action_mask is not None:
            q = apply_action_mask(q, action_mask)
        return q, hidden


@dataclass
class Transition:
    observation: torch.Tensor
    previous_action: int
    action: int
    reward: float
    done: bool
    action_mask: torch.Tensor
    next_observation: torch.Tensor
    next_action_mask: torch.Tensor


@dataclass
class SequenceSample:
    replay_index: int
    transitions: list[Transition]
    burn_in: int
    weight: float


class PrioritizedEpisodeReplay:
    """Episode replay that samples learn windows with preceding burn-in."""

    def __init__(self, capacity_steps=100_000, alpha=0.6):
        self.capacity_steps = capacity_steps
        self.alpha = alpha
        self.episodes = []
        self.priorities = []
        self.steps = 0

    def __len__(self):
        return self.steps

    def add(self, episode):
        episode = list(episode)
        if not episode:
            return
        self.episodes.append(episode)
        self.priorities.append(max(self.priorities, default=1.0))
        self.steps += len(episode)
        while self.steps > self.capacity_steps and len(self.episodes) > 1:
            self.steps -= len(self.episodes.pop(0))
            self.priorities.pop(0)

    def sample(self, batch_size, burn_in=10, unroll=20, beta=0.4):
        if not self.episodes:
            raise ValueError("cannot sample empty replay")
        p = torch.tensor(self.priorities, dtype=torch.float64).clamp_min(1e-6)
        p = p.pow(self.alpha)
        p /= p.sum()
        picks = torch.multinomial(p, batch_size, replacement=True).tolist()
        samples = []
        for replay_index in picks:
            episode = self.episodes[replay_index]
            learn_start = random.randrange(len(episode))
            start = max(0, learn_start - burn_in)
            end = min(len(episode), learn_start + unroll)
            probability = float(p[replay_index])
            weight = (len(self.episodes) * probability) ** (-beta)
            samples.append(SequenceSample(
                replay_index, episode[start:end], learn_start - start, weight))
        scale = max(sample.weight for sample in samples)
        for sample in samples:
            sample.weight /= scale
        return samples

    def update_priorities(self, indices, errors):
        for index, error in zip(indices, errors):
            self.priorities[index] = float(abs(error)) + 1e-5
