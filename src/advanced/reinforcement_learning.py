"""Skeleton for future deep reinforcement learning trading environment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - optional dependency
    import gym  # type: ignore


@dataclass
class TradingEnvConfig:
    window_size: int = 60
    transaction_cost: float = 0.0001


class TradingEnv(gym.Env):
    """Minimal trading environment stub for RL experimentation."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, prices: np.ndarray, features: np.ndarray, config: TradingEnvConfig | None = None) -> None:
        super().__init__()
        self.config = config or TradingEnvConfig()
        self.prices = prices
        self.features = features
        self.current_step = self.config.window_size
        self.position = 0.0
        self.cash = 0.0
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))
        feature_dim = features.shape[1]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.config.window_size, feature_dim + 2))

    def _get_observation(self) -> np.ndarray:
        window_prices = self.prices[self.current_step - self.config.window_size : self.current_step]
        window_features = self.features[self.current_step - self.config.window_size : self.current_step]
        pos = np.full((self.config.window_size, 1), self.position)
        cash = np.full((self.config.window_size, 1), self.cash)
        return np.hstack([window_prices.reshape(-1, 1), window_features, pos, cash])

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:  # type: ignore[override]
        target_position = float(np.clip(action[0], -1.0, 1.0))
        price = self.prices[self.current_step]
        reward = (target_position * (self.prices[self.current_step] - self.prices[self.current_step - 1])) - self.config.transaction_cost * abs(target_position - self.position)
        self.position = target_position
        self.cash += reward
        self.current_step += 1
        terminated = self.current_step >= len(self.prices) - 1
        observation = self._get_observation()
        info = {"cash": self.cash}
        return observation, float(reward), terminated, False, info

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):  # type: ignore[override]
        super().reset(seed=seed)
        self.current_step = self.config.window_size
        self.position = 0.0
        self.cash = 0.0
        return self._get_observation(), {}

    def render(self, mode: str = "human") -> None:  # pragma: no cover - stub
        pass


def train_rl_agent(environment: TradingEnv) -> None:
    """Placeholder RL training loop (e.g., DQN or PPO)."""

    # TODO: Integrate with stable-baselines3 or custom RL algorithms.
    for episode in range(10):
        obs, _ = environment.reset()
        done = False
        while not done:
            action = environment.action_space.sample()
            obs, reward, done, _, info = environment.step(action)
            # Insert agent update logic here.
    print("RL training stub complete")
