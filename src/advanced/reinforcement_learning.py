"""Deep reinforcement learning trading environment and training utilities.

This module provides:
- TradingEnv: A gymnasium environment for RL agent training
- train_rl_agent: Training function using stable-baselines3 PPO/A2C

The environment simulates trading with:
- Position management (long/short/flat)
- Transaction costs
- Risk-adjusted rewards (Sharpe-based)
- Equity curve tracking

Dependencies:
    - gymnasium (or gym for backward compatibility)
    - stable-baselines3 (for PPO/A2C algorithms)
    - numpy, pandas

Example usage:
    >>> from src.advanced.reinforcement_learning import TradingEnv, train_rl_agent
    >>> env = TradingEnv(prices, features, config=TradingEnvConfig(window_size=60))
    >>> model = train_rl_agent(env, total_timesteps=10000)
    >>> obs, _ = env.reset()
    >>> action, _ = model.predict(obs)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - optional dependency
    import gym  # type: ignore
    from gym import spaces  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class TradingEnvConfig:
    """Configuration for the TradingEnv environment.

    Parameters
    ----------
    window_size : int, default=60
        Number of historical bars to include in observation.
    transaction_cost : float, default=0.0001
        Transaction cost as a fraction of trade value (0.0001 = 1 basis point).
    initial_capital : float, default=10000.0
        Starting capital for the trading simulation.
    max_position : float, default=1.0
        Maximum position size (1.0 = fully invested).
    reward_scaling : float, default=1.0
        Multiplier for rewards to help with learning stability.
    use_sharpe_reward : bool, default=True
        If True, use Sharpe-ratio based rewards instead of raw returns.
    sharpe_window : int, default=20
        Rolling window for Sharpe ratio calculation.
    """
    window_size: int = 60
    transaction_cost: float = 0.0001
    initial_capital: float = 10000.0
    max_position: float = 1.0
    reward_scaling: float = 1.0
    use_sharpe_reward: bool = True
    sharpe_window: int = 20


class TradingEnv(gym.Env):
    """Trading environment for reinforcement learning.

    This environment simulates a simple trading scenario where an agent
    can take positions (long/short/flat) based on price and feature data.
    Rewards are based on risk-adjusted returns (Sharpe ratio).

    Observation Space:
        Box containing:
        - Historical price returns (window_size,)
        - Historical features (window_size, n_features)
        - Current position
        - Current cash/equity ratio

    Action Space:
        Box(-1, 1) representing target position:
        - -1: Full short position
        - 0: Flat (no position)
        - 1: Full long position

    Parameters
    ----------
    prices : np.ndarray
        1D array of price data.
    features : np.ndarray
        2D array of feature data (n_samples, n_features).
    config : TradingEnvConfig, optional
        Environment configuration. Uses defaults if not provided.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        prices: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
        config: Optional[TradingEnvConfig] = None,
    ) -> None:
        """Initialize the trading environment."""
        super().__init__()

        self.config = config or TradingEnvConfig()

        # Handle default/empty initialization
        if prices is None:
            prices = np.array([100.0] * (self.config.window_size + 100))
        if features is None:
            features = np.zeros((len(prices), 1))

        self.prices = np.asarray(prices, dtype=np.float32)
        self.features = np.asarray(features, dtype=np.float32)

        # Ensure features is 2D
        if self.features.ndim == 1:
            self.features = self.features.reshape(-1, 1)

        self.n_features = self.features.shape[1]

        # Validate data length
        min_length = self.config.window_size + 10
        if len(self.prices) < min_length:
            raise ValueError(
                f"prices array too short: got {len(self.prices)}, "
                f"need at least {min_length}"
            )
        if len(self.features) != len(self.prices):
            raise ValueError(
                f"features length ({len(self.features)}) must match "
                f"prices length ({len(self.prices)})"
            )

        # Define action space: continuous position target [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # Define observation space
        # Observation includes: price returns, features, position, cash ratio
        obs_dim = self.config.window_size * (1 + self.n_features) + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Initialize state
        self.current_step = self.config.window_size
        self.position = 0.0
        self.cash = self.config.initial_capital
        self.equity_history: list = []
        self.returns_history: list = []

    def _get_observation(self) -> np.ndarray:
        """Construct observation from current state."""
        # Get window of price returns
        window_prices = self.prices[
            self.current_step - self.config.window_size : self.current_step
        ]
        # Calculate returns
        price_returns = np.diff(window_prices) / window_prices[:-1]
        # Pad to maintain window size
        price_returns = np.concatenate([[0], price_returns])

        # Get window of features
        window_features = self.features[
            self.current_step - self.config.window_size : self.current_step
        ]

        # Current state indicators
        current_equity = self._get_equity()
        cash_ratio = self.cash / current_equity if current_equity > 0 else 0.0

        # Flatten and combine
        obs = np.concatenate([
            price_returns.flatten(),
            window_features.flatten(),
            [self.position, cash_ratio],
        ]).astype(np.float32)

        return obs

    def _get_equity(self) -> float:
        """Calculate current equity value.

        The position represents a fraction of initial capital invested.
        Position of 1.0 means fully long, -1.0 means fully short, 0 is flat.
        Equity = cash (which already includes initial capital and accumulated PnL).
        """
        return self.cash

    def _calculate_reward(self, pnl: float) -> float:
        """Calculate reward based on configuration.

        Uses Sharpe-ratio based reward if configured, otherwise raw PnL.
        """
        self.returns_history.append(pnl)

        if not self.config.use_sharpe_reward or len(self.returns_history) < 2:
            return pnl * self.config.reward_scaling

        # Calculate rolling Sharpe-like reward
        recent_returns = np.array(
            self.returns_history[-self.config.sharpe_window:]
        )
        mean_return = np.mean(recent_returns)
        std_return = np.std(recent_returns)

        if std_return > 0:
            sharpe = mean_return / std_return
        else:
            sharpe = mean_return

        return sharpe * self.config.reward_scaling

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment.

        Parameters
        ----------
        action : np.ndarray
            Target position as array of shape (1,), value in [-1, 1].

        Returns
        -------
        observation : np.ndarray
            New observation after the step.
        reward : float
            Reward from this step.
        terminated : bool
            Whether the episode has ended.
        truncated : bool
            Whether the episode was truncated (always False).
        info : dict
            Additional information (equity, drawdown, etc.).
        """
        # Clip action to valid range
        target_position = float(np.clip(action[0], -1.0, 1.0))

        # Get price change
        prev_price = self.prices[self.current_step - 1]
        current_price = self.prices[self.current_step]
        price_return = (current_price - prev_price) / prev_price

        # Calculate PnL from position (position is fraction of capital invested)
        # PnL = position * price_return * initial_capital
        position_pnl = self.position * price_return * self.config.initial_capital

        # Calculate transaction cost from position change
        position_change = abs(target_position - self.position)
        transaction_cost = (
            position_change * self.config.transaction_cost * self.config.initial_capital
        )

        # Update state
        self.cash += position_pnl - transaction_cost
        self.position = target_position

        # Track equity
        current_equity = self._get_equity()
        self.equity_history.append(current_equity)

        # Calculate reward
        pnl = position_pnl - transaction_cost
        reward = self._calculate_reward(pnl)

        # Move to next step
        self.current_step += 1

        # Check if episode is done
        terminated = self.current_step >= len(self.prices) - 1
        truncated = False

        # Calculate drawdown for info
        if len(self.equity_history) > 0:
            peak = max(self.equity_history)
            drawdown = (peak - current_equity) / peak if peak > 0 else 0.0
        else:
            drawdown = 0.0

        observation = self._get_observation()
        info = {
            "equity": current_equity,
            "cash": self.cash,
            "position": self.position,
            "drawdown": drawdown,
            "total_return": (current_equity - self.config.initial_capital)
            / self.config.initial_capital,
        }

        return observation, float(reward), terminated, truncated, info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.
        options : dict, optional
            Additional options (not used).

        Returns
        -------
        observation : np.ndarray
            Initial observation.
        info : dict
            Initial info dict.
        """
        super().reset(seed=seed)

        self.current_step = self.config.window_size
        self.position = 0.0
        self.cash = self.config.initial_capital
        self.equity_history = [self.config.initial_capital]
        self.returns_history = []

        observation = self._get_observation()
        info = {
            "equity": self.config.initial_capital,
            "cash": self.cash,
            "position": 0.0,
            "drawdown": 0.0,
            "total_return": 0.0,
        }

        return observation, info

    def render(self, mode: str = "human") -> None:  # pragma: no cover
        """Render the environment state."""
        if mode == "human":
            equity = self._get_equity()
            print(
                f"Step {self.current_step}: "
                f"Position={self.position:.2f}, "
                f"Equity=${equity:.2f}, "
                f"Cash=${self.cash:.2f}"
            )


def train_rl_agent(
    environment: TradingEnv,
    algorithm: str = "PPO",
    total_timesteps: int = 10000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    verbose: int = 1,
    tensorboard_log: Optional[str] = None,
    seed: Optional[int] = None,
) -> Any:
    """Train an RL agent on the trading environment.

    Parameters
    ----------
    environment : TradingEnv
        The trading environment to train on.
    algorithm : str, default='PPO'
        RL algorithm to use: 'PPO' or 'A2C'.
    total_timesteps : int, default=10000
        Total number of timesteps to train for.
    learning_rate : float, default=3e-4
        Learning rate for the optimizer.
    n_steps : int, default=2048
        Number of steps to run per update (PPO only).
    batch_size : int, default=64
        Mini-batch size for updates (PPO only).
    n_epochs : int, default=10
        Number of epochs for PPO updates.
    gamma : float, default=0.99
        Discount factor for future rewards.
    verbose : int, default=1
        Verbosity level (0=none, 1=info, 2=debug).
    tensorboard_log : str, optional
        Path for TensorBoard logs.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    model
        Trained stable-baselines3 model.

    Raises
    ------
    ImportError
        If stable-baselines3 is not installed.
    ValueError
        If algorithm is not 'PPO' or 'A2C'.

    Example
    -------
    >>> import numpy as np
    >>> prices = np.random.randn(1000).cumsum() + 100
    >>> features = np.random.randn(1000, 5)
    >>> env = TradingEnv(prices, features)
    >>> model = train_rl_agent(env, total_timesteps=5000)
    >>> obs, _ = env.reset()
    >>> action, _ = model.predict(obs)
    """
    try:
        from stable_baselines3 import A2C, PPO
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError:
        raise ImportError(
            "stable-baselines3 is required for RL training. "
            "Install with: pip install stable-baselines3"
        )

    # Custom callback for logging
    class TrainingCallback(BaseCallback):
        """Callback for logging training progress."""

        def __init__(self, verbose: int = 0):
            super().__init__(verbose)
            self.episode_rewards: list = []
            self.episode_lengths: list = []

        def _on_step(self) -> bool:
            # Log episode statistics when available
            if len(self.model.ep_info_buffer) > 0:
                ep_info = self.model.ep_info_buffer[-1]
                self.episode_rewards.append(ep_info.get("r", 0))
                self.episode_lengths.append(ep_info.get("l", 0))

                if self.verbose > 0 and len(self.episode_rewards) % 10 == 0:
                    avg_reward = np.mean(self.episode_rewards[-10:])
                    logger.info(
                        f"Episode {len(self.episode_rewards)}: "
                        f"Avg Reward = {avg_reward:.4f}"
                    )
            return True

    # Select algorithm
    algorithm = algorithm.upper()
    if algorithm == "PPO":
        model = PPO(
            "MlpPolicy",
            environment,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            verbose=verbose,
            tensorboard_log=tensorboard_log,
            seed=seed,
        )
    elif algorithm == "A2C":
        model = A2C(
            "MlpPolicy",
            environment,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            verbose=verbose,
            tensorboard_log=tensorboard_log,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Use 'PPO' or 'A2C'.")

    logger.info(f"Training {algorithm} agent for {total_timesteps} timesteps...")

    # Train the model
    callback = TrainingCallback(verbose=verbose)
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Log final statistics
    if len(callback.episode_rewards) > 0:
        final_avg_reward = np.mean(callback.episode_rewards[-10:])
        logger.info(f"Training complete. Final avg reward: {final_avg_reward:.4f}")

    return model


def evaluate_agent(
    model: Any,
    environment: TradingEnv,
    n_episodes: int = 10,
) -> Dict[str, float]:
    """Evaluate a trained agent on the environment.

    Parameters
    ----------
    model
        Trained stable-baselines3 model.
    environment : TradingEnv
        Environment to evaluate on.
    n_episodes : int, default=10
        Number of episodes to run for evaluation.

    Returns
    -------
    dict
        Dictionary with evaluation metrics:
        - mean_return: Average total return across episodes
        - std_return: Standard deviation of returns
        - mean_sharpe: Average Sharpe ratio
        - max_drawdown: Maximum drawdown observed
    """
    returns = []
    drawdowns = []

    for _ in range(n_episodes):
        obs, _ = environment.reset()
        done = False
        episode_returns = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = environment.step(action)
            done = terminated or truncated
            episode_returns.append(reward)

        returns.append(info["total_return"])
        drawdowns.append(info["drawdown"])

    returns = np.array(returns)
    drawdowns = np.array(drawdowns)

    # Calculate Sharpe ratio (assuming ~252 trading days)
    if returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
    else:
        sharpe = 0.0

    return {
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "mean_sharpe": float(sharpe),
        "max_drawdown": float(drawdowns.max()),
    }
