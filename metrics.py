"""Metrics collection for PPO + RSS training runs."""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


@dataclass
class TrainingMetrics:
    """All metrics collected during a single training run."""

    reward_curve_x: List[int] = field(default_factory=list)
    reward_curve_y: List[float] = field(default_factory=list)
    loss_curve_x: List[int] = field(default_factory=list)
    loss_curve_y: List[float] = field(default_factory=list)
    collision_curve_x: List[int] = field(default_factory=list)
    collision_curve_y: List[float] = field(default_factory=list)

    eval_timesteps: List[int] = field(default_factory=list)
    eval_reward_mean: List[float] = field(default_factory=list)
    eval_reward_std: List[float] = field(default_factory=list)
    eval_collision_rate: List[float] = field(default_factory=list)
    eval_intervention_rate: List[float] = field(default_factory=list)
    eval_min_ttc: List[float] = field(default_factory=list)
    eval_min_distance: List[float] = field(default_factory=list)

    final_reward_mean: float = 0.0
    final_reward_std: float = 0.0
    final_collision_rate: float = 0.0
    final_intervention_rate: float = 0.0
    final_min_ttc: float = float("inf")
    final_min_distance: float = float("inf")

    wall_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reward_curve_x": self.reward_curve_x,
            "reward_curve_y": self.reward_curve_y,
            "loss_curve_x": self.loss_curve_x,
            "loss_curve_y": self.loss_curve_y,
            "collision_curve_x": self.collision_curve_x,
            "collision_curve_y": self.collision_curve_y,
            "eval_timesteps": self.eval_timesteps,
            "eval_reward_mean": self.eval_reward_mean,
            "eval_reward_std": self.eval_reward_std,
            "eval_collision_rate": self.eval_collision_rate,
            "eval_intervention_rate": self.eval_intervention_rate,
            "eval_min_ttc": self.eval_min_ttc,
            "eval_min_distance": self.eval_min_distance,
            "final_reward_mean": self.final_reward_mean,
            "final_reward_std": self.final_reward_std,
            "final_collision_rate": self.final_collision_rate,
            "final_intervention_rate": self.final_intervention_rate,
            "final_min_ttc": self.final_min_ttc,
            "final_min_distance": self.final_min_distance,
            "wall_time_seconds": self.wall_time_seconds,
        }


class MetricsCollector(BaseCallback):
    """Single callback collecting all training and evaluation metrics."""

    def __init__(
        self,
        eval_env_builder,
        reward_window: int = 10,
        collision_window: int = 20,
        n_eval_episodes: int = 6,
        eval_freq_steps: int = 6_000,
        eval_seed_base: int = 0,
        rss_config=None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._eval_env_builder = eval_env_builder
        self.reward_window = max(1, int(reward_window))
        self.collision_window = max(1, int(collision_window))
        self.n_eval_episodes = max(1, int(n_eval_episodes))
        self.eval_freq_steps = max(1, int(eval_freq_steps))
        self.eval_seed_base = int(eval_seed_base)
        self.rss_config = rss_config

        self.metrics = TrainingMetrics()

        # Internal state
        self._episode_rewards: List[float] = []
        self._episode_collisions: List[int] = []
        self._running_returns: Optional[np.ndarray] = None
        self._last_loss_step: Optional[int] = None
        self._last_eval_step: int = 0
        self._eval_env = None
        self._t_start: Optional[float] = None
        self._n_envs: int = 1

    def _on_training_start(self) -> None:
        self._t_start = time.time()
        self._n_envs = getattr(self.model, "n_envs", 1)
        self._running_returns = np.zeros(self._n_envs, dtype=float)
        self._last_loss_step = None
        self._last_eval_step = int(self.num_timesteps)
        if self._eval_env is None:
            self._eval_env = self._eval_env_builder()

    def _capture_loss(self) -> None:
        logger_values = getattr(self.model.logger, "name_to_value", {})
        loss = logger_values.get("train/loss", None)
        if loss is None:
            return
        step = int(self.num_timesteps)
        if self._last_loss_step == step:
            return
        self.metrics.loss_curve_x.append(step)
        self.metrics.loss_curve_y.append(float(loss))
        self._last_loss_step = step

    def _on_rollout_start(self) -> None:
        self._capture_loss()

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", None)
        dones = self.locals.get("dones", None)
        infos = self.locals.get("infos", None)
        if rewards is None or dones is None:
            return True

        rewards_arr = np.asarray(rewards, dtype=float).flatten()
        dones_arr = np.asarray(dones, dtype=bool).flatten()
        if infos is None:
            infos = [{} for _ in range(len(rewards_arr))]

        if self._running_returns is None or len(self._running_returns) != len(rewards_arr):
            self._running_returns = np.zeros(len(rewards_arr), dtype=float)

        self._running_returns += rewards_arr

        for i, done in enumerate(dones_arr):
            if not done:
                continue
            ep_reward = float(self._running_returns[i])
            self._running_returns[i] = 0.0
            crashed = bool(infos[i].get("crashed", False))

            self._episode_rewards.append(ep_reward)
            self._episode_collisions.append(int(crashed))

            rw = self._episode_rewards[-self.reward_window:]
            cw = self._episode_collisions[-self.collision_window:]
            ep_idx = len(self._episode_rewards)
            self.metrics.reward_curve_x.append(ep_idx)
            self.metrics.reward_curve_y.append(float(np.mean(rw)))
            self.metrics.collision_curve_x.append(ep_idx)
            self.metrics.collision_curve_y.append(float(np.mean(cw)))

        # Periodic evaluation
        if (self.num_timesteps - self._last_eval_step) >= self.eval_freq_steps:
            self._run_periodic_eval(int(self.num_timesteps))
            self._last_eval_step = int(self.num_timesteps)

        return True

    def _run_periodic_eval(self, step: int) -> None:
        rewards_list: List[float] = []
        collisions = 0
        intervention_steps = 0
        total_steps = 0
        ep_min_ttc_list: List[float] = []
        ep_min_distance_list: List[float] = []

        max_steps_per_ep = 500
        for ep in range(self.n_eval_episodes):
            obs, _ = self._eval_env.reset(seed=self.eval_seed_base + step + ep)
            done = False
            truncated = False
            ep_reward = 0.0
            crashed = False
            ep_min_ttc = np.inf
            ep_min_distance = np.inf
            ep_step = 0

            while not (done or truncated) and ep_step < max_steps_per_ep:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self._eval_env.step(action)
                ep_reward += float(reward)
                crashed = crashed or bool(info.get("crashed", False))
                total_steps += 1
                ep_step += 1
                intervention_steps += int(bool(info.get("rss_intervened", False)))
                ttc = float(info.get("rss_min_ttc", np.inf))
                min_dist = float(info.get("rss_min_distance", np.inf))
                if np.isfinite(ttc):
                    ep_min_ttc = min(ep_min_ttc, ttc)
                if np.isfinite(min_dist):
                    ep_min_distance = min(ep_min_distance, min_dist)

            rewards_list.append(ep_reward)
            collisions += int(crashed)
            ep_min_ttc_list.append(ep_min_ttc if np.isfinite(ep_min_ttc) else np.nan)
            ep_min_distance_list.append(ep_min_distance if np.isfinite(ep_min_distance) else np.nan)

        valid_ttc = [v for v in ep_min_ttc_list if np.isfinite(v)]
        valid_dist = [v for v in ep_min_distance_list if np.isfinite(v)]

        self.metrics.eval_timesteps.append(step)
        self.metrics.eval_reward_mean.append(float(np.mean(rewards_list)))
        self.metrics.eval_reward_std.append(float(np.std(rewards_list)))
        self.metrics.eval_collision_rate.append(collisions / max(self.n_eval_episodes, 1))
        self.metrics.eval_intervention_rate.append(intervention_steps / max(total_steps, 1))
        self.metrics.eval_min_ttc.append(float(np.mean(valid_ttc)) if valid_ttc else np.nan)
        self.metrics.eval_min_distance.append(float(np.mean(valid_dist)) if valid_dist else np.nan)

    def _on_training_end(self) -> None:
        self._capture_loss()
        if self._t_start is not None:
            self.metrics.wall_time_seconds = time.time() - self._t_start
        # Run final evaluation on target env
        if self._eval_env is not None:
            step = int(self.num_timesteps)
            if not self.metrics.eval_timesteps or self.metrics.eval_timesteps[-1] != step:
                self._run_periodic_eval(step)
            self._eval_env.close()
            self._eval_env = None

    def run_final_evaluation(self, eval_env_builder) -> None:
        """Run a more thorough final evaluation on a fresh env."""
        import gymnasium as gym

        env = eval_env_builder()
        rewards_list: List[float] = []
        collisions = 0
        intervention_steps = 0
        total_steps = 0
        ep_min_ttc_list: List[float] = []
        ep_min_distance_list: List[float] = []
        n_ep = max(20, self.n_eval_episodes)
        max_steps_per_ep = 500

        for ep in range(n_ep):
            obs, _ = env.reset(seed=self.eval_seed_base + 100000 + ep)
            done = False
            truncated = False
            ep_reward = 0.0
            crashed = False
            ep_min_ttc = np.inf
            ep_min_distance = np.inf
            ep_step = 0

            while not (done or truncated) and ep_step < max_steps_per_ep:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                ep_reward += float(reward)
                crashed = crashed or bool(info.get("crashed", False))
                total_steps += 1
                ep_step += 1
                intervention_steps += int(bool(info.get("rss_intervened", False)))
                ttc = float(info.get("rss_min_ttc", np.inf))
                min_dist = float(info.get("rss_min_distance", np.inf))
                if np.isfinite(ttc):
                    ep_min_ttc = min(ep_min_ttc, ttc)
                if np.isfinite(min_dist):
                    ep_min_distance = min(ep_min_distance, min_dist)

            rewards_list.append(ep_reward)
            collisions += int(crashed)
            ep_min_ttc_list.append(ep_min_ttc if np.isfinite(ep_min_ttc) else np.nan)
            ep_min_distance_list.append(ep_min_distance if np.isfinite(ep_min_distance) else np.nan)

        valid_ttc_final = [v for v in ep_min_ttc_list if np.isfinite(v)]
        valid_dist_final = [v for v in ep_min_distance_list if np.isfinite(v)]

        self.metrics.final_reward_mean = float(np.mean(rewards_list))
        self.metrics.final_reward_std = float(np.std(rewards_list))
        self.metrics.final_collision_rate = collisions / max(n_ep, 1)
        self.metrics.final_intervention_rate = intervention_steps / max(total_steps, 1)
        self.metrics.final_min_ttc = float(np.mean(valid_ttc_final)) if valid_ttc_final else np.nan
        self.metrics.final_min_distance = float(np.mean(valid_dist_final)) if valid_dist_final else np.nan

        env.close()
