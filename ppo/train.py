"""Single training function for PPO + RSS experiments. Called by experiment.py."""

import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
import torch as th
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from config import (
    ACTION_LABELS,
    BLOCKED_GAP_THRESHOLD,
    BLOCKED_PENALTY,
    BLOCKED_SPEED_RATIO,
    CURRICULUM_PHASES,
    ENV_CONFIG,
    ENV_ID,
    EVAL_FREQ_STEPS,
    LC_ATTEMPT_BONUS,
    LOG_DIR,
    MODEL_DIR,
    N_ENVS,
    N_EVAL_EPISODES,
    POLICY_NET_ARCH,
    PPO_PARAMS,
    REWARD_WINDOW,
    RSS_CONFIG,
    TOTAL_TIMESTEPS,
)
from rss import RSSConfig, RSSSafetyWrapper
from scene_utils import check_ego_blocked
from metrics import MetricsCollector


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)


class ForceExploreWrapper(gym.Wrapper):
    """Informed exploration: only force LC when blocked behind a slow vehicle.

    Checks blocked state (same condition as BlockedPenaltyWrapper) and only
    overrides to LEFT/RIGHT when ego is actually stuck. This ensures the
    agent associates LC actions with escaping blocked situations.
    """

    def __init__(self, env, explore_prob=0.50, explore_actions=(0, 2),
                 decay_steps=100_000, gap_threshold=BLOCKED_GAP_THRESHOLD,
                 speed_ratio=BLOCKED_SPEED_RATIO):
        super().__init__(env)
        self._explore_prob = explore_prob
        self._explore_actions = explore_actions
        self._decay_steps = decay_steps
        self._gap_threshold = gap_threshold
        self._speed_ratio = speed_ratio
        self._step_count = 0

    def _check_blocked(self):
        blocked, _, _ = check_ego_blocked(self, self._gap_threshold, self._speed_ratio)
        return blocked

    def step(self, action):
        self._step_count += 1
        current_prob = self._explore_prob * max(0.0, 1.0 - self._step_count / self._decay_steps)

        original_action = action
        if int(action) not in self._explore_actions and np.random.random() < current_prob:
            if self._check_blocked():
                action = int(np.random.choice(self._explore_actions))

        obs, reward, terminated, truncated, info = self.env.step(action)
        if info is None:
            info = {}
        if int(original_action) != int(action):
            info["force_explored"] = True
            info["force_original_action"] = int(original_action)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class BlockedPenaltyWrapper(gym.Wrapper):
    """Penalize staying blocked + reward lane-change attempts when blocked.

    Must be placed ABOVE RSS in the wrapper chain (BlockedPenalty → RSS → BaseEnv)
    so it can see the original action before RSS overrides it.

    Negative reward per step when ego is close behind a slower front vehicle.
    Positive reward for attempting LEFT/RIGHT while blocked.
    """

    def __init__(self, env, penalty=BLOCKED_PENALTY,
                 gap_threshold=BLOCKED_GAP_THRESHOLD,
                 speed_ratio=BLOCKED_SPEED_RATIO,
                 lc_attempt_bonus=LC_ATTEMPT_BONUS):
        super().__init__(env)
        self._penalty = penalty
        self._gap_threshold = gap_threshold
        self._speed_ratio = speed_ratio
        self._lc_bonus = lc_attempt_bonus

    def _check_blocked(self):
        blocked, _, _ = check_ego_blocked(self, self._gap_threshold, self._speed_ratio)
        return blocked

    def step(self, action):
        is_blocked = self._check_blocked()
        bonus = 0.0

        if is_blocked:
            bonus += self._penalty
            if int(action) in (0, 2):  # LEFT or RIGHT
                bonus += self._lc_bonus

        obs, reward, terminated, truncated, info = self.env.step(action)

        if bonus != 0.0:
            reward = float(reward) + bonus
            info["blocked"] = is_blocked
            info["blocked_bonus"] = bonus

        return obs, reward, terminated, truncated, info


def _merge_env_config(overrides: Optional[Dict] = None) -> Dict:
    cfg = deepcopy(ENV_CONFIG)
    if overrides:
        cfg.update(overrides)
    return cfg


def _make_env_wrapper(rss_config: Optional[RSSConfig], use_blocked_penalty: bool = False,
                      use_force_explore: bool = False):
    """Create env wrapper factory.

    Wrapper order (inside-out):
        BaseEnv → BlockedPenalty → RSS → [ForceExplore] → FlattenObservation
    ForceExplore is outside RSS so it can override agent actions before RSS safety check.
    BlockedPenalty is inside RSS so it sees post-safety-check actions.
    """
    if rss_config is None:
        def _wrap(env):
            if use_blocked_penalty:
                env = BlockedPenaltyWrapper(env)
            if use_force_explore:
                env = ForceExploreWrapper(env)
            return FlattenObservation(env)
        return _wrap

    def _wrap(env):
        if use_blocked_penalty:
            env = BlockedPenaltyWrapper(env)
        env = RSSSafetyWrapper(env, rss_config=rss_config)
        if use_force_explore:
            env = ForceExploreWrapper(env)
        return FlattenObservation(env)

    return _wrap


def _build_vec_env(
    n_envs: int,
    seed: int,
    env_config: Dict,
    rss_config: Optional[RSSConfig] = None,
    log_dir: Path = LOG_DIR,
    use_blocked_penalty: bool = False,
    use_force_explore: bool = False,
):
    monitor_dir = Path(log_dir) / f"monitor_seed{seed}"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    return make_vec_env(
        ENV_ID,
        n_envs=n_envs,
        seed=seed,
        env_kwargs={"config": env_config, "render_mode": None},
        wrapper_class=_make_env_wrapper(rss_config,
                                        use_blocked_penalty=use_blocked_penalty,
                                        use_force_explore=use_force_explore),
        monitor_dir=str(monitor_dir),
    )


def _build_single_env(
    env_config: Dict,
    rss_config: Optional[RSSConfig] = None,
    render_mode: Optional[str] = None,
):
    env = gym.make(ENV_ID, config=env_config, render_mode=render_mode)
    if rss_config is not None:
        env = RSSSafetyWrapper(env, rss_config=rss_config)
    env = FlattenObservation(env)
    return env


def _allocate_phase_timesteps(total: int, phases: List[Dict]) -> List[int]:
    raw = [max(1, int(total * float(p.get("ratio", 0.0)))) for p in phases]
    diff = total - sum(raw)
    raw[-1] += diff
    if raw[-1] < 1:
        deficit = 1 - raw[-1]
        raw[-1] = 1
        for i in range(len(raw) - 1):
            available = raw[i] - 1
            if available <= 0:
                continue
            take = min(available, deficit)
            raw[i] -= take
            deficit -= take
            if deficit == 0:
                break
        if deficit > 0:
            raise ValueError("Not enough timesteps to allocate across phases.")
    return raw


def _build_phase_plan(total_timesteps: int, use_curriculum: bool) -> List[Dict]:
    if use_curriculum:
        phases = CURRICULUM_PHASES
    else:
        phases = [{"name": "single_phase", "ratio": 1.0, "overrides": {}}]
    steps = _allocate_phase_timesteps(total_timesteps, phases)
    return [
        {
            "name": p["name"],
            "timesteps": s,
            "env_config": _merge_env_config(p.get("overrides", {})),
        }
        for p, s in zip(phases, steps)
    ]


def _train_phase(
    model,
    phase: Dict,
    seed: int,
    n_envs: int,
    device: str,
    rss_config,
    collector: MetricsCollector,
    verbose: int,
    log_dir: Path,
    use_blocked_penalty: bool = False,
    use_force_explore: bool = False,
):
    phase_name = phase["name"]
    phase_steps = int(phase["timesteps"])
    phase_env_cfg = phase["env_config"]

    train_env = _build_vec_env(
        n_envs=n_envs,
        seed=seed + 100,
        env_config=phase_env_cfg,
        rss_config=rss_config,
        log_dir=log_dir,
        use_blocked_penalty=use_blocked_penalty,
        use_force_explore=use_force_explore,
    )

    if model is None:
        policy_kwargs = {
            "activation_fn": th.nn.Tanh,
            "net_arch": dict(pi=POLICY_NET_ARCH, vf=POLICY_NET_ARCH),
        }
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(log_dir),
            device=device,
            seed=seed,
            verbose=verbose,
            **PPO_PARAMS,
        )
    else:
        model.set_env(train_env)

    model.learn(
        total_timesteps=phase_steps,
        callback=collector,
        progress_bar=False,
        reset_num_timesteps=False,
    )

    train_env.close()
    return model


def run_training(
    exp_name: str,
    use_rss: bool,
    use_curriculum: bool,
    seed: int,
    total_timesteps: int = TOTAL_TIMESTEPS,
    n_envs: int = N_ENVS,
    device: str = "cpu",
    verbose: int = 0,
    rss_overrides: Optional[Dict] = None,
    train_rss_overrides: Optional[Dict] = None,
    use_blocked_penalty: bool = False,
    use_force_explore: bool = False,
    model_dir: Path = MODEL_DIR,
    log_dir: Path = LOG_DIR,
    rss_cfg_per_phase: Optional[List[Optional[RSSConfig]]] = None,
    force_explore_per_phase: Optional[List[bool]] = None,
    custom_phase_plan: Optional[List[Dict]] = None,
) -> Dict:
    """Train PPO agent. Returns structured metrics dict.

    train_rss_overrides: RSS overrides for training only (e.g. relaxed side_gap for LC exploration).
    rss_overrides: RSS overrides for evaluation only.
    use_force_explore: force LEFT/RIGHT actions during early training for LC exploration.
    rss_cfg_per_phase: per-phase RSS config (None = no RSS for that phase). Overrides train_rss_config.
    force_explore_per_phase: per-phase ForceExplore toggle.
    custom_phase_plan: override auto-built phase plan. List of {"name", "timesteps", "env_config"} dicts.
    """
    _set_seed(seed)

    rss_config = None
    train_rss_config = None
    if use_rss:
        rss_params = dict(RSS_CONFIG)
        if rss_overrides:
            rss_params.update(rss_overrides)
        rss_config = RSSConfig(**rss_params)

        train_rss_params = dict(rss_params)
        if train_rss_overrides:
            train_rss_params.update(train_rss_overrides)
        train_rss_config = RSSConfig(**train_rss_params)

    if custom_phase_plan is not None:
        phase_plan = custom_phase_plan
    else:
        phase_plan = _build_phase_plan(total_timesteps, use_curriculum)
    target_env_cfg = _merge_env_config()

    # Create collector with eval env builder pointing to target config
    collector = MetricsCollector(
        eval_env_builder=lambda: _build_single_env(target_env_cfg, rss_config=rss_config),
        reward_window=REWARD_WINDOW,
        n_eval_episodes=N_EVAL_EPISODES,
        eval_freq_steps=EVAL_FREQ_STEPS,
        eval_seed_base=seed + 30000,
        rss_config=rss_config,
        verbose=0,
    )

    model = None
    for i, phase in enumerate(phase_plan):
        # Per-phase RSS config
        if rss_cfg_per_phase is not None and i < len(rss_cfg_per_phase):
            phase_rss = rss_cfg_per_phase[i]
        else:
            phase_rss = train_rss_config

        # Per-phase ForceExplore toggle
        if force_explore_per_phase is not None and i < len(force_explore_per_phase):
            phase_force = force_explore_per_phase[i]
        else:
            phase_force = use_force_explore

        rss_label = "no RSS" if phase_rss is None else "RSS"
        fe_label = "+FE" if phase_force else ""
        print(f"  [{phase['name']}] {phase['timesteps']} steps "
              f"(vehicles={phase['env_config']['vehicles_count']}, "
              f"density={phase['env_config']['vehicles_density']}, "
              f"{rss_label}{fe_label})")
        model = _train_phase(model, phase, seed, n_envs, device,
                            phase_rss, collector, verbose, log_dir,
                            use_blocked_penalty=use_blocked_penalty,
                            use_force_explore=phase_force)

    # Final evaluation
    collector.run_final_evaluation(lambda: _build_single_env(target_env_cfg, rss_config=rss_config))

    # Save model
    save_dir = Path(model_dir) / f"{exp_name}_seed{seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "final_model"
    model.save(str(save_path))

    # Also save the last eval env's evaluate_policy stats for logging
    metrics = collector.metrics.to_dict()
    metrics["model_path"] = str(save_path) + ".zip"

    return metrics
