"""Single training function for PPO + RSS experiments. Called by experiment.py."""

import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

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
    CURRICULUM_PHASES,
    ENV_CONFIG,
    ENV_ID,
    EVAL_FREQ_STEPS,
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
from rss_safety import RSSConfig, RSSSafetyWrapper
from metrics import MetricsCollector


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)


def _merge_env_config(overrides: Optional[Dict] = None) -> Dict:
    cfg = deepcopy(ENV_CONFIG)
    if overrides:
        cfg.update(overrides)
    return cfg


def _make_env_wrapper(rss_config: Optional[RSSConfig]):
    if rss_config is None:
        return FlattenObservation

    def _wrap(env):
        return FlattenObservation(RSSSafetyWrapper(env, rss_config=rss_config))

    return _wrap


def _build_vec_env(
    n_envs: int,
    seed: int,
    env_config: Dict,
    rss_config: Optional[RSSConfig] = None,
    log_dir: Path = LOG_DIR,
):
    monitor_dir = Path(log_dir) / f"monitor_seed{seed}"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    return make_vec_env(
        ENV_ID,
        n_envs=n_envs,
        seed=seed,
        env_kwargs={"config": env_config, "render_mode": None},
        wrapper_class=_make_env_wrapper(rss_config),
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
    model_dir: Path = MODEL_DIR,
    log_dir: Path = LOG_DIR,
) -> Dict:
    """Train PPO agent. Returns structured metrics dict."""
    _set_seed(seed)

    rss_config = None
    if use_rss:
        rss_params = dict(RSS_CONFIG)
        if rss_overrides:
            rss_params.update(rss_overrides)
        rss_config = RSSConfig(**rss_params)

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
    for phase in phase_plan:
        print(f"  [{phase['name']}] {phase['timesteps']} steps "
              f"(vehicles={phase['env_config']['vehicles_count']}, "
              f"density={phase['env_config']['vehicles_density']})")
        model = _train_phase(model, phase, seed, n_envs, device, rss_config, collector, verbose, log_dir)

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
