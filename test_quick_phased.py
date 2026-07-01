"""Minimal test: 5k steps with phase-based approach to validate code path."""

from copy import deepcopy

import config
from config import ENV_CONFIG, RSS_CONFIG, TRAIN_RSS_OVERRIDES
from rss import RSSConfig
from train import run_training

SEED = 42


def main():
    train_rss_params = dict(RSS_CONFIG)
    train_rss_params.update(TRAIN_RSS_OVERRIDES)
    relaxed_rss = RSSConfig(**train_rss_params)

    target_cfg = deepcopy(ENV_CONFIG)
    custom_phase_plan = [
        {"name": "phase1_no_rss", "timesteps": 2500, "env_config": target_cfg},
        {"name": "phase2_relaxed_rss", "timesteps": 2500, "env_config": target_cfg},
    ]

    rss_cfg_per_phase = [None, relaxed_rss]

    print("Quick validation: 5k steps total")
    metrics = run_training(
        exp_name="test_quick_phased",
        use_rss=True,
        use_curriculum=False,
        seed=SEED,
        total_timesteps=5000,
        device="cpu",
        verbose=0,
        rss_overrides={},
        train_rss_overrides=TRAIN_RSS_OVERRIDES,
        use_blocked_penalty=True,
        use_force_explore=False,
        rss_cfg_per_phase=rss_cfg_per_phase,
        custom_phase_plan=custom_phase_plan,
        n_envs=1,
    )
    print(f"Model path: {metrics.get('model_path', 'N/A')}")
    print("Code path OK — no hang")


if __name__ == "__main__":
    main()
