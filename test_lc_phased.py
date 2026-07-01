"""Phase-based RSS curriculum training — NO ForceExplore (it backfires).

Phase 1: No RSS + BlockedBonus → Agent discovers LCs naturally via reward shaping
Phase 2: Relaxed RSS + BlockedBonus → Agent learns FASTER is safe under RSS
"""

from copy import deepcopy
from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO

import config
from config import ENV_CONFIG, RSS_CONFIG, TRAIN_RSS_OVERRIDES
from rss import RSSConfig, RSSSafetyWrapper
from train import run_training

SEED = 42
PHASE1_STEPS = 60_000
PHASE2_STEPS = 140_000


def evaluate(model_path, n_episodes=10, use_rss=True):
    if use_rss:
        rss_cfg = RSSConfig(**RSS_CONFIG)
        base = gym.make("highway-fast-v0", config=ENV_CONFIG)
        env = FlattenObservation(RSSSafetyWrapper(base, rss_config=rss_cfg))
    else:
        base = gym.make("highway-fast-v0", config=ENV_CONFIG)
        env = FlattenObservation(base)
    ppo = PPO.load(model_path, device="cpu")

    actions = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    speeds = []
    crashes = 0
    steps = 0

    for _ in range(n_episodes):
        obs, _ = env.reset()
        for __ in range(400):
            action, _ = ppo.predict(obs, deterministic=True)
            if isinstance(action, np.ndarray):
                action = int(action.item())
            else:
                action = int(action)
            obs, reward, terminated, truncated, info = env.step(action)
            actions[action] += 1
            speeds.append(float(env.unwrapped.vehicle.speed))
            if info.get("crashed", False):
                crashes += 1
            steps += 1
            if terminated or truncated:
                break

    total = max(sum(actions.values()), 1)
    lc_total = actions[0] + actions[2]
    return {
        "lc_total": lc_total,
        "lc_pct": lc_total / total * 100,
        "speed": float(np.mean(speeds)),
        "crashes": crashes,
        "left": actions[0],
        "right": actions[2],
        "faster": actions[3],
        "slower": actions[4],
        "idle": actions[1],
        "steps": steps,
    }


def main():
    train_rss_params = dict(RSS_CONFIG)
    train_rss_params.update(TRAIN_RSS_OVERRIDES)
    relaxed_rss = RSSConfig(**train_rss_params)

    target_cfg = deepcopy(ENV_CONFIG)
    custom_phase_plan = [
        {"name": "phase1_no_rss", "timesteps": PHASE1_STEPS, "env_config": target_cfg},
        {"name": "phase2_relaxed_rss", "timesteps": PHASE2_STEPS, "env_config": target_cfg},
    ]

    rss_cfg_per_phase = [None, relaxed_rss]

    orig_lc = config.ENV_CONFIG.get("lane_change_reward", 1.0)
    # Use moderate lc_reward from earlier sweep success range (0.3-0.9 gave 5-14% LCs)
    config.ENV_CONFIG["lane_change_reward"] = 1.0

    print("=" * 70)
    print("PHASE-BASED RSS CURRICULUM (NO ForceExplore)")
    print(f"  Phase 1 ({PHASE1_STEPS} steps): NO RSS + BlockedBonus")
    print(f"    lc_reward={config.ENV_CONFIG['lane_change_reward']}, high_speed={config.ENV_CONFIG['high_speed_reward']}")
    print(f"    Agent discovers LCs naturally via reward shaping")
    print(f"  Phase 2 ({PHASE2_STEPS} steps): Relaxed RSS + BlockedBonus")
    print(f"    RSS: min_dist={TRAIN_RSS_OVERRIDES['min_distance']}m, resp={TRAIN_RSS_OVERRIDES['response_time']}s")
    print(f"    Agent learns FASTER/IDLE are safe under RSS")
    print("=" * 70)

    metrics = run_training(
        exp_name="test_lc_phased_v2",
        use_rss=True,
        use_curriculum=False,
        seed=SEED,
        total_timesteps=PHASE1_STEPS + PHASE2_STEPS,
        device="cpu",
        verbose=0,
        rss_overrides={},
        train_rss_overrides=TRAIN_RSS_OVERRIDES,
        use_blocked_penalty=True,
        use_force_explore=False,
        rss_cfg_per_phase=rss_cfg_per_phase,
        custom_phase_plan=custom_phase_plan,
    )

    config.ENV_CONFIG["lane_change_reward"] = orig_lc

    model_path = metrics.get("model_path", "")
    if model_path and Path(model_path).exists():
        print(f"\n{'='*60}")
        print("EVALUATION (NO RSS) — raw policy behavior")
        print(f"{'='*60}")
        r = evaluate(model_path, use_rss=False)
        print(f"  LC={r['lc_total']} ({r['lc_pct']:.1f}%) "
              f"L={r['left']} R={r['right']} "
              f"FASTER={r['faster']} SLOWER={r['slower']} IDLE={r['idle']}")
        print(f"  speed={r['speed']:.1f} m/s  crashes={r['crashes']}")

        print(f"\n{'='*60}")
        print("EVALUATION (strict RSS) — safety-filtered behavior")
        print(f"{'='*60}")
        r_rss = evaluate(model_path, use_rss=True)
        print(f"  LC={r_rss['lc_total']} ({r_rss['lc_pct']:.1f}%) "
              f"L={r_rss['left']} R={r_rss['right']} "
              f"FASTER={r_rss['faster']} SLOWER={r_rss['slower']} IDLE={r_rss['idle']}")
        print(f"  speed={r_rss['speed']:.1f} m/s  crashes={r_rss['crashes']}")

        if r_rss['lc_total'] > 0 and r_rss['faster'] > 0:
            print(f"\n  SUCCESS: Agent both lane-changes AND uses FASTER under RSS!")
        elif r_rss['lc_total'] > 0:
            print(f"\n  PARTIAL: Agent lane-changes under RSS but no FASTER")
        elif r_rss['faster'] > 0:
            print(f"\n  PARTIAL: Agent uses FASTER under RSS but no LCs")
        else:
            print(f"\n  No LCs or FASTER under RSS")
    else:
        print(f"  Model not found: {model_path}")


if __name__ == "__main__":
    main()
