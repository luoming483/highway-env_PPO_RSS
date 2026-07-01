"""Sweep lane_change_reward with RELAXED RSS during training (not no RSS)."""

import sys
from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO

import config
from config import ENV_CONFIG, RSS_CONFIG, TRAIN_RSS_OVERRIDES
from rss import RSSConfig, RSSSafetyWrapper
from ppo.train import run_training

SEED = 42
STEPS = 50_000
REWARD_VALUES = [0.3, 0.5, 0.7, 0.9]


def evaluate(model_path, n_episodes=10):
    rss_cfg = RSSConfig(**RSS_CONFIG)
    base = gym.make("highway-fast-v0", config=ENV_CONFIG)
    env = FlattenObservation(RSSSafetyWrapper(base, rss_config=rss_cfg))
    ppo = PPO.load(model_path, device="cpu")

    actions = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    speeds = []
    crashes = 0
    steps = 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        for _ in range(400):
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
    results = []

    for lc_reward in REWARD_VALUES:
        config.ENV_CONFIG["lane_change_reward"] = lc_reward

        print(f"\n{'='*60}")
        print(f"lane_change_reward = {lc_reward} (WITH relaxed RSS training)")
        print(f"  RSS train: min_dist={TRAIN_RSS_OVERRIDES['min_distance']}m, "
              f"resp_time={TRAIN_RSS_OVERRIDES['response_time']}s, "
              f"side_gap={TRAIN_RSS_OVERRIDES['lane_change_side_gap']}m")
        print(f"{'='*60}")

        metrics = run_training(
            exp_name=f"sweep_lc_relaxed_{lc_reward}",
            use_rss=True,
            use_curriculum=False,
            seed=SEED,
            total_timesteps=STEPS,
            device="cpu",
            verbose=0,
            rss_overrides={},
            train_rss_overrides=TRAIN_RSS_OVERRIDES,
        )

        model_path = metrics.get("model_path", "")
        if model_path and Path(model_path).exists():
            r = evaluate(model_path)
            r["lc_reward"] = lc_reward
            results.append(r)
            print(f"  LC={r['lc_total']} ({r['lc_pct']:.1f}%) "
                  f"L={r['left']} R={r['right']} "
                  f"FASTER={r['faster']} SLOWER={r['slower']} IDLE={r['idle']} "
                  f"speed={r['speed']:.1f} m/s crashes={r['crashes']}")
        else:
            print(f"  Model not found: {model_path}")

    print(f"\n{'='*85}")
    print("SWEEP SUMMARY (with relaxed RSS training)")
    print(f"{'='*85}")
    print(f"{'lc_reward':>10} {'LC/ep':>8} {'LC%':>7} {'Speed':>7} {'L':>5} {'R':>5} "
          f"{'FASTER':>7} {'SLOWER':>7} {'IDLE':>5} {'Crashes':>7}")
    print("-" * 78)
    best = None
    for r in results:
        ep = max(r["steps"] // 10, 1)
        print(f"{r['lc_reward']:>10.1f} {r['lc_total']/ep:>8.1f} {r['lc_pct']:>6.1f}% "
              f"{r['speed']:>6.1f} {r['left']:>5} {r['right']:>5} "
              f"{r['faster']:>7} {r['slower']:>7} {r['idle']:>5} {r['crashes']:>7}")
        # Score: speed + LC bonus, penalize crashes
        if r['crashes'] == 0 and r['faster'] > 0 and 3 < r['lc_pct'] < 20:
            score = r['speed'] * 10 + r['lc_pct']
            if best is None or score > best[1]:
                best = (r, score)

    if best:
        r, _ = best
        print(f"\nBEST: lc_reward={r['lc_reward']} speed={r['speed']:.1f} "
              f"LC={r['lc_pct']:.1f}% FASTER={r['faster']}")
    else:
        print("\nNo clear best — all degenerate or too conservative. Need wider sweep.")


if __name__ == "__main__":
    main()
