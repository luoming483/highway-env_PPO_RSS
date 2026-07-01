"""Sweep lane_change_reward to find optimal balance: speed vs lane-change frequency."""

import sys
from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO

from config import ENV_CONFIG, RSS_CONFIG, PPO_PARAMS
from rss import RSSConfig, RSSSafetyWrapper
from train import run_training

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
        cfg = dict(ENV_CONFIG)
        cfg["lane_change_reward"] = lc_reward

        print(f"\n{'='*60}")
        print(f"lane_change_reward = {lc_reward}")
        print(f"{'='*60}")

        # Monkey-patch ENV_CONFIG for this run (train.py reads from config)
        import config
        config.ENV_CONFIG["lane_change_reward"] = lc_reward

        metrics = run_training(
            exp_name=f"sweep_lc_{lc_reward}",
            use_rss=False,
            use_curriculum=False,
            seed=SEED,
            total_timesteps=STEPS,
            device="cpu",
            verbose=0,
        )

        model_path = metrics.get("model_path", "")
        if model_path and Path(model_path).exists():
            r = evaluate(model_path)
            r["lc_reward"] = lc_reward
            results.append(r)
            print(f"  LC={r['lc_total']} ({r['lc_pct']:.1f}%) "
                  f"L={r['left']} R={r['right']} "
                  f"FASTER={r['faster']} SLOWER={r['slower']} "
                  f"speed={r['speed']:.1f} m/s crashes={r['crashes']}")
        else:
            print(f"  Model not found: {model_path}")

    print(f"\n{'='*80}")
    print("SWEEP SUMMARY")
    print(f"{'='*80}")
    print(f"{'lc_reward':>10} {'LC/ep':>8} {'LC%':>7} {'Speed':>7} {'L':>5} {'R':>5} "
          f"{'FASTER':>7} {'SLOWER':>7} {'Crashes':>7}")
    print("-" * 70)
    for r in results:
        ep = r["steps"] // 10  # approximate episodes
        print(f"{r['lc_reward']:>10.1f} {r['lc_total']/ep:>8.1f} {r['lc_pct']:>6.1f}% "
              f"{r['speed']:>6.1f} {r['left']:>5} {r['right']:>5} "
              f"{r['faster']:>7} {r['slower']:>7} {r['crashes']:>7}")


if __name__ == "__main__":
    main()
