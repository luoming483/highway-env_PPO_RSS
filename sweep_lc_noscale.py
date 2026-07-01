"""Sweep WITH relaxed RSS + normalize_reward=FALSE + moderate lane_change_reward."""

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
from train import run_training

SEED = 42
STEPS = 50_000

# Test moderate lane_change_reward values at FULL scale (no normalization)
# At full scale: speed ≈ 0.58/step, LC reward ≈ lc_value (one-time)
# Target: LC should be ~5-10 steps of speed reward → lc_value ≈ 3.0-6.0
REWARD_VALUES = [3.0, 5.0, 7.0, 10.0]


def evaluate(model_path, n_episodes=10):
    rss_cfg = RSSConfig(**RSS_CONFIG)
    # Eval always uses original config (with normalize_reward)
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
    # Save originals to restore later
    orig_norm = config.ENV_CONFIG.get("normalize_reward", True)
    orig_lc = config.ENV_CONFIG.get("lane_change_reward", 1.0)

    for lc_reward in REWARD_VALUES:
        # Use full-scale rewards for training
        config.ENV_CONFIG["normalize_reward"] = False
        config.ENV_CONFIG["lane_change_reward"] = lc_reward

        print(f"\n{'='*60}")
        print(f"lane_change_reward = {lc_reward} (FULL scale, relaxed RSS training)")
        print(f"  high_speed_reward = {config.ENV_CONFIG['high_speed_reward']} (full scale)")
        print(f"  normalize_reward = False")
        print(f"{'='*60}")

        metrics = run_training(
            exp_name=f"sweep_lc_noscale_{lc_reward}",
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

    print(f"\n{'='*90}")
    print("SWEEP SUMMARY (relaxed RSS + full-scale rewards)")
    print(f"{'='*90}")
    print(f"{'lc_reward':>10} {'LC/ep':>8} {'LC%':>7} {'Speed':>7} {'L':>5} {'R':>5} "
          f"{'FASTER':>7} {'SLOWER':>7} {'IDLE':>5} {'Crashes':>7}")
    print("-" * 80)
    for r in results:
        ep = max(r["steps"] // 10, 1)
        print(f"{r['lc_reward']:>10.1f} {r['lc_total']/ep:>8.1f} {r['lc_pct']:>6.1f}% "
              f"{r['speed']:>6.1f} {r['left']:>5} {r['right']:>5} "
              f"{r['faster']:>7} {r['slower']:>7} {r['idle']:>5} {r['crashes']:>7}")

    best = None
    for r in results:
        if r['crashes'] == 0 and r['faster'] > 0 and 2 < r['lc_pct'] < 25:
            score = r['speed'] * 10 + r['lc_pct'] * 0.5
            if best is None or score > best[1]:
                best = (r, score)

    # Restore original config
    config.ENV_CONFIG["normalize_reward"] = orig_norm
    config.ENV_CONFIG["lane_change_reward"] = orig_lc

    if best:
        r, _ = best
        print(f"\nBEST: lc_reward={r['lc_reward']} speed={r['speed']:.1f} "
              f"LC={r['lc_pct']:.1f}% FASTER={r['faster']} "
              f"L={r['left']} R={r['right']}")
    else:
        print("\nNo clear best found.")
        for r in results:
            print(f"  lc_reward={r['lc_reward']}: FASTER={r['faster']} LC%={r['lc_pct']:.1f} "
                  f"speed={r['speed']:.1f} crashes={r['crashes']}")


if __name__ == "__main__":
    main()
