"""Quick test: Phase 1 alone (no RSS) — does agent learn LCs with current config?"""

from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO

import config
from config import ENV_CONFIG, RSS_CONFIG
from rss import RSSConfig, RSSSafetyWrapper
from ppo.train import run_training

SEED = 42
STEPS = 50_000


def evaluate(model_path, n_episodes=10):
    # Eval WITHOUT RSS to see raw policy behavior
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
    orig_lc = config.ENV_CONFIG.get("lane_change_reward", 1.0)
    orig_norm = config.ENV_CONFIG.get("normalize_reward", True)

    # Test different lc_reward values WITHOUT RSS, normalize_reward=True (default)
    for lc_reward in [1.0, 3.0, 5.0]:
        config.ENV_CONFIG["lane_change_reward"] = lc_reward
        config.ENV_CONFIG["normalize_reward"] = True

        print(f"\n{'='*60}")
        print(f"Phase 1 only: NO RSS, lc_reward={lc_reward}, normalize_reward=True")
        print(f"{'='*60}")

        metrics = run_training(
            exp_name=f"test_p1_lc{lc_reward}",
            use_rss=False,
            use_curriculum=False,
            seed=SEED,
            total_timesteps=STEPS,
            device="cpu",
            verbose=0,
            use_blocked_penalty=True,
            use_force_explore=True,
        )

        model_path = metrics.get("model_path", "")
        if model_path and Path(model_path).exists():
            r = evaluate(model_path)
            print(f"  LC={r['lc_total']} ({r['lc_pct']:.1f}%) "
                  f"L={r['left']} R={r['right']} "
                  f"FASTER={r['faster']} SLOWER={r['slower']} IDLE={r['idle']} "
                  f"speed={r['speed']:.1f} m/s crashes={r['crashes']}")

    config.ENV_CONFIG["lane_change_reward"] = orig_lc
    config.ENV_CONFIG["normalize_reward"] = orig_norm


if __name__ == "__main__":
    main()
