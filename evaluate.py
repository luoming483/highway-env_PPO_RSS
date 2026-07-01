"""Evaluate a trained PPO model, optionally with RSS safety shield."""

import argparse
from pathlib import Path
from typing import Dict, Optional

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO

from config import ENV_CONFIG, ENV_ID, MODEL_DIR, N_EVAL_EPISODES, RSS_CONFIG, SEEDS
from rss import RSSConfig, RSSSafetyWrapper


def evaluate_model(
    model_path: Path,
    n_episodes: int = N_EVAL_EPISODES,
    use_rss: bool = False,
    device: str = "cpu",
    seed: int = SEEDS[0],
) -> Dict:
    rss_config = RSSConfig(**RSS_CONFIG) if use_rss else None

    env = gym.make(ENV_ID, config=ENV_CONFIG, render_mode=None)
    if rss_config is not None:
        env = RSSSafetyWrapper(env, rss_config=rss_config)
    env = FlattenObservation(env)

    resolved = model_path
    if not resolved.exists():
        if resolved.suffix != ".zip" and (resolved.with_suffix(".zip")).exists():
            resolved = resolved.with_suffix(".zip")

    model = PPO.load(str(resolved), device=device)
    print(f"[Loaded] {resolved}")

    rewards_list = []
    collisions = 0
    intervention_steps = 0
    total_steps = 0
    ep_min_ttc_list = []
    ep_min_distance_list = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        truncated = False
        ep_reward = 0.0
        crashed = False
        ep_min_ttc = np.inf
        ep_min_distance = np.inf

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += float(reward)
            crashed = crashed or bool(info.get("crashed", False))
            total_steps += 1
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
        print(f"[Episode {ep}] Reward={ep_reward:.3f}, Crashed={'Yes' if crashed else 'No'}")

    env.close()

    result = {
        "reward_mean": float(np.mean(rewards_list)),
        "reward_std": float(np.std(rewards_list)),
        "collision_rate": collisions / max(n_episodes, 1),
        "intervention_rate": intervention_steps / max(total_steps, 1) if use_rss else 0.0,
        "min_ttc_mean": float(np.nanmean(ep_min_ttc_list)) if ep_min_ttc_list else np.nan,
        "min_distance_mean": float(np.nanmean(ep_min_distance_list)) if ep_min_distance_list else np.nan,
    }

    print(f"\n[Summary] Mean reward: {result['reward_mean']:.3f} +/- {result['reward_std']:.3f}")
    print(f"[Summary] Collision rate: {result['collision_rate']:.2%}")
    if use_rss:
        print(f"[Summary] Intervention rate: {result['intervention_rate']:.2%}")
        print(f"[Summary] Min TTC: {result['min_ttc_mean']:.3f}")
        print(f"[Summary] Min distance: {result['min_distance_mean']:.3f}")

    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO model on highway-env.")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to model zip file.")
    parser.add_argument("--episodes", type=int, default=N_EVAL_EPISODES)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "auto", "cuda"])
    parser.add_argument("--rss", action="store_true", help="Enable RSS safety shield during evaluation.")
    parser.add_argument("--seed", type=int, default=SEEDS[0])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.model_path is None:
        model_path = Path(MODEL_DIR) / "ppo_highway_sb3_final"
    else:
        model_path = Path(args.model_path)
    evaluate_model(
        model_path=model_path,
        n_episodes=args.episodes,
        use_rss=args.rss,
        device=args.device,
        seed=args.seed,
    )
