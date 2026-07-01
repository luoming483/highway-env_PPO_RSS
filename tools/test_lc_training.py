"""Quick validation: does PPO learn lane-changing with revised reward + relaxed RSS?

Test: train PPO WITHOUT RSS, evaluate WITH RSS. Let PPO explore LC freely.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO

from config import ENV_CONFIG, PPO_PARAMS, RSS_CONFIG
from rss import RSSConfig, RSSSafetyWrapper
from ppo.train import run_training

SEED = 42
QUICK_STEPS = 50_000


def evaluate_lc(model_path: str, n_episodes: int = 10):
    """Evaluate trained model for lane-change behavior under strict RSS."""
    rss_cfg = RSSConfig(**RSS_CONFIG)
    base = gym.make("highway-fast-v0", config=ENV_CONFIG)
    env = FlattenObservation(RSSSafetyWrapper(base, rss_config=rss_cfg))

    total_lc = 0
    crashes = 0
    actions = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    speeds = []

    ppo = PPO.load(model_path, device="cpu")

    for ep in range(n_episodes):
        obs, _ = env.reset()
        for _ in range(400):
            action, _ = ppo.predict(obs, deterministic=True)
            if isinstance(action, np.ndarray):
                action = int(action.item())
            else:
                action = int(action)
            obs, reward, terminated, truncated, info = env.step(action)
            actions[action] = actions.get(action, 0) + 1
            speeds.append(float(env.unwrapped.vehicle.speed))
            if action in (0, 2):
                total_lc += 1
            if info.get("crashed", False):
                crashes += 1
            if terminated or truncated:
                break

    print(f"\n{'='*60}")
    print(f"Evaluation over {n_episodes} episodes (WITH strict RSS):")
    print(f"  Avg speed:    {np.mean(speeds):.1f} m/s")
    print(f"  Total LC:     {total_lc} ({total_lc/n_episodes:.1f}/ep)")
    print(f"  Crashes:      {crashes}/{n_episodes}")
    print(f"  Action dist:  LEFT={actions[0]} IDLE={actions[1]} RIGHT={actions[2]} "
          f"FASTER={actions[3]} SLOWER={actions[4]}")
    lc_pct = (actions[0] + actions[2]) / max(sum(actions.values()), 1) * 100
    print(f"  LC action %:  {lc_pct:.1f}%")
    return total_lc > 0, total_lc


def main():
    print(f"Training PPO WITHOUT RSS (seed={SEED}, timesteps={QUICK_STEPS})")
    print(f"  Strategy: Free exploration (no RSS during training)")
    print(f"  lane_change_reward: {ENV_CONFIG['lane_change_reward']}")
    print(f"  ent_coef:           {PPO_PARAMS['ent_coef']}")
    print(f"  high_speed_reward:  {ENV_CONFIG['high_speed_reward']}")

    metrics = run_training(
        exp_name="test_lc",
        use_rss=False,  # NO RSS during training!
        use_curriculum=False,
        seed=SEED,
        total_timesteps=QUICK_STEPS,
        device="cpu",
        verbose=0,
    )

    model_path = metrics.get("model_path", "")
    print(f"\nModel saved: {model_path}")

    if model_path and Path(model_path).exists():
        has_lc, lc_count = evaluate_lc(model_path)
        if has_lc:
            print(f"\nSUCCESS: PPO learned to lane-change! ({lc_count} total LCs)")
        else:
            print(f"\nFAILED: Still 0 LCs after {QUICK_STEPS} steps without RSS.")
    else:
        print("Model file not found, skipping evaluation.")


if __name__ == "__main__":
    main()
