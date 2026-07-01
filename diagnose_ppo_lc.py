"""Diagnose: why new PPO model shows 0 LCs in compare_experts but 33 LCs in phased test."""

import gymnasium as gym
import highway_env
import numpy as np
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO

from config import ENV_CONFIG, RSS_CONFIG
from rss import RSSConfig, RSSSafetyWrapper

MODEL_PATH = "results/models/test_lc_phased_v3_seed42/final_model.zip"


def eval_env(env, ppo, n_episodes=10, max_steps=400):
    actions = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    speeds = []
    crashes = 0
    total_steps = 0
    lc_attempted = 0  # before RSS override
    lc_blocked = 0

    for _ in range(n_episodes):
        obs, _ = env.reset()
        for __ in range(max_steps):
            action, _ = ppo.predict(obs, deterministic=True)
            if isinstance(action, np.ndarray):
                action = int(action.item())
            else:
                action = int(action)

            # Check RSS block: if agent wants LC, does RSS stop it?
            if action in (0, 2):
                lc_attempted += 1

            obs, reward, terminated, truncated, info = env.step(action)
            actions[action] += 1
            speeds.append(float(env.unwrapped.vehicle.speed))

            # If agent tried LC but RSS overrode it
            if action in (0, 2) and info.get("rss_intervened", False):
                lc_blocked += 1

            if info.get("crashed", False):
                crashes += 1
            total_steps += 1
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
        "steps": total_steps,
        "lc_attempted": lc_attempted,
        "lc_blocked": lc_blocked,
    }


def main():
    ppo = PPO.load(MODEL_PATH, device="cpu")
    rss_cfg = RSSConfig(**RSS_CONFIG)

    # ---- Test 1: ENV_CONFIG (same as phased test) ----
    print("=" * 60)
    print("TEST 1: ENV_CONFIG (phased test eval setup)")
    print(f"  vehicles={ENV_CONFIG['vehicles_count']}, density={ENV_CONFIG['vehicles_density']}")
    print(f"  action: {ENV_CONFIG.get('action', 'default')}")
    print("=" * 60)
    base1 = gym.make("highway-fast-v0", config=ENV_CONFIG)
    env1 = FlattenObservation(RSSSafetyWrapper(base1, rss_config=rss_cfg))
    r1 = eval_env(env1, ppo)
    env1.close()
    print(f"  LC={r1['lc_total']} ({r1['lc_pct']:.1f}%) L={r1['left']} R={r1['right']} "
          f"FASTER={r1['faster']} SLOWER={r1['slower']} IDLE={r1['idle']}")
    print(f"  speed={r1['speed']:.1f} m/s crashes={r1['crashes']}")
    print(f"  LC attempted={r1['lc_attempted']} blocked={r1['lc_blocked']}")

    # ---- Test 2: compare_experts PPO env (with target_speeds) ----
    print(f"\n{'='*60}")
    print("TEST 2: compare_experts PPO env (with target_speeds)")
    ppo_config = dict(ENV_CONFIG)
    ppo_config["action"] = {
        "type": "DiscreteMetaAction",
        "target_speeds": [0, 5, 10, 15, 20, 25, 30],
    }
    print(f"  vehicles={ppo_config['vehicles_count']}, density={ppo_config['vehicles_density']}")
    print(f"  action: {ppo_config['action']}")
    print("=" * 60)
    base2 = gym.make("highway-fast-v0", config=ppo_config)
    env2 = FlattenObservation(RSSSafetyWrapper(base2, rss_config=rss_cfg))
    r2 = eval_env(env2, ppo)
    env2.close()
    print(f"  LC={r2['lc_total']} ({r2['lc_pct']:.1f}%) L={r2['left']} R={r2['right']} "
          f"FASTER={r2['faster']} SLOWER={r2['slower']} IDLE={r2['idle']}")
    print(f"  speed={r2['speed']:.1f} m/s crashes={r2['crashes']}")
    print(f"  LC attempted={r2['lc_attempted']} blocked={r2['lc_blocked']}")

    # ---- Test 3: IDM env (compare_experts style, different densities) ----
    for density_name, density_cfg in [
        ("sparse", {"vehicles_count": 10, "vehicles_density": 0.8}),
        ("medium", {"vehicles_count": 20, "vehicles_density": 1.5}),
        ("dense", {"vehicles_count": 30, "vehicles_density": 2.0}),
    ]:
        print(f"\n{'='*60}")
        print(f"TEST 3: compare_experts env density={density_name} ({density_cfg})")
        test_config = {
            "observation": {
                "type": "Kinematics",
                "vehicles_count": density_cfg["vehicles_count"],
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "absolute": False,
            },
            "action": {"type": "DiscreteMetaAction", "target_speeds": [0, 5, 10, 15, 20, 25, 30]},
            "lanes_count": 4,
            "vehicles_count": density_cfg["vehicles_count"],
            "vehicles_density": density_cfg["vehicles_density"],
            "duration": 50,
            "simulation_frequency": 8,
            "policy_frequency": 4,
            "collision_reward": -5.0,
            "normalize_reward": True,
            "offroad_terminal": True,
        }
        print(f"  vehicles={test_config['vehicles_count']}, density={test_config['vehicles_density']}")
        print("=" * 60)
        # PPO needs 20 vehicles observation space — this will WARN/POTENTIALLY fail
        try:
            base3 = gym.make("highway-fast-v0", config=test_config)
            env3 = FlattenObservation(RSSSafetyWrapper(base3, rss_config=rss_cfg))
            r3 = eval_env(env3, ppo, n_episodes=4, max_steps=200)
            env3.close()
            print(f"  LC={r3['lc_total']} ({r3['lc_pct']:.1f}%) L={r3['left']} R={r3['right']} "
                  f"FASTER={r3['faster']} SLOWER={r3['slower']} IDLE={r3['idle']}")
            print(f"  speed={r3['speed']:.1f} m/s crashes={r3['crashes']}")
            print(f"  LC attempted={r3['lc_attempted']} blocked={r3['lc_blocked']}")
        except Exception as e:
            print(f"  ERROR: {e}")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("CONCLUSION")
    print(f"{'='*60}")
    print(f"  Test 1 (ENV_CONFIG):      LC={r1['lc_total']}, FASTER={r1['faster']}")
    print(f"  Test 2 (+target_speeds):   LC={r2['lc_total']}, FASTER={r2['faster']}")
    if r1['lc_total'] > 0 and r2['lc_total'] == 0:
        print(f"  >> target_speeds overrides causing 0 LCs!")
    elif r1['lc_total'] == 0:
        print(f"  >> Model no longer LCs even in original env — policy may have changed?")
    else:
        print(f"  >> Both envs show LCs — issue is with compare_experts step limit or seeds")


if __name__ == "__main__":
    main()
