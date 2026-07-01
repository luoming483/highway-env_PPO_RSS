"""Deep metric comparison: Does lane-change capability translate to real benefits?

New metrics beyond average speed:
  1. Blocked ratio: % of steps stuck behind slower vehicle
  2. Overtake count: successful passes of slower front vehicles
  3. Min speed / speed floor: worst-case speed (lower = more blocked)
  4. Lane utilization entropy: diversity of lane usage
  5. Post-LC speed delta: speed change after lane change
  6. Time-to-destination: steps to cover fixed longitudinal distance

Compares: Stackelberg, PPO+RSS (new), MoE Hybrid
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gymnasium as gym
import highway_env
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO

from config import ENV_CONFIG, RSS_CONFIG
from rss import RSSConfig, RSSSafetyWrapper
from stackelberg.config import GameConfig
from stackelberg.expert import StackelbergExpert
from moe_hybrid import HybridExpert, make_env as moe_make_env

SEEDS = [42, 123, 456, 789]
MAX_STEPS = 300
PPO_MODEL = "results/models/test_lc_phased_v3_seed42/final_model.zip"

# Density configs (use PPO-compatible vehicle counts)
DENSITY_CONFIGS = {
    "sparse": {"vehicles_count": 20, "vehicles_density": 0.8},
    "medium": {"vehicles_count": 20, "vehicles_density": 1.2},
    "dense":  {"vehicles_count": 20, "vehicles_density": 1.5},
}

DESIRED_SPEED = 25.0  # m/s
BLOCKED_SPEED_RATIO = 0.85  # front_speed < 85% of ego_speed = blocked


def check_blocked(env):
    """Check if ego is blocked behind a significantly slower front vehicle."""
    ego = env.unwrapped.vehicle
    road = env.unwrapped.road
    front, _ = road.neighbour_vehicles(ego, ego.lane_index)
    if front is None:
        return False, float("inf"), 0.0
    try:
        lane = road.network.get_lane(ego.lane_index)
        ego_s = float(lane.local_coordinates(ego.position)[0])
        front_s = float(lane.local_coordinates(front.position)[0])
        gap = front_s - ego_s
        blocked = gap < 80.0 and float(front.speed) < BLOCKED_SPEED_RATIO * float(ego.speed)
        return blocked, gap, float(front.speed)
    except (ValueError, IndexError):
        return False, float("inf"), 0.0


def compute_lane_distribution(env, history):
    """Compute lane usage distribution from lane index history."""
    if not history:
        return {}
    counts = defaultdict(int)
    for lane_idx in history:
        counts[str(lane_idx)] += 1
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


def lane_entropy(lane_dist):
    """Shannon entropy of lane distribution."""
    probs = list(lane_dist.values())
    probs = [p for p in probs if p > 0]
    return -sum(p * np.log(p) for p in probs)


def eval_deep(env, flat_env, controller_type, controller, ppo=None, seed=None):
    """Run deep metric evaluation for one controller."""
    obs, _ = flat_env.reset(seed=seed) if flat_env else (None, None)
    if controller_type == "PPO+RSS":
        obs, _ = flat_env.reset(seed=seed)

    # Tracking
    speeds = []
    blocked_flags = []  # per-step bool
    front_gaps = []
    front_speeds = []
    lane_history = []
    lc_actions = 0
    overtakes = 0
    prev_lane = None
    prev_front_id = None
    crashed = False
    total_reward = 0.0

    for step in range(MAX_STEPS):
        ego = env.unwrapped.vehicle
        current_lane = ego.lane_index
        lane_history.append(current_lane)

        # Get action
        if controller_type == "Stackelberg":
            action, info = controller.decide(env, dt=0.25)
        elif controller_type == "PPO+RSS":
            action, _ = ppo.predict(obs, deterministic=True)
            if isinstance(action, np.ndarray):
                action = int(action.item())
            else:
                action = int(action)
        else:  # MoE
            action, info = controller.decide(env, obs, dt=0.25)

        # Step
        if flat_env:
            obs, reward, terminated, truncated, env_info = flat_env.step(action)
        else:
            obs, reward, terminated, truncated, env_info = env.step(action)
        total_reward += float(reward)

        # Track metrics
        speeds.append(float(ego.speed))
        is_blocked, gap, f_speed = check_blocked(env)
        blocked_flags.append(is_blocked)
        front_gaps.append(gap if np.isfinite(gap) else 80.0)
        front_speeds.append(f_speed)

        # Overtake detection: ego passes a front vehicle (lane change + new front vehicle)
        front, _ = env.unwrapped.road.neighbour_vehicles(ego, current_lane)
        current_front_id = id(front) if front else None
        if prev_lane is not None and current_lane != prev_lane:
            lc_actions += 1
            # Check if this LC resulted in overtaking the previous front vehicle
            if prev_front_id is not None and current_front_id != prev_front_id:
                # We changed lanes and have a different front vehicle = potential overtake
                overtakes += 1
        prev_lane = current_lane
        prev_front_id = current_front_id

        if env_info.get("crashed", False):
            crashed = True
        if terminated or truncated:
            break

    # Compute metrics
    avg_speed = float(np.mean(speeds)) if speeds else 0.0
    min_speed = float(np.min(speeds)) if speeds else 0.0
    blocked_ratio = float(np.mean(blocked_flags)) if blocked_flags else 0.0
    avg_front_gap = float(np.mean(front_gaps)) if front_gaps else 0.0
    avg_front_speed = float(np.mean([s for s in front_speeds if s > 0])) if front_speeds else 0.0
    lane_dist = compute_lane_distribution(env, lane_history)
    ent = lane_entropy(lane_dist)

    # Speed when blocked vs unblocked
    blocked_speeds = [s for s, b in zip(speeds, blocked_flags) if b]
    unblocked_speeds = [s for s, b in zip(speeds, blocked_flags) if not b]
    speed_blocked = float(np.mean(blocked_speeds)) if blocked_speeds else 0.0
    speed_unblocked = float(np.mean(unblocked_speeds)) if unblocked_speeds else 0.0

    return {
        "steps": len(speeds),
        "avg_speed": avg_speed,
        "min_speed": min_speed,
        "blocked_ratio": blocked_ratio,
        "avg_front_gap": avg_front_gap,
        "avg_front_speed": avg_front_speed,
        "lane_entropy": ent,
        "lane_dist": lane_dist,
        "lc_actions": lc_actions,
        "overtakes": overtakes,
        "speed_blocked": speed_blocked,
        "speed_unblocked": speed_unblocked,
        "crashed": crashed,
        "total_reward": total_reward,
    }


def print_table(results_by_method):
    """Print comparison table."""
    metrics = [
        ("avg_speed", "Avg Speed (m/s)", "{:.1f}"),
        ("min_speed", "Min Speed (m/s)", "{:.1f}"),
        ("blocked_ratio", "Blocked Ratio", "{:.1%}"),
        ("avg_front_gap", "Avg Front Gap (m)", "{:.1f}"),
        ("avg_front_speed", "Avg Front Speed (m/s)", "{:.1f}"),
        ("lc_actions", "LC Actions", "{:.1f}"),
        ("overtakes", "Overtakes", "{:.1f}"),
        ("lane_entropy", "Lane Entropy", "{:.3f}"),
        ("speed_blocked", "Speed When Blocked", "{:.1f}"),
        ("speed_unblocked", "Speed Unblocked", "{:.1f}"),
        ("total_reward", "Total Reward", "{:.1f}"),
    ]

    for density in ["sparse", "medium", "dense"]:
        print(f"\n{'='*100}")
        print(f"  DENSITY: {density} ({DENSITY_CONFIGS[density]})")
        print(f"{'='*100}")
        header = f"{'Metric':<25s}"
        for m in ["Stackelberg", "PPO+RSS", "MoE_Hybrid"]:
            header += f" {m:>18s}"
        print(header)
        print("-" * 100)

        for key, label, fmt in metrics:
            row = f"{label:<25s}"
            for method in ["Stackelberg", "PPO+RSS", "MoE_Hybrid"]:
                vals = [r[key] for r in results_by_method[method] if r["density"] == density]
                if vals:
                    mean_val = np.mean(vals)
                    std_val = np.std(vals)
                    row += f" {fmt.format(mean_val) + chr(0xB1) + fmt.format(std_val):>18s}"
                else:
                    row += f" {'N/A':>18s}"
            print(row)

    # Overall
    print(f"\n{'='*100}")
    print("  OVERALL (all densities)")
    print(f"{'='*100}")
    header = f"{'Metric':<25s}"
    for m in ["Stackelberg", "PPO+RSS", "MoE_Hybrid"]:
        header += f" {m:>18s}"
    print(header)
    print("-" * 100)
    for key, label, fmt in metrics:
        row = f"{label:<25s}"
        for method in ["Stackelberg", "PPO+RSS", "MoE_Hybrid"]:
            vals = [r[key] for r in results_by_method[method]]
            if vals:
                mean_val = np.mean(vals)
                std_val = np.std(vals)
                row += f" {fmt.format(mean_val) + chr(0xB1) + fmt.format(std_val):>18s}"
            else:
                row += f" {'N/A':>18s}"
        print(row)


def main():
    ppo_model = PPO.load(PPO_MODEL, device="cpu")
    rss_cfg = RSSConfig(**RSS_CONFIG)
    game_cfg = GameConfig()

    results_by_method = {"Stackelberg": [], "PPO+RSS": [], "MoE_Hybrid": []}

    for density_name, density_cfg in DENSITY_CONFIGS.items():
        for seed in SEEDS:
            print(f"[{density_name}] seed={seed} ...", end=" ", flush=True)

            # --- PPO env config (20 vehicles, 140-dim obs) ---
            ppo_config = dict(ENV_CONFIG)
            ppo_config["vehicles_density"] = density_cfg["vehicles_density"]
            ppo_config["action"] = {"type": "DiscreteMetaAction", "target_speeds": [0,5,10,15,20,25,30]}

            # --- Stackelberg ---
            env_s = gym.make("highway-fast-v0", config=ppo_config)
            env_s.reset(seed=seed)
            r_s = eval_deep(env_s, None, "Stackelberg", StackelbergExpert(game_cfg), seed=seed)
            r_s["density"] = density_name
            r_s["seed"] = seed
            results_by_method["Stackelberg"].append(r_s)
            env_s.close()
            print(f"[S: spd={r_s['avg_speed']:.1f} blk={r_s['blocked_ratio']:.0%} ovt={r_s['overtakes']}]", end=" ")

            # --- PPO+RSS ---
            env_p = gym.make("highway-fast-v0", config=ppo_config)
            env_p = RSSSafetyWrapper(env_p, rss_config=rss_cfg)
            env_p.reset(seed=seed)
            flat_p = FlattenObservation(env_p)
            r_p = eval_deep(env_p, flat_p, "PPO+RSS", None, ppo=ppo_model, seed=seed)
            r_p["density"] = density_name
            r_p["seed"] = seed
            results_by_method["PPO+RSS"].append(r_p)
            flat_p.close()
            print(f"[P: spd={r_p['avg_speed']:.1f} blk={r_p['blocked_ratio']:.0%} ovt={r_p['overtakes']}]", end=" ")

            # --- MoE Hybrid ---
            base_env, flat_env = moe_make_env(
                vehicles=20,
                duration=30,
                density=density_cfg["vehicles_density"],
                seed=seed,
                render=False,
            )
            hybrid = HybridExpert(ppo_model_path=PPO_MODEL)
            hybrid.reset()
            r_m = eval_deep(base_env, flat_env, "MoE_Hybrid", hybrid, seed=seed)
            r_m["density"] = density_name
            r_m["seed"] = seed
            results_by_method["MoE_Hybrid"].append(r_m)
            flat_env.close()
            print(f"[M: spd={r_m['avg_speed']:.1f} blk={r_m['blocked_ratio']:.0%} ovt={r_m['overtakes']}]")

    # Print results
    print_table(results_by_method)

    # Save JSON
    save_path = Path("results/data/deep_metrics.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    flat_data = []
    for method, entries in results_by_method.items():
        for r in entries:
            r_copy = dict(r)
            r_copy["method"] = method
            flat_data.append(r_copy)
    with open(save_path, "w") as f:
        json.dump(flat_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
