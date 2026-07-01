"""Ablation experiment: Can threshold tuning replace MoE architecture?

Tests three hypotheses:
  H1: "Relax Stackelberg FSM thresholds → match MoE speed without sacrificing safety"
  H2: "Tighten RSS thresholds → match MoE safety without losing speed"
  H3: "MoE's advantage is architecture (multi-expert), not parameter choice"

Variants:
  A. Stackelberg-Aggressive: FSM TTC 5.0→3.0s, gap_margin 4x→2x, min_safe 5→3m
  B. Stackelberg-Default: current config
  C. PPO+RSS-Safe: RSS TTC 3.0→5.0s, min_distance 8→20m
  D. PPO+RSS-Default: current RSS config
  E. MoE-Hybrid: current gate

Usage:
    D:\\anaconda\\envs\\ppo_main\\python.exe tools/ablation_threshold.py
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
from gymnasium.wrappers import FlattenObservation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import RSS_CONFIG
from rss import RSSConfig, RSSSafetyWrapper
from stackelberg.config import GameConfig
from stackelberg.expert import StackelbergExpert
from moe_hybrid import HybridExpert, MoEGate, SceneFeatures

# ---- Variant configs ----

def make_aggressive_game_config() -> GameConfig:
    """Stackelberg with FSM thresholds relaxed to RSS-level aggressiveness."""
    c = GameConfig()
    c.ttc_safe_threshold = 3.0       # was 5.0 — match RSS level
    c.gap_safety_margin = 0.8        # was 1.2 — accept tighter gaps
    c.min_safe_distance = 3.0        # was 5.0 — reduce safety buffer
    c.min_cruise_speed = 18.0        # was 15.0 — raise speed floor
    c.cost_improvement_threshold = 0.05  # was 0.10 — easier to trigger LC
    c.rear_ttc_warning = 4.0         # was 6.0 — less rear-aware braking
    c.rear_ttc_critical = 2.0        # was 3.0
    c.rear_gap_warning = 20.0        # was 30.0
    return c


def make_safe_rss_config() -> dict:
    """RSS with thresholds tightened to FSM-level conservatism."""
    cfg = dict(RSS_CONFIG)
    cfg["ttc_threshold"] = 5.0       # was 3.0 — match FSM level
    cfg["min_distance"] = 20.0       # was 8.0 — match FSM safety margin
    cfg["response_time"] = 1.5       # was 1.0 — more conservative
    cfg["lane_change_side_gap"] = 12.0  # was 8.0
    return cfg


# ---- Environment factory ----
ENV_BASE = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 20,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "absolute": False,
    },
    "action": {
        "type": "DiscreteMetaAction",
        "target_speeds": [0, 5, 10, 15, 20, 25, 30],
    },
    "lanes_count": 4,
    "vehicles_count": 20,
    "simulation_frequency": 8,
    "policy_frequency": 4,
    "collision_reward": -5.0,
    "normalize_reward": True,
    "offroad_terminal": True,
}


def make_env(density: float, duration: int, seed: int, rss_config_dict: dict):
    cfg = dict(ENV_BASE)
    cfg["vehicles_density"] = density
    cfg["duration"] = duration
    base = gym.make("highway-fast-v0", config=cfg)
    rss_env = RSSSafetyWrapper(base, rss_config=RSSConfig(**rss_config_dict))
    flat = FlattenObservation(rss_env)
    return rss_env, flat


# ---- Evaluation ----
@dataclass
class EvalResult:
    method: str
    density_label: str
    density: float
    seed: int
    crashed: bool
    steps: int
    avg_speed: float
    min_ttc: float
    min_gap: float
    lc_count: int
    actions: dict
    extra: dict = None


def run_stackelberg(env, game_config: GameConfig, max_steps: int = 200) -> EvalResult:
    expert = StackelbergExpert(game_config)
    obs, _ = env.reset()
    expert.reset()
    total_steps = 0
    crashed = False
    speeds, ttcs, gaps = [], [], []
    actions = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    lc_count = 0

    for _ in range(max_steps):
        action, info = expert.decide(env, dt=0.25)
        obs, reward, terminated, truncated, env_info = env.step(action)
        total_steps += 1
        actions[action] = actions.get(action, 0) + 1
        speeds.append(float(env.unwrapped.vehicle.speed))

        # TTC/gap
        ego = env.unwrapped.vehicle
        road = env.unwrapped.road
        front, _ = road.neighbour_vehicles(ego, ego.lane_index)
        if front is not None:
            try:
                lane = road.network.get_lane(ego.lane_index)
                ego_s = float(lane.local_coordinates(ego.position)[0])
                front_s = float(lane.local_coordinates(front.position)[0])
                gap = front_s - ego_s
                rel = float(ego.speed - front.speed)
                gaps.append(gap)
                ttcs.append(gap / rel if gap > 0 and rel > 1e-6 else float("inf"))
            except (ValueError, IndexError):
                pass

        if action in (0, 2):  # LEFT or RIGHT
            lc_count += 1

        if env_info.get("crashed", False):
            crashed = True
        if terminated or truncated:
            break

    return EvalResult(
        method=f"Stackelberg-{('Aggressive' if game_config.ttc_safe_threshold < 4.0 else 'Default')}",
        density_label="",
        density=0,
        seed=0,
        crashed=crashed,
        steps=total_steps,
        avg_speed=float(np.mean(speeds)) if speeds else 0.0,
        min_ttc=float(np.min(ttcs)) if ttcs else float("inf"),
        min_gap=float(np.min(gaps)) if gaps else float("inf"),
        lc_count=lc_count,
        actions=actions,
    )


def run_ppo_rss(flat_env, ppo_model, max_steps: int = 200) -> EvalResult:
    obs, _ = flat_env.reset()
    total_steps = 0
    crashed = False
    speeds, ttcs, gaps = [], [], []
    actions = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    for _ in range(max_steps):
        action, _ = ppo_model.predict(obs, deterministic=True)
        if isinstance(action, np.ndarray):
            action = int(action.item())
        else:
            action = int(action)
        obs, reward, terminated, truncated, env_info = flat_env.step(action)
        total_steps += 1
        actions[action] = actions.get(action, 0) + 1

        env = flat_env.unwrapped.unwrapped if hasattr(flat_env, 'unwrapped') else flat_env
        ego = env.unwrapped.vehicle
        road = env.unwrapped.road
        speeds.append(float(ego.speed))
        front, _ = road.neighbour_vehicles(ego, ego.lane_index)
        if front is not None:
            try:
                lane = road.network.get_lane(ego.lane_index)
                ego_s = float(lane.local_coordinates(ego.position)[0])
                front_s = float(lane.local_coordinates(front.position)[0])
                gap = front_s - ego_s
                rel = float(ego.speed - front.speed)
                gaps.append(gap)
                ttcs.append(gap / rel if gap > 0 and rel > 1e-6 else float("inf"))
            except (ValueError, IndexError):
                pass

        if env_info.get("crashed", False):
            crashed = True
        if terminated or truncated:
            break

    return EvalResult(
        method="PPO+RSS",
        density_label="",
        density=0,
        seed=0,
        crashed=crashed,
        steps=total_steps,
        avg_speed=float(np.mean(speeds)) if speeds else 0.0,
        min_ttc=float(np.min(ttcs)) if ttcs else float("inf"),
        min_gap=float(np.min(gaps)) if gaps else float("inf"),
        lc_count=actions.get(0, 0) + actions.get(2, 0),
        actions=actions,
    )


def run_moe_hybrid(env, flat_env, hybrid, max_steps: int = 200) -> EvalResult:
    obs, _ = flat_env.reset()
    hybrid.reset()
    total_steps = 0
    crashed = False
    speeds, ttcs, gaps = [], [], []
    actions = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    expert_counts = {"rss_emergency": 0, "stackelberg": 0, "ppo_rss": 0}

    for _ in range(max_steps):
        action, info = hybrid.decide(env, obs)
        obs, reward, terminated, truncated, env_info = flat_env.step(action)
        total_steps += 1
        actions[action] = actions.get(action, 0) + 1
        speeds.append(info["scene_ego_speed"])
        ttcs.append(info["scene_front_ttc"])
        gaps.append(info["scene_front_gap"])
        expert_counts[info["moe_expert"]] += 1
        if env_info.get("crashed", False):
            crashed = True
        if terminated or truncated:
            break

    return EvalResult(
        method="MoE-Hybrid",
        density_label="",
        density=0,
        seed=0,
        crashed=crashed,
        steps=total_steps,
        avg_speed=float(np.mean(speeds)) if speeds else 0.0,
        min_ttc=float(np.min(ttcs)) if ttcs else float("inf"),
        min_gap=float(np.min(gaps)) if gaps else float("inf"),
        lc_count=actions.get(0, 0) + actions.get(2, 0),
        actions=actions,
        extra={"expert_dist": {k: v / max(total_steps, 1) for k, v in expert_counts.items()}},
    )


# ---- Main ----
def main():
    from stable_baselines3 import PPO

    model_path = str(Path("runs/20260615_163841/models/our_method_seed42/final_model.zip").resolve())
    ppo_model = PPO.load(model_path, device="cpu")

    densities = [("sparse", 0.8), ("medium", 1.2), ("dense", 1.5)]
    seeds = [42, 123, 456, 789]
    duration = 25
    all_results: List[EvalResult] = []

    # Variant A: Stackelberg-Aggressive
    print("=" * 60)
    print("Variant A: Stackelberg-Aggressive (FSM relaxed)")
    print("=" * 60)
    agg_config = make_aggressive_game_config()
    for dname, dval in densities:
        for seed in seeds:
            env, _ = make_env(dval, duration, seed, RSS_CONFIG)
            result = run_stackelberg(env, agg_config)
            result.density_label = dname
            result.density = dval
            result.seed = seed
            result.method = "Stackelberg-Aggressive"
            all_results.append(result)
            print(f"  {dname:7s} seed={seed}: speed={result.avg_speed:.1f} crashed={result.crashed} "
                  f"ttc={result.min_ttc:.1f}s gap={result.min_gap:.1f}m lc={result.lc_count}")

    # Variant B: Stackelberg-Default
    print()
    print("=" * 60)
    print("Variant B: Stackelberg-Default")
    print("=" * 60)
    def_config = GameConfig()
    for dname, dval in densities:
        for seed in seeds:
            env, _ = make_env(dval, duration, seed, RSS_CONFIG)
            result = run_stackelberg(env, def_config)
            result.density_label = dname
            result.density = dval
            result.seed = seed
            result.method = "Stackelberg-Default"
            all_results.append(result)
            print(f"  {dname:7s} seed={seed}: speed={result.avg_speed:.1f} crashed={result.crashed} "
                  f"ttc={result.min_ttc:.1f}s gap={result.min_gap:.1f}m lc={result.lc_count}")

    # Variant C: PPO+RSS-Safe
    print()
    print("=" * 60)
    print("Variant C: PPO+RSS-Safe (RSS tightened)")
    print("=" * 60)
    safe_rss = make_safe_rss_config()
    for dname, dval in densities:
        for seed in seeds:
            env, flat = make_env(dval, duration, seed, safe_rss)
            result = run_ppo_rss(flat, ppo_model)
            result.density_label = dname
            result.density = dval
            result.seed = seed
            result.method = "PPO+RSS-Safe"
            all_results.append(result)
            print(f"  {dname:7s} seed={seed}: speed={result.avg_speed:.1f} crashed={result.crashed} "
                  f"ttc={result.min_ttc:.1f}s gap={result.min_gap:.1f}m lc={result.lc_count}")

    # Variant D: PPO+RSS-Default
    print()
    print("=" * 60)
    print("Variant D: PPO+RSS-Default")
    print("=" * 60)
    for dname, dval in densities:
        for seed in seeds:
            env, flat = make_env(dval, duration, seed, RSS_CONFIG)
            result = run_ppo_rss(flat, ppo_model)
            result.density_label = dname
            result.density = dval
            result.seed = seed
            result.method = "PPO+RSS-Default"
            all_results.append(result)
            print(f"  {dname:7s} seed={seed}: speed={result.avg_speed:.1f} crashed={result.crashed} "
                  f"ttc={result.min_ttc:.1f}s gap={result.min_gap:.1f}m lc={result.lc_count}")

    # Variant E: MoE-Hybrid
    print()
    print("=" * 60)
    print("Variant E: MoE-Hybrid")
    print("=" * 60)
    hybrid = HybridExpert(ppo_model_path=model_path)
    for dname, dval in densities:
        for seed in seeds:
            env, flat = make_env(dval, duration, seed, RSS_CONFIG)
            result = run_moe_hybrid(env, flat, hybrid)
            result.density_label = dname
            result.density = dval
            result.seed = seed
            result.method = "MoE-Hybrid"
            all_results.append(result)
            dist = result.extra.get("expert_dist", {})
            print(f"  {dname:7s} seed={seed}: speed={result.avg_speed:.1f} crashed={result.crashed} "
                  f"ttc={result.min_ttc:.1f}s gap={result.min_gap:.1f}m lc={result.lc_count} "
                  f"Stack={dist.get('stackelberg',0):.0%} PPO={dist.get('ppo_rss',0):.0%}")

    # ---- Summary ----
    print()
    print("=" * 80)
    print("ABLATION SUMMARY")
    print("=" * 80)
    methods_order = ["Stackelberg-Default", "Stackelberg-Aggressive",
                     "PPO+RSS-Default", "PPO+RSS-Safe", "MoE-Hybrid"]

    for method in methods_order:
        entries = [r for r in all_results if r.method == method]
        if not entries:
            continue
        crashes = sum(1 for r in entries if r.crashed)
        speeds = [r.avg_speed for r in entries]
        ttcs = [r.min_ttc for r in entries if np.isfinite(r.min_ttc) and r.min_ttc < 100]
        gaps = [r.min_gap for r in entries if np.isfinite(r.min_gap) and r.min_gap < 500]
        lcs = [r.lc_count for r in entries]
        print(f"\n{method:25s}: n={len(entries)}")
        print(f"  Crashes: {crashes}/{len(entries)} ({crashes/len(entries)*100:.0f}%)")
        print(f"  Speed:   {np.mean(speeds):.1f} +/- {np.std(speeds):.1f} m/s")
        print(f"  Min TTC: {np.mean(ttcs):.1f} +/- {np.std(ttcs):.1f} s")
        print(f"  Min Gap: {np.mean(gaps):.1f} +/- {np.std(gaps):.1f} m")
        print(f"  LC count:{np.mean(lcs):.1f} +/- {np.std(lcs):.1f}")

    # Save raw data
    save_data = []
    for r in all_results:
        save_data.append({
            "method": r.method,
            "density": r.density_label,
            "seed": r.seed,
            "crashed": r.crashed,
            "steps": r.steps,
            "avg_speed": r.avg_speed,
            "min_ttc": r.min_ttc,
            "min_gap": r.min_gap,
            "lc_count": r.lc_count,
            "actions": r.actions,
        })
    out_path = Path(__file__).resolve().parent.parent / "results/data/ablation_threshold.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
