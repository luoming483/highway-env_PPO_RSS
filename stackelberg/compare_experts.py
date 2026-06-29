"""Compare Stackelberg Expert vs Baseline strategies across multiple seeds.

Methods compared:
    1. Stackelberg Expert  — game-theoretic lane-change + FSM governance
    2. IDM Baseline         — highway-env built-in IDM controller (IDLE every step)
    3. Random Policy        — uniformly random DiscreteMetaAction
    4. PPO+RSS              — trained PPO with RSS safety shield (requires model)

Metrics per seed:
    collision, steps, avg_speed, lc_count, min_ttc, min_gap,
    emergency_brakes, actions distribution, fsm_states distribution

Usage:
    D:\\anaconda\\envs\\ppo_main\\python.exe -m stackelberg.compare_experts
    D:\\anaconda\\envs\\ppo_main\\python.exe -m stackelberg.compare_experts --ppo-model results/models/ppo_model.zip
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stackelberg import GameConfig, StackelbergExpert

# ---- Config ----
SEEDS = [42, 123, 456, 789, 1024, 2048, 4096, 8192]
DENSITY_LEVELS = {
    "sparse":  {"vehicles_count": 10, "vehicles_density": 0.8},
    "medium":  {"vehicles_count": 20, "vehicles_density": 1.5},
    "dense":   {"vehicles_count": 30, "vehicles_density": 2.0},
}

MAX_STEPS = 200


def make_env(density_name: str, seed: int) -> gym.Env:
    density = DENSITY_LEVELS[density_name]
    config = {
        "observation": {"type": "Kinematics", "vehicles_count": density["vehicles_count"]},
        "action": {
            "type": "DiscreteMetaAction",
            "target_speeds": [0, 5, 10, 15, 20, 25, 30],
        },
        "lanes_count": 3,
        "vehicles_count": density["vehicles_count"],
        "vehicles_density": density["vehicles_density"],
        "duration": 50,
        "simulation_frequency": 15,
        "policy_frequency": 4,
        "collision_reward": -5.0,
        "normalize_reward": True,
        "offroad_terminal": True,
    }
    env = gym.make("highway-fast-v0", config=config)
    env.reset(seed=seed)
    return env


@dataclass
class RunResult:
    method: str
    density: str
    seed: int
    crashed: bool = False
    steps: int = 0
    avg_speed: float = 0.0
    lc_count: int = 0
    min_ttc: float = float("inf")
    min_gap: float = float("inf")
    emergency_brakes: int = 0
    actions: Dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0})
    speeds: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "density": self.density,
            "seed": self.seed,
            "crashed": self.crashed,
            "steps": self.steps,
            "avg_speed": self.avg_speed,
            "lc_count": self.lc_count,
            "min_ttc": self.min_ttc if np.isfinite(self.min_ttc) else None,
            "min_gap": self.min_gap if np.isfinite(self.min_gap) else None,
            "emergency_brakes": self.emergency_brakes,
            "actions": self.actions,
        }


def run_stackelberg(env: gym.Env, density: str = "", seed: int = 0) -> RunResult:
    expert = StackelbergExpert(GameConfig())
    expert.reset()
    done = False
    step = 0
    speeds = []
    lc_count = 0
    emergency_brakes = 0
    min_ttc = float("inf")
    min_gap = float("inf")
    actions = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    crashed = False
    fsm = expert.fsm

    while not done and step < MAX_STEPS:
        action, info = expert.decide(env, dt=0.25)
        obs, reward, terminated, truncated, env_info = env.step(action)
        step += 1
        speeds.append(float(env.unwrapped.vehicle.speed))
        actions[action] = actions.get(action, 0) + 1
        if action in (0, 2):
            lc_count += 1
        reason = info.get("fsm_reason", "")
        if "emergency_brake" in reason:
            emergency_brakes += 1
        cur_gap, cur_rel, _, _ = fsm._get_gaps_from_env(env, lane_offset=0)
        cur_ttc = fsm._safety_gate.predict_ttc(cur_gap, cur_rel)
        if cur_gap < min_gap:
            min_gap = cur_gap
        if np.isfinite(cur_ttc) and cur_ttc < min_ttc:
            min_ttc = cur_ttc
        if env_info.get("crashed", False):
            crashed = True
        done = terminated or truncated

    return RunResult(
        method="Stackelberg", density=density, seed=seed,
        crashed=crashed, steps=step,
        avg_speed=np.mean(speeds) if speeds else 0.0,
        lc_count=lc_count, min_ttc=min_ttc, min_gap=min_gap,
        emergency_brakes=emergency_brakes, actions=actions, speeds=speeds,
    )


def run_idm_baseline(env: gym.Env, density: str = "", seed: int = 0) -> RunResult:
    """IDM baseline: always emit IDLE, let highway-env's built-in IDM controller decide."""
    done = False
    step = 0
    speeds = []
    lc_count = 0
    min_ttc = float("inf")
    min_gap = float("inf")
    actions = {1: 0}
    crashed = False
    ego = env.unwrapped.vehicle

    while not done and step < MAX_STEPS:
        obs, reward, terminated, truncated, env_info = env.step(1)  # IDLE
        step += 1
        speeds.append(float(ego.speed))
        actions[1] = actions.get(1, 0) + 1

        # Compute gap/TTC using same method as Stackelberg for fair comparison
        try:
            road = env.unwrapped.road
            ego_lane = ego.lane_index
            lane = road.network.get_lane(ego_lane)
            ego_s, _ = lane.local_coordinates(ego.position)
            front, _ = road.neighbour_vehicles(ego, ego_lane)
            if front is not None:
                front_s, _ = lane.local_coordinates(front.position)
                gap = float(front_s) - float(ego_s)
                rel_speed = float(ego.speed) - float(front.speed)
                if gap < min_gap:
                    min_gap = gap
                if rel_speed > 1e-6 and gap > 0:
                    ttc = gap / rel_speed
                    if ttc < min_ttc:
                        min_ttc = ttc
        except (ValueError, IndexError, AttributeError):
            pass

        if env_info.get("crashed", False):
            crashed = True
        done = terminated or truncated

    return RunResult(
        method="IDM_Baseline", density=density, seed=seed,
        crashed=crashed, steps=step,
        avg_speed=np.mean(speeds) if speeds else 0.0,
        lc_count=lc_count, min_ttc=min_ttc, min_gap=min_gap,
        emergency_brakes=0, actions=actions, speeds=speeds,
    )


def run_random(env: gym.Env, density: str = "", seed: int = 0) -> RunResult:
    """Random baseline: uniformly random actions from DiscreteMetaAction space."""
    done = False
    step = 0
    speeds = []
    lc_count = 0
    min_ttc = float("inf")
    min_gap = float("inf")
    actions = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    crashed = False
    ego = env.unwrapped.vehicle

    while not done and step < MAX_STEPS:
        action = np.random.randint(0, 5)
        obs, reward, terminated, truncated, env_info = env.step(action)
        step += 1
        speeds.append(float(ego.speed))
        actions[action] = actions.get(action, 0) + 1
        if action in (0, 2):
            lc_count += 1

        try:
            road = env.unwrapped.road
            ego_lane = ego.lane_index
            lane = road.network.get_lane(ego_lane)
            ego_s, _ = lane.local_coordinates(ego.position)
            front, _ = road.neighbour_vehicles(ego, ego_lane)
            if front is not None:
                front_s, _ = lane.local_coordinates(front.position)
                gap = float(front_s) - float(ego_s)
                rel_speed = float(ego.speed) - float(front.speed)
                if gap < min_gap:
                    min_gap = gap
                if rel_speed > 1e-6 and gap > 0:
                    ttc = gap / rel_speed
                    if ttc < min_ttc:
                        min_ttc = ttc
        except (ValueError, IndexError, AttributeError):
            pass

        if env_info.get("crashed", False):
            crashed = True
        done = terminated or truncated

    return RunResult(
        method="Random", density=density, seed=seed,
        crashed=crashed, steps=step,
        avg_speed=np.mean(speeds) if speeds else 0.0,
        lc_count=lc_count, min_ttc=min_ttc, min_gap=min_gap,
        emergency_brakes=0, actions=actions, speeds=speeds,
    )


def run_ppo_rss(env: gym.Env, model_path: str, density: str = "", seed: int = 0) -> Optional[RunResult]:
    """PPO+RSS baseline using a trained PPO model."""
    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("  [SKIP] stable-baselines3 not installed")
        return None

    model = PPO.load(model_path, device="cpu")
    from config import RSS_CONFIG
    from rss_safety import RSSConfig as _RSSConfig, RSSSafetyWrapper

    rss_env = RSSSafetyWrapper(env, rss_config=_RSSConfig(**RSS_CONFIG))
    from gymnasium.wrappers import FlattenObservation
    rss_env = FlattenObservation(rss_env)

    done = False
    step = 0
    speeds = []
    lc_count = 0
    min_ttc = float("inf")
    min_gap = float("inf")
    actions = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    crashed = False

    while not done and step < MAX_STEPS:
        action, _ = model.predict(rss_env.observation(obs) if 'obs' in dir() else obs, deterministic=True)
        obs, reward, terminated, truncated, env_info = rss_env.step(action)
        step += 1
        speeds.append(float(env.unwrapped.vehicle.speed))
        actions[action] = actions.get(action, 0) + 1
        if action in (0, 2):
            lc_count += 1
        ttc = float(env_info.get("rss_min_ttc", float("inf")))
        gap = float(env_info.get("rss_min_distance", float("inf")))
        if np.isfinite(ttc) and ttc < min_ttc:
            min_ttc = ttc
        if np.isfinite(gap) and gap < min_gap:
            min_gap = gap
        if env_info.get("crashed", False):
            crashed = True
        done = terminated or truncated

    return RunResult(
        method="PPO+RSS", density=density, seed=seed,
        crashed=crashed, steps=step,
        avg_speed=np.mean(speeds) if speeds else 0.0,
        lc_count=lc_count, min_ttc=min_ttc, min_gap=min_gap,
        emergency_brakes=0, actions=actions, speeds=speeds,
    )


def print_summary(all_results: List[RunResult]):
    """Print comparison summary grouped by method and density."""
    from collections import defaultdict

    grouped = defaultdict(list)
    for r in all_results:
        grouped[(r.method, r.density)].append(r)

    methods = ["Stackelberg", "IDM_Baseline", "Random"]
    densities = list(DENSITY_LEVELS.keys())

    print()
    print("=" * 120)
    print("COMPARISON SUMMARY")
    print("=" * 120)

    header = f"{'Method':<16s} {'Density':<8s} {'Seeds':<6s} {'Collisions':<11s} {'AvgSpeed':<9s} {'LC/1000st':<10s} {'MinTTC':<7s} {'MinGap':<7s} {'EmergBrake':<10s}"
    print(header)
    print("-" * 120)

    for method in methods:
        for density in densities:
            key = (method, density)
            if key not in grouped:
                continue
            runs = grouped[key]
            n = len(runs)
            collisions = sum(1 for r in runs if r.crashed)
            avg_speed = np.mean([r.avg_speed for r in runs])
            total_steps = sum(r.steps for r in runs)
            total_lc = sum(r.lc_count for r in runs)
            lc_per_1k = (total_lc / max(total_steps, 1)) * 1000
            min_ttc_vals = [r.min_ttc for r in runs if np.isfinite(r.min_ttc)]
            min_ttc_avg = np.mean(min_ttc_vals) if min_ttc_vals else float("nan")
            min_gap_vals = [r.min_gap for r in runs if np.isfinite(r.min_gap)]
            min_gap_avg = np.mean(min_gap_vals) if min_gap_vals else float("nan")
            emerg = sum(r.emergency_brakes for r in runs)

            ttc_str = f"{min_ttc_avg:.1f}s" if not np.isnan(min_ttc_avg) else "N/A"
            gap_str = f"{min_gap_avg:.1f}m" if not np.isnan(min_gap_avg) else "N/A"

            print(f"{method:<16s} {density:<8s} {n:<6d} {collisions}/{n} ({collisions/n:.0%}){'':>3s} "
                  f"{avg_speed:6.1f} m/s  {lc_per_1k:6.1f}      {ttc_str:<7s} {gap_str:<7s} {emerg:<10d}")

    print("-" * 120)
    print()


def print_gating_analysis(all_results: List[RunResult]):
    """Analyze scenarios where Stackelberg and IDM differ, to inform MoE gating design."""
    print("=" * 120)
    print("GATING ANALYSIS — When does each method fail?")
    print("=" * 120)

    # Collect per-method results across densities
    from collections import defaultdict
    by_method = defaultdict(list)
    for r in all_results:
        by_method[r.method].append(r)

    for method in ["Stackelberg", "IDM_Baseline", "Random"]:
        runs = by_method.get(method, [])
        if not runs:
            continue
        collisions = sum(1 for r in runs if r.crashed)
        n = len(runs)
        avg_speed = np.mean([r.avg_speed for r in runs])
        avg_min_ttc = np.mean([r.min_ttc for r in runs if np.isfinite(r.min_ttc)])

        # Count scenarios by density
        by_density = defaultdict(list)
        for r in runs:
            by_density[r.density].append(r)

        print(f"\n  {method}:")
        print(f"    Collision rate: {collisions}/{n} ({collisions/n:.1%})")
        print(f"    Avg speed:      {avg_speed:.1f} m/s")
        print(f"    Avg min TTC:    {avg_min_ttc:.1f}s")
        for density in ["sparse", "medium", "dense"]:
            dr = by_density.get(density, [])
            if dr:
                c = sum(1 for r in dr if r.crashed)
                spd = np.mean([r.avg_speed for r in dr])
                lc = sum(r.lc_count for r in dr)
                print(f"    {density:<8s}:  collisions={c}/{len(dr)}  avg_speed={spd:.1f} m/s  lc_actions={lc}")

    print()
    print("Gating recommendations will be based on the above metrics.")
    print("Key questions:")
    print("  1. Does Stackelberg outperform IDM in high-density / low-TTC scenarios?")
    print("  2. Does IDM achieve higher avg speed in sparse traffic?")
    print("  3. Is there a clear TTC/gap threshold where one method dominates?")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppo-model", type=str, default=None,
                        help="Path to trained PPO model .zip file")
    parser.add_argument("--seeds", type=int, default=8,
                        help="Number of seeds to test")
    parser.add_argument("--density", type=str, default=None,
                        help="Only test one density level")
    parser.add_argument("--json-output", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    seeds = SEEDS[:args.seeds]
    densities = [args.density] if args.density else list(DENSITY_LEVELS.keys())

    all_results: List[RunResult] = []

    print("Stackelberg Expert — Comparison Experiment")
    print("=" * 60)
    print(f"Methods: Stackelberg | IDM_Baseline | Random", end="")
    if args.ppo_model:
        print(" | PPO+RSS", end="")
    print()
    print(f"Densities: {densities}")
    print(f"Seeds: {len(seeds)}")
    print(f"Max steps/episode: {MAX_STEPS}")
    print()

    for density in densities:
        print(f"\n{'='*60}")
        print(f"Density: {density} ({DENSITY_LEVELS[density]})")
        print(f"{'='*60}")

        for i, seed in enumerate(seeds):
            print(f"  Seed {seed:4d} ({i+1}/{len(seeds)})...", end=" ", flush=True)

            # Stackelberg
            env = make_env(density, seed)
            t0 = time.time()
            result = run_stackelberg(env, density=density, seed=seed)
            dt = time.time() - t0
            all_results.append(result)
            env.close()
            print(f"[Stackelberg] crash={result.crashed} spd={result.avg_speed:.1f} lc={result.lc_count} "
                  f"ttc={result.min_ttc:.1f}s dt={dt:.2f}s", end=" | ", flush=True)

            # IDM Baseline
            env = make_env(density, seed)
            result = run_idm_baseline(env, density=density, seed=seed)
            all_results.append(result)
            env.close()
            print(f"[IDM] crash={result.crashed} spd={result.avg_speed:.1f}", end=" | ", flush=True)

            # Random
            env = make_env(density, seed)
            result = run_random(env, density=density, seed=seed)
            all_results.append(result)
            env.close()
            print(f"[Random] crash={result.crashed} spd={result.avg_speed:.1f} lc={result.lc_count}")

            # PPO+RSS (if model provided)
            if args.ppo_model:
                env = make_env(density, seed)
                result = run_ppo_rss(env, args.ppo_model)
                if result:
                    result.density = density
                    result.seed = seed
                    all_results.append(result)
                env.close()

    print_summary(all_results)
    print_gating_analysis(all_results)

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = [r.to_dict() for r in all_results]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
