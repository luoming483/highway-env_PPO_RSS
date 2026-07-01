"""Evaluate new phased-trained PPO+RSS model + regenerate all plots.

1. Run compare_experts (Stackelberg, IDM, Random, PPO+RSS) with new model
2. Run MoE hybrid batch eval with new model
3. Regenerate all plots
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools.compare_experts import (
    DENSITY_LEVELS,
    make_env,
    make_env_ppo,
    run_stackelberg,
    run_idm_baseline,
    run_random,
    run_ppo_rss,
    RunResult,
)
from moe_hybrid import HybridExpert, MoEGate, make_env as moe_make_env

SEEDS = [42, 123, 456, 789]
DENSITIES = list(DENSITY_LEVELS.keys())
MAX_STEPS = 200
SAVE_DATA = Path("results/data")
SAVE_PLOTS = Path("results/plots")
PPO_MODEL = "results/models/test_lc_phased_v3_seed42/final_model.zip"


def run_comparison():
    """Run cross-expert comparison with new PPO+RSS model."""
    print("=" * 60)
    print("CROSS-EXPERT COMPARISON (new phased-trained PPO+RSS)")
    print("=" * 60)

    all_results = []

    for density in DENSITIES:
        print(f"\nDensity: {density} ({DENSITY_LEVELS[density]})")

        for i, seed in enumerate(SEEDS):
            print(f"  Seed {seed:4d} ({i+1}/{len(SEEDS)})...", end=" ", flush=True)

            # Stackelberg
            env = make_env(density, seed)
            result = run_stackelberg(env, density=density, seed=seed)
            all_results.append(result.to_dict())
            env.close()
            print(f"[Stack] crash={result.crashed} spd={result.avg_speed:.1f} lc={result.lc_count} ttc={result.min_ttc:.1f}s", end=" | ")

            # IDM Baseline
            env = make_env(density, seed)
            result = run_idm_baseline(env, density=density, seed=seed)
            all_results.append(result.to_dict())
            env.close()
            print(f"[IDM] crash={result.crashed} spd={result.avg_speed:.1f}", end=" | ")

            # Random
            env = make_env(density, seed)
            result = run_random(env, density=density, seed=seed)
            all_results.append(result.to_dict())
            env.close()
            print(f"[Rand] crash={result.crashed} spd={result.avg_speed:.1f}", end="")

            # New PPO+RSS
            ppo_env = make_env_ppo(seed)
            result = run_ppo_rss(ppo_env, PPO_MODEL, density=f"{density}(20v)", seed=seed)
            if result:
                all_results.append(result.to_dict())
                print(f" | [PPO] crash={result.crashed} spd={result.avg_speed:.1f} lc={result.lc_count}", end="")
            ppo_env.close()
            print()

    # Save
    SAVE_DATA.mkdir(parents=True, exist_ok=True)
    with open(SAVE_DATA / "compare_experts.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(all_results)} records to compare_experts.json")
    return all_results


def run_moe_hybrid():
    """Run MoE hybrid batch eval with new PPO+RSS model."""
    print("\n" + "=" * 60)
    print("MoE HYBRID EVALUATION (new phased-trained PPO+RSS)")
    print("=" * 60)

    all_results = []

    for density in DENSITIES:
        n_vehicles = DENSITY_LEVELS[density]["vehicles_count"]
        density_val = DENSITY_LEVELS[density]["vehicles_density"]

        for seed in SEEDS:
            print(f"  Density={density} ({n_vehicles}v), Seed={seed}...", end=" ", flush=True)

            base_env, flat_env = moe_make_env(
                vehicles=n_vehicles,
                duration=30,
                density=density_val,
                seed=seed,
                render=False,
            )

            hybrid = HybridExpert(ppo_model_path=PPO_MODEL)
            hybrid.reset()
            obs, _ = flat_env.reset(seed=seed)

            total_reward = 0.0
            crashed = False
            expert_counts = {"rss_emergency": 0, "stackelberg": 0, "ppo_rss": 0}
            actions = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
            speeds = []
            min_ttc = float("inf")
            min_gap = float("inf")
            lc_count = 0
            steps = 0

            for _ in range(MAX_STEPS):
                action, info = hybrid.decide(base_env, obs, dt=0.25)
                obs, reward, terminated, truncated, env_info = flat_env.step(action)
                total_reward += float(reward)
                steps += 1
                actions[action] = actions.get(action, 0) + 1
                speeds.append(info["scene_ego_speed"])
                expert_counts[info["moe_expert"]] += 1

                if action in (0, 2):
                    lc_count += 1
                ttc = info["scene_front_ttc"]
                gap = info["scene_front_gap"]
                if np.isfinite(ttc) and ttc < min_ttc:
                    min_ttc = ttc
                if np.isfinite(gap) and gap < min_gap:
                    min_gap = gap
                if env_info.get("crashed", False):
                    crashed = True
                if terminated or truncated:
                    break

            flat_env.close()

            expert_dist = {k: v / max(steps, 1) for k, v in expert_counts.items()}
            result = {
                "method": "MoE_Hybrid",
                "density": density,
                "seed": seed,
                "crashed": crashed,
                "steps": steps,
                "avg_speed": float(np.mean(speeds)) if speeds else 0.0,
                "lc_count": lc_count,
                "min_ttc": float(min_ttc) if np.isfinite(min_ttc) else None,
                "min_gap": float(min_gap) if np.isfinite(min_gap) else None,
                "expert_dist": expert_dist,
                "actions": actions,
                "reward": total_reward,
            }
            all_results.append(result)
            print(f"speed={result['avg_speed']:.1f} crash={crashed} lc={lc_count} "
                  f"Stack={expert_dist['stackelberg']:.0%} PPO={expert_dist['ppo_rss']:.0%}")

    # Save
    with open(SAVE_DATA / "moe_hybrid_eval.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(all_results)} records to moe_hybrid_eval.json")
    return all_results


def print_results_summary():
    """Print a concise summary of the new model results."""
    with open(SAVE_DATA / "compare_experts.json") as f:
        compare = json.load(f)
    with open(SAVE_DATA / "moe_hybrid_eval.json") as f:
        moe = json.load(f)

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY — NEW PHASED-TRAINED PPO+RSS MODEL")
    print("=" * 80)

    methods = ["Stackelberg", "PPO+RSS", "MoE_Hybrid", "IDM_Baseline", "Random"]
    for method in methods:
        entries = [r for r in compare + moe if r["method"] == method]
        if not entries:
            continue
        n = len(entries)
        crashes = sum(1 for r in entries if r["crashed"])
        speeds = [r["avg_speed"] for r in entries]
        lcs = [r.get("lc_count", 0) for r in entries]
        ttc_vals = [r.get("min_ttc", float("inf")) for r in entries if r.get("min_ttc") and np.isfinite(r["min_ttc"])]
        print(f"  {method:<16s}: crash={crashes}/{n} speed={np.mean(speeds):.1f}±{np.std(speeds):.1f} m/s "
              f"LC={np.mean(lcs):.1f}±{np.std(lcs):.1f} "
              f"minTTC={np.mean(ttc_vals):.1f}s" if ttc_vals else f"minTTC=N/A")

    # Expert distribution for MoE
    moe_entries = [r for r in moe if r["method"] == "MoE_Hybrid"]
    if moe_entries:
        stack_use = np.mean([r["expert_dist"]["stackelberg"] for r in moe_entries])
        ppo_use = np.mean([r["expert_dist"]["ppo_rss"] for r in moe_entries])
        rss_use = np.mean([r["expert_dist"]["rss_emergency"] for r in moe_entries])
        print(f"  MoE expert distribution: Stackelberg={stack_use:.0%} PPO+RSS={ppo_use:.0%} RSS_Emergency={rss_use:.0%}")


def main():
    t0 = time.time()

    print(f"New PPO model: {PPO_MODEL}")
    print(f"Densities: {DENSITIES}")
    print(f"Seeds: {SEEDS}")
    print(f"Max steps: {MAX_STEPS}")

    # Step 1: Cross-expert comparison
    compare_data = run_comparison()

    # Step 2: MoE hybrid evaluation
    moe_data = run_moe_hybrid()

    # Step 3: Print summary
    print_results_summary()

    # Step 4: Regenerate plots
    print("\n" + "=" * 60)
    print("REGENERATING PLOTS...")
    print("=" * 60)
    from tools.plot_moe_results import (
        fig1_cross_expert_comparison,
        fig2_moe_gate_analysis,
        fig3_pareto_frontier,
        fig4_architecture_diagram,
        fig5_summary_table,
        fig6_speed_profile,
    )
    fig1_cross_expert_comparison(compare_data, moe_data)
    fig2_moe_gate_analysis(moe_data)
    fig3_pareto_frontier(compare_data, moe_data)
    fig4_architecture_diagram()
    fig5_summary_table(compare_data, moe_data)
    fig6_speed_profile(compare_data, moe_data)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")
    print("Done!")


if __name__ == "__main__":
    main()
