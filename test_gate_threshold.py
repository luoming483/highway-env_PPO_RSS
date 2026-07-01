"""Quick test: gate threshold sensitivity analysis for MoE hybrid expert."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from moe_hybrid import HybridExpert, MoEGate, make_env
from deep_metrics_compare import eval_deep, DENSITY_CONFIGS, PPO_MODEL, SEEDS


def run_moe_with_threshold(threshold, density_name, density_cfg, seed):
    """Run MoE with a specific gate cost_improvement_threshold."""
    base_env, flat_env = make_env(
        vehicles=20, duration=30,
        density=density_cfg["vehicles_density"],
        seed=seed, render=False,
    )
    gate = MoEGate(cost_improvement_threshold=threshold)
    hybrid = HybridExpert(ppo_model_path=PPO_MODEL, gate=gate)
    hybrid.reset()
    r = eval_deep(base_env, flat_env, "MoE_Hybrid", hybrid, seed=seed)
    flat_env.close()
    r["stack_usage"] = hybrid.expert_distribution["stackelberg"]
    return r


def main():
    thresholds = [0.10, 0.05, 0.03]
    seed = 42

    print(f"{'Threshold':<12s} {'Density':>8s} {'Speed':>7s} {'Blocked':>8s} "
          f"{'LC':>5s} {'Overtake':>9s} {'LaneEnt':>8s} {'StackUse':>9s} "
          f"{'SpdBlk':>7s} {'SpdUnblk':>7s} {'Crashed':>8s}")
    print("-" * 100)

    for threshold in thresholds:
        for dname, dcfg in DENSITY_CONFIGS.items():
            r = run_moe_with_threshold(threshold, dname, dcfg, seed)
            print(f"{threshold:>12.2f} {dname:>8s} "
                  f"{r['avg_speed']:>6.1f} {r['blocked_ratio']:>7.1%} "
                  f"{r['lc_actions']:>5d} {r['overtakes']:>9d} "
                  f"{r['lane_entropy']:>8.3f} {r['stack_usage']:>8.0%} "
                  f"{r['speed_blocked']:>6.1f} {r['speed_unblocked']:>7.1f} "
                  f"{str(r['crashed']):>8s}")
        print()

    # Averages
    print(f"{'='*100}")
    print("AVERAGE ACROSS DENSITIES")
    print(f"{'='*100}")
    for threshold in thresholds:
        speeds, blk, lcs, ovt, ent, use = [], [], [], [], [], []
        for dname, dcfg in DENSITY_CONFIGS.items():
            r = run_moe_with_threshold(threshold, dname, dcfg, seed)
            speeds.append(r["avg_speed"])
            blk.append(r["blocked_ratio"])
            lcs.append(r["lc_actions"])
            ovt.append(r["overtakes"])
            ent.append(r["lane_entropy"])
            use.append(r["stack_usage"])
        print(f"threshold={threshold:.2f}  speed={np.mean(speeds):.1f}  blocked={np.mean(blk):.1%}  "
              f"LC={np.mean(lcs):.1f}  overtakes={np.mean(ovt):.1f}  "
              f"entropy={np.mean(ent):.3f}  stack_use={np.mean(use):.0%}")


if __name__ == "__main__":
    main()
