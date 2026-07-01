"""Experiment runner for PPO + RSS dual-layer framework.

Usage:
    python experiment.py                     # Run default experiments into runs/<timestamp>/
    python experiment.py --experiments baseline,our_method  # Run a subset
    python experiment.py --seeds 42           # Single seed for quick test
    python experiment.py --timesteps 10000    # Override timesteps
    python experiment.py --skip-plots         # Save data only, no plots
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from config import (
    ACTIVE_EXPERIMENTS,
    EXPERIMENTS,
    LOG_DIR,
    MODEL_DIR,
    RUNS_DIR,
    SEEDS,
    TOTAL_TIMESTEPS,
)
from train import run_training
from plotting import (
    plot_collision_comparison,
    plot_final_performance_bar,
    plot_loss_comparison,
    plot_reward_comparison,
    plot_safety_metrics,
    plot_training_reward_curve,
    set_style,
)


def _build_run_dirs(output_root: Path, run_name: str) -> dict:
    run_dir = output_root / run_name
    dirs = {
        "run": run_dir,
        "models": run_dir / "models",
        "logs": run_dir / "logs",
        "data": run_dir / "data",
        "plots": run_dir / "plots",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def run_experiments(
    experiments: list,
    seeds: list,
    total_timesteps: int,
    device: str = "cpu",
    verbose: int = 0,
    model_dir: Path = None,
    log_dir: Path = None,
) -> dict:
    """Run all specified experiments across all seeds. Returns {exp_name: {seed: metrics}}."""
    model_dir = MODEL_DIR if model_dir is None else Path(model_dir)
    log_dir = LOG_DIR if log_dir is None else Path(log_dir)
    all_results = {}
    total_runs = len(experiments) * len(seeds)
    run_idx = 0

    for exp_name in experiments:
        exp_cfg = EXPERIMENTS[exp_name]
        print(f"\n{'='*60}")
        print(f"[{run_idx + 1}/{total_runs}] Experiment: {exp_cfg['label']}")
        print(f"  RSS={exp_cfg['use_rss']}, Curriculum={exp_cfg['use_curriculum']}")
        print(f"{'='*60}")

        seed_results = {}
        for seed in seeds:
            run_idx += 1
            print(f"\n--- Seed={seed} [{run_idx}/{total_runs}] ---")
            t_start = time.time()

            metrics = run_training(
                exp_name=exp_name,
                use_rss=exp_cfg["use_rss"],
                use_curriculum=exp_cfg["use_curriculum"],
                seed=seed,
                total_timesteps=total_timesteps,
                device=device,
                verbose=verbose,
                rss_overrides=exp_cfg.get("rss_overrides", {}),
                train_rss_overrides=exp_cfg.get("train_rss_overrides", {}),
                use_blocked_penalty=exp_cfg.get("use_blocked_penalty", False),
                use_force_explore=exp_cfg.get("use_force_explore", False),
                model_dir=model_dir,
                log_dir=log_dir,
            )

            elapsed = time.time() - t_start
            print(f"  Done in {elapsed:.1f}s | "
                  f"Final reward: {metrics.get('final_reward_mean', 0):.2f}, "
                  f"Collision: {metrics.get('final_collision_rate', 0):.2%}")
            seed_results[seed] = metrics

        all_results[exp_name] = seed_results

    return all_results


def save_results(all_results: dict, data_dir: Path):
    """Save all metrics as JSON for reproducibility."""
    data_dir.mkdir(parents=True, exist_ok=True)

    def _make_json_safe(obj):
        if isinstance(obj, dict):
            return {str(k): _make_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_make_json_safe(v) for v in obj]
        if isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return obj
        return obj

    serializable = {}
    for exp_name, seed_results in all_results.items():
        serializable[exp_name] = {}
        for seed, metrics in seed_results.items():
            serializable[exp_name][str(seed)] = _make_json_safe(metrics)

    with (data_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] Raw results: {data_dir / 'results.json'}")


def generate_plots(all_results: dict, plot_dir: Path):
    """Generate all paper-quality comparison plots."""
    set_style()
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Generating Plots ---")
    plot_reward_comparison(all_results, EXPERIMENTS, plot_dir / "01_reward_comparison.png")
    plot_collision_comparison(all_results, EXPERIMENTS, plot_dir / "02_collision_comparison.png")
    plot_loss_comparison(all_results, EXPERIMENTS, plot_dir / "03_loss_comparison.png")
    plot_safety_metrics(all_results, EXPERIMENTS, plot_dir / "04_safety_metrics.png")
    plot_final_performance_bar(all_results, EXPERIMENTS, plot_dir / "05_final_performance.png")
    plot_training_reward_curve(all_results, EXPERIMENTS, plot_dir / "06_training_reward.png")
    print(f"\n[Done] All plots saved to: {plot_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Run PPO+RSS experiments on highway-env.")
    p.add_argument("--experiments", type=str, default=",".join(ACTIVE_EXPERIMENTS),
                   help="Comma-separated experiment names to run.")
    p.add_argument("--seeds", type=str, default=",".join(map(str, SEEDS)),
                   help="Comma-separated random seeds.")
    p.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS,
                   help="Total training timesteps per experiment.")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "auto", "cuda"])
    p.add_argument("--verbose", type=int, default=0, help="SB3 verbosity (0=silent, 1=info).")
    p.add_argument("--skip-plots", action="store_true", help="Skip plot generation.")
    p.add_argument("--run-name", type=str, default=None,
                   help="Name for the output run directory. Defaults to a timestamp.")
    p.add_argument("--output-root", type=str, default=str(RUNS_DIR),
                   help="Root directory for versioned outputs.")
    return p.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    args = parse_args()
    experiments = [e.strip() for e in args.experiments.split(",") if e.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    run_dirs = _build_run_dirs(Path(args.output_root), run_name)

    print(f"{'#'*60}")
    print(f"PPO + RSS Dual-Layer Framework Experiments")
    print(f"  Experiments: {len(experiments)} ({', '.join(experiments)})")
    print(f"  Seeds: {len(seeds)} ({', '.join(map(str, seeds))})")
    print(f"  Timesteps per run: {args.timesteps}")
    print(f"  Total runs: {len(experiments) * len(seeds)}")
    print(f"  Output run: {run_dirs['run']}")
    print(f"{'#'*60}")

    t_total = time.time()
    all_results = run_experiments(
        experiments=experiments,
        seeds=seeds,
        total_timesteps=args.timesteps,
        device=args.device,
        verbose=args.verbose,
        model_dir=run_dirs["models"],
        log_dir=run_dirs["logs"],
    )

    save_results(all_results, run_dirs["data"])
    if not args.skip_plots:
        generate_plots(all_results, run_dirs["plots"])

    total_time = time.time() - t_total
    print(f"\n{'#'*60}")
    print(f"All experiments complete!")
    print(f"  Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"  Results: {run_dirs['data']}")
    print(f"  Plots: {run_dirs['plots']}")
    print(f"{'#'*60}")
