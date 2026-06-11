"""Paper-quality plotting for PPO + RSS experiment results."""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def set_style():
    """Apply publication-ready matplotlib style."""
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _common_x(seed_results: Dict[int, dict], key: str) -> np.ndarray:
    """Find a common x-axis across all seeds by taking the union of eval timesteps."""
    all_x = set()
    for metrics in seed_results.values():
        for x in metrics.get(key, []):
            all_x.add(int(x))
    return np.array(sorted(all_x), dtype=float)


def _interp_at(x_vals: np.ndarray, x_data: List, y_data: List) -> np.ndarray:
    """Interpolate y_data at x_vals, extrapolate with edge values."""
    if len(x_data) == 0:
        return np.full_like(x_vals, np.nan)
    return np.interp(x_vals, x_data, y_data, left=np.nan, right=np.nan)


def _aggregate_eval(seed_results: Dict[int, dict], y_key: str, x_key: str = "eval_timesteps"):
    """Aggregate eval metrics across seeds: returns (x_vals, mean_y, std_y)."""
    x_vals = _common_x(seed_results, x_key)
    if len(x_vals) == 0:
        return np.array([]), np.array([]), np.array([])

    all_curves = []
    for metrics in seed_results.values():
        x_data = metrics.get(x_key, [])
        y_data = metrics.get(y_key, [])
        if len(x_data) > 0:
            all_curves.append(_interp_at(x_vals, x_data, y_data))

    if not all_curves:
        return np.array([]), np.array([]), np.array([])

    stacked = np.array(all_curves)
    mean_y = np.nanmean(stacked, axis=0)
    std_y = np.nanstd(stacked, axis=0)
    return x_vals, mean_y, std_y


def plot_reward_comparison(
    all_results: dict,
    experiment_configs: dict,
    save_path: Path,
):
    """Multi-experiment eval reward comparison with error bands (std across seeds)."""
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for exp_name, exp_cfg in experiment_configs.items():
        if exp_name not in all_results:
            continue
        x, mean_y, std_y = _aggregate_eval(
            all_results[exp_name], y_key="eval_reward_mean", x_key="eval_timesteps"
        )
        if len(x) == 0:
            continue
        ax.plot(x, mean_y, color=exp_cfg["color"], linestyle=exp_cfg["linestyle"],
                marker=exp_cfg["marker"], markersize=5, label=exp_cfg["label"], linewidth=1.8)
        ax.fill_between(x, mean_y - std_y, mean_y + std_y,
                        color=exp_cfg["color"], alpha=0.12)

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean Eval Reward")
    ax.set_title("Reward Convergence Comparison")
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[Plot] Saved: {save_path}")


def plot_collision_comparison(
    all_results: dict,
    experiment_configs: dict,
    save_path: Path,
):
    """Multi-experiment collision rate comparison."""
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for exp_name, exp_cfg in experiment_configs.items():
        if exp_name not in all_results:
            continue
        x, mean_y, std_y = _aggregate_eval(
            all_results[exp_name], y_key="eval_collision_rate", x_key="eval_timesteps"
        )
        if len(x) == 0:
            continue
        mean_y_pct = np.array(mean_y) * 100.0
        std_y_pct = np.array(std_y) * 100.0
        ax.plot(x, mean_y_pct, color=exp_cfg["color"], linestyle=exp_cfg["linestyle"],
                marker=exp_cfg["marker"], markersize=5, label=exp_cfg["label"], linewidth=1.8)
        ax.fill_between(x, mean_y_pct - std_y_pct, mean_y_pct + std_y_pct,
                        color=exp_cfg["color"], alpha=0.12)

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Collision Rate (%)")
    ax.set_title("Collision Rate Comparison")
    ax.set_ylim(0, None)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[Plot] Saved: {save_path}")


def plot_loss_comparison(
    all_results: dict,
    experiment_configs: dict,
    save_path: Path,
):
    """Multi-experiment training loss comparison."""
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for exp_name, exp_cfg in experiment_configs.items():
        if exp_name not in all_results:
            continue
        # Aggregate loss across seeds
        all_loss_x = set()
        for m in all_results[exp_name].values():
            for lx in m.get("loss_curve_x", []):
                all_loss_x.add(int(lx))
        common_x = np.array(sorted(all_loss_x), dtype=float)
        if len(common_x) == 0:
            continue

        curves = []
        for m in all_results[exp_name].values():
            lx = m.get("loss_curve_x", [])
            ly = m.get("loss_curve_y", [])
            if len(lx) > 0:
                curves.append(np.interp(common_x, lx, ly, left=np.nan, right=np.nan))
        if not curves:
            continue
        mean_loss = np.nanmean(curves, axis=0)
        std_loss = np.nanstd(curves, axis=0)

        ax.plot(common_x, mean_loss, color=exp_cfg["color"], linestyle=exp_cfg["linestyle"],
                label=exp_cfg["label"], linewidth=1.5)
        ax.fill_between(common_x, mean_loss - std_loss, mean_loss + std_loss,
                        color=exp_cfg["color"], alpha=0.12)

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Training Loss")
    ax.set_title("Loss Curve Comparison")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[Plot] Saved: {save_path}")


def plot_safety_metrics(
    all_results: dict,
    experiment_configs: dict,
    save_path: Path,
):
    """2x2 safety metric panel: TTC, intervention rate, min distance, training efficiency."""
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    rss_experiments = [n for n, c in experiment_configs.items() if c.get("use_rss") and n in all_results]

    # (a) Min TTC
    ax = axes[0, 0]
    for exp_name, exp_cfg in experiment_configs.items():
        if exp_name not in all_results or not exp_cfg.get("use_rss"):
            continue
        x, mean_y, std_y = _aggregate_eval(all_results[exp_name], "eval_min_ttc", "eval_timesteps")
        if len(x) == 0:
            continue
        ax.plot(x, mean_y, color=exp_cfg["color"], linestyle=exp_cfg["linestyle"],
                marker=exp_cfg["marker"], markersize=4, label=exp_cfg["label"], linewidth=1.8)
        ax.fill_between(x, mean_y - std_y, mean_y + std_y, color=exp_cfg["color"], alpha=0.12)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Min TTC (s)")
    ax.set_title("(a) Minimum Time-to-Collision")
    if rss_experiments:
        ax.legend(loc="upper left", framealpha=0.9)

    # (b) Intervention rate
    ax = axes[0, 1]
    for exp_name, exp_cfg in experiment_configs.items():
        if exp_name not in all_results or not exp_cfg.get("use_rss"):
            continue
        x, mean_y, std_y = _aggregate_eval(all_results[exp_name], "eval_intervention_rate", "eval_timesteps")
        if len(x) == 0:
            continue
        mean_pct = np.array(mean_y) * 100.0
        std_pct = np.array(std_y) * 100.0
        ax.plot(x, mean_pct, color=exp_cfg["color"], linestyle=exp_cfg["linestyle"],
                marker=exp_cfg["marker"], markersize=4, label=exp_cfg["label"], linewidth=1.8)
        ax.fill_between(x, mean_pct - std_pct, mean_pct + std_pct, color=exp_cfg["color"], alpha=0.12)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Intervention Rate (%)")
    ax.set_title("(b) RSS Intervention Rate")
    ax.set_ylim(0, None)
    if rss_experiments:
        ax.legend(loc="upper right", framealpha=0.9)

    # (c) Min safe distance
    ax = axes[1, 0]
    for exp_name, exp_cfg in experiment_configs.items():
        if exp_name not in all_results or not exp_cfg.get("use_rss"):
            continue
        x, mean_y, std_y = _aggregate_eval(all_results[exp_name], "eval_min_distance", "eval_timesteps")
        if len(x) == 0:
            continue
        ax.plot(x, mean_y, color=exp_cfg["color"], linestyle=exp_cfg["linestyle"],
                marker=exp_cfg["marker"], markersize=4, label=exp_cfg["label"], linewidth=1.8)
        ax.fill_between(x, mean_y - std_y, mean_y + std_y, color=exp_cfg["color"], alpha=0.12)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Min Distance (m)")
    ax.set_title("(c) Minimum Inter-Vehicle Distance")
    if rss_experiments:
        ax.legend(loc="upper left", framealpha=0.9)

    # (d) Wall time (training efficiency)
    ax = axes[1, 1]
    labels = []
    times_mean = []
    times_std = []
    colors = []
    for exp_name in all_results:
        exp_cfg = experiment_configs[exp_name]
        wall_times = [m.get("wall_time_seconds", 0) for m in all_results[exp_name].values()]
        labels.append(exp_cfg["label"])
        times_mean.append(np.mean(wall_times))
        times_std.append(np.std(wall_times))
        colors.append(exp_cfg["color"])

    bars = ax.bar(labels, times_mean, yerr=times_std, color=colors, capsize=5, alpha=0.85)
    ax.set_ylabel("Wall Time (s)")
    ax.set_title("(d) Training Efficiency")
    ax.tick_params(axis="x", rotation=15, labelsize=8)

    plt.subplots_adjust(hspace=0.35, wspace=0.3)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[Plot] Saved: {save_path}")


def plot_final_performance_bar(
    all_results: dict,
    experiment_configs: dict,
    save_path: Path,
):
    """Bar chart: final reward and collision rate across experiments."""
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    exp_order = list(all_results.keys())
    labels = []
    reward_mean = []
    reward_std = []
    collision_mean = []
    collision_std = []
    colors = []

    for exp_name in exp_order:
        if exp_name not in all_results:
            continue
        exp_cfg = experiment_configs[exp_name]
        seeds = all_results[exp_name]
        r_means = [m.get("final_reward_mean", 0) for m in seeds.values()]
        c_rates = [m.get("final_collision_rate", 0) * 100.0 for m in seeds.values()]
        labels.append(exp_cfg["label"])
        reward_mean.append(np.mean(r_means))
        reward_std.append(np.std(r_means))
        collision_mean.append(np.mean(c_rates))
        collision_std.append(np.std(c_rates))
        colors.append(exp_cfg["color"])

    # Final reward
    ax1.bar(labels, reward_mean, yerr=reward_std, color=colors, capsize=6, alpha=0.85)
    ax1.set_ylabel("Final Mean Reward")
    ax1.set_title("Final Reward Comparison")
    ax1.tick_params(axis="x", rotation=15, labelsize=8)

    # Final collision rate
    ax2.bar(labels, collision_mean, yerr=collision_std, color=colors, capsize=6, alpha=0.85)
    ax2.set_ylabel("Collision Rate (%)")
    ax2.set_title("Final Collision Rate Comparison")
    ax2.set_ylim(0, None)
    ax2.tick_params(axis="x", rotation=15, labelsize=8)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[Plot] Saved: {save_path}")


def plot_training_reward_curve(
    all_results: dict,
    experiment_configs: dict,
    save_path: Path,
):
    """Per-episode training reward curve (moving average, not eval)."""
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for exp_name, exp_cfg in experiment_configs.items():
        if exp_name not in all_results:
            continue
        all_rx = set()
        for m in all_results[exp_name].values():
            for rx in m.get("reward_curve_x", []):
                all_rx.add(int(rx))
        common_x = np.array(sorted(all_rx), dtype=float)
        if len(common_x) == 0:
            continue

        curves = []
        for m in all_results[exp_name].values():
            rx = m.get("reward_curve_x", [])
            ry = m.get("reward_curve_y", [])
            if len(rx) > 0:
                curves.append(np.interp(common_x, rx, ry, left=np.nan, right=np.nan))
        if not curves:
            continue
        mean_r = np.nanmean(curves, axis=0)
        std_r = np.nanstd(curves, axis=0)

        ax.plot(common_x, mean_r, color=exp_cfg["color"], linestyle=exp_cfg["linestyle"],
                label=exp_cfg["label"], linewidth=1.5)
        ax.fill_between(common_x, mean_r - std_r, mean_r + std_r,
                        color=exp_cfg["color"], alpha=0.12)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Training Reward (Moving Avg)")
    ax.set_title("Training Reward Curves")
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[Plot] Saved: {save_path}")
