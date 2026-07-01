"""SCI-quality plotting for MoE Highway research results.

Generates publication-ready figures:
    Fig 1: Cross-expert performance comparison (4-panel)
    Fig 2: MoE gate behavior analysis (3-panel)
    Fig 3: Safety-efficiency Pareto frontier
    Fig 4: Architecture pipeline diagram

Usage:
    D:\\anaconda\\envs\\ppo_main\\python.exe tools/plot_moe_results.py
"""

import json
from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

# ============================================================
# Global style — SCI publication quality
# ============================================================
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.8,
    "lines.markersize": 7,
    "errorbar.capsize": 3,
})

SAVE_DIR = Path(__file__).resolve().parent.parent / "results" / "plots"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Consistent color palette (colorblind-friendly, from Tableau/Category10)
C_STACKELBERG = "#2c3e8c"   # deep blue
C_PPO_RSS = "#27ae60"        # green
C_MOE_HYBRID = "#e67e22"     # orange
C_IDM = "#95a5a6"            # gray
C_RANDOM = "#e74c3c"         # red
C_STACK_LIGHT = "#a8b8e0"
C_PPO_LIGHT = "#a3e4bc"
C_MOE_LIGHT = "#f5cba7"

DENSITIES = ["sparse", "medium", "dense"]
DENSITY_LABELS = ["Sparse\n(5v)", "Medium\n(15v)", "Dense\n(25v)"]
METHODS_COMPARE = ["Stackelberg", "PPO+RSS", "IDM_Baseline", "Random", "MoE_Hybrid"]
METHOD_LABELS = ["Stackelberg", "PPO+RSS", "IDM", "Random", "MoE Hybrid"]
METHOD_COLORS = {
    "Stackelberg": C_STACKELBERG,
    "PPO+RSS": C_PPO_RSS,
    "IDM_Baseline": C_IDM,
    "Random": C_RANDOM,
    "MoE_Hybrid": C_MOE_HYBRID,
}


def load_data():
    data_dir = Path(__file__).resolve().parent.parent / "results" / "data"
    with open(data_dir / "compare_experts.json") as f:
        compare = json.load(f)
    with open(data_dir / "moe_hybrid_eval.json") as f:
        moe = json.load(f)
    return compare, moe


def _agg(data: List[dict], key: str) -> Dict:
    """Aggregate: method -> density -> list of values."""
    out = {}
    for r in data:
        m = r["method"]
        d = r["density"]
        val = r.get(key)
        if val is None:
            continue
        out.setdefault(m, {}).setdefault(d, []).append(val)
    return out


def _mean_std(vals: List[float]):
    if not vals:
        return 0, 0
    return float(np.mean(vals)), float(np.std(vals))


# ============================================================
# Figure 1: Cross-Expert Performance Comparison (4-panel)
# ============================================================
def fig1_cross_expert_comparison(compare_data, moe_data):
    print("[Fig 1] Cross-expert performance comparison...")

    all_data = compare_data + moe_data
    methods = ["Stackelberg", "PPO+RSS", "MoE_Hybrid", "IDM_Baseline", "Random"]
    labels = ["Stackelberg", "PPO+RSS", "MoE\nHybrid", "IDM", "Random"]
    colors = [C_STACKELBERG, C_PPO_RSS, C_MOE_HYBRID, C_IDM, C_RANDOM]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- (a) Collision Rate ---
    ax = axes[0, 0]
    x = np.arange(len(DENSITIES))
    width = 0.15
    for i, (method, label, color) in enumerate(zip(methods, labels, colors)):
        means, stds = [], []
        for dkey, dlabel in zip(["sparse", "medium", "dense"], DENSITIES):
            vals = [r["crashed"] for r in all_data if r["method"] == method and r["density"].startswith(dkey)]
            rate = sum(vals) / max(len(vals), 1) * 100
            means.append(rate)
            stds.append(0)  # binary outcome, no meaningful std per bar
        bars = ax.bar(x + i * width, means, width, color=color, label=label, alpha=0.9, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Collision Rate (%)")
    ax.set_title("(a) Collision Rate by Traffic Density")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(DENSITY_LABELS)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper left", ncol=2, framealpha=0.9, fontsize=8)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(25))

    # --- (b) Average Speed ---
    ax = axes[0, 1]
    for i, (method, label, color) in enumerate(zip(methods, labels, colors)):
        means, stds = [], []
        for dkey in DENSITIES:
            vals = [r["avg_speed"] for r in all_data if r["method"] == method and r["density"].startswith(dkey)]
            m, s = _mean_std(vals)
            means.append(m)
            stds.append(s)
        ax.bar(x + i * width, means, width, yerr=stds, color=color, label=label, alpha=0.9, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Average Speed (m/s)")
    ax.set_title("(b) Average Speed by Traffic Density")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(DENSITY_LABELS)
    ax.legend(loc="upper left", ncol=2, framealpha=0.9, fontsize=8)

    # --- (c) Minimum TTC ---
    ax = axes[1, 0]
    for i, (method, label, color) in enumerate(zip(methods, labels, colors)):
        means, stds = [], []
        for dkey in DENSITIES:
            vals = []
            for r in all_data:
                if r["method"] == method and r["density"].startswith(dkey):
                    ttc = r.get("min_ttc", float("inf"))
                    if np.isfinite(ttc) and ttc < 100:
                        vals.append(ttc)
            m, s = _mean_std(vals)
            means.append(m)
            stds.append(s)
        ax.bar(x + i * width, means, width, yerr=stds, color=color, label=label, alpha=0.9, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Minimum TTC (s)")
    ax.set_title("(c) Minimum Time-to-Collision by Density")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(DENSITY_LABELS)
    ax.axhline(y=3.0, color=C_RANDOM, linestyle="--", linewidth=1.0, alpha=0.7, label="RSS threshold (3s)")
    ax.legend(loc="upper left", ncol=2, framealpha=0.9, fontsize=8)

    # --- (d) Lane Change Count ---
    ax = axes[1, 1]
    for i, (method, label, color) in enumerate(zip(methods, labels, colors)):
        means, stds = [], []
        for dkey in DENSITIES:
            vals = [r["lc_count"] for r in all_data if r["method"] == method and r["density"].startswith(dkey)]
            m, s = _mean_std(vals)
            means.append(m)
            stds.append(s)
        ax.bar(x + i * width, means, width, yerr=stds, color=color, label=label, alpha=0.9, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Lane Change Count")
    ax.set_title("(d) Lane Change Frequency by Density")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(DENSITY_LABELS)
    ax.legend(loc="upper right", ncol=2, framealpha=0.9, fontsize=8)

    fig.suptitle("Cross-Expert Performance Comparison", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = SAVE_DIR / "fig1_cross_expert_comparison.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Figure 2: MoE Gate Behavior Analysis (3-panel)
# ============================================================
def fig2_moe_gate_analysis(moe_data):
    print("[Fig 2] MoE gate behavior analysis...")

    fig = plt.figure(figsize=(16, 5))

    # --- (a) Expert usage stacked bar per density ---
    ax1 = fig.add_subplot(1, 3, 1)
    x = np.arange(len(DENSITIES))
    width = 0.5
    bottom_stack = np.zeros(len(DENSITIES))
    expert_keys = ["stackelberg", "ppo_rss", "rss_emergency"]
    expert_labels = ["Stackelberg", "PPO+RSS", "RSS Emergency"]
    expert_colors = [C_STACKELBERG, C_PPO_RSS, C_RANDOM]

    for ek, el, ec in zip(expert_keys, expert_labels, expert_colors):
        means = []
        for dkey in DENSITIES:
            vals = [r["expert_dist"].get(ek, 0) * 100 for r in moe_data if r["density"] == dkey]
            means.append(np.mean(vals) if vals else 0)
        ax1.bar(x, means, width, bottom=bottom_stack, color=ec, label=el, alpha=0.9, edgecolor="white", linewidth=0.5)
        bottom_stack += np.array(means)

    ax1.set_ylabel("Expert Usage (%)")
    ax1.set_title("(a) Expert Selection Distribution")
    ax1.set_xticks(x)
    ax1.set_xticklabels(DENSITY_LABELS)
    ax1.set_ylim(0, 105)
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=8)

    # --- (b) Stackelberg usage vs speed (scatter per seed) ---
    ax2 = fig.add_subplot(1, 3, 2)
    for dkey, marker, dlabel in zip(DENSITIES, ["o", "s", "^"], ["Sparse", "Medium", "Dense"]):
        pts = [(r["expert_dist"]["stackelberg"] * 100, r["avg_speed"])
               for r in moe_data if r["density"] == dkey]
        if pts:
            xs, ys = zip(*pts)
            ax2.scatter(xs, ys, marker=marker, s=80, alpha=0.8, label=dlabel, edgecolors="black", linewidth=0.5)
    ax2.set_xlabel("Stackelberg Usage (%)")
    ax2.set_ylabel("Average Speed (m/s)")
    ax2.set_title("(b) LC Expert Usage vs. Speed")
    ax2.legend(loc="best", framealpha=0.9, fontsize=8)

    # --- (c) Per-seed expert usage heatmap ---
    ax3 = fig.add_subplot(1, 3, 3)
    seeds = sorted(set(r["seed"] for r in moe_data))
    density_seed_data = {}
    for dkey in DENSITIES:
        row = []
        for seed in seeds:
            for r in moe_data:
                if r["density"] == dkey and r["seed"] == seed:
                    row.append(r["expert_dist"]["stackelberg"] * 100)
                    break
            else:
                row.append(0)
        density_seed_data[dkey] = row

    ax3.grid(False)
    rows = np.array([density_seed_data[d] for d in DENSITIES])
    im = ax3.imshow(rows, cmap="YlOrRd", aspect="auto", vmin=0, vmax=60)
    ax3.set_xticks(range(len(seeds)))
    ax3.set_xticklabels([str(s) for s in seeds])
    ax3.set_yticks(range(len(DENSITIES)))
    ax3.set_yticklabels(["Sparse", "Medium", "Dense"])
    ax3.set_xlabel("Seed")
    ax3.set_title("(c) Stackelberg Usage (%) Heatmap")
    for i in range(rows.shape[0]):
        for j in range(rows.shape[1]):
            ax3.text(j, i, f"{rows[i,j]:.0f}%", ha="center", va="center", fontsize=9,
                     color="white" if rows[i, j] > 30 else "black")
    cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label("% Stackelberg")

    fig.suptitle("MoE Hybrid Gate Behavior Analysis", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = SAVE_DIR / "fig2_moe_gate_analysis.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Figure 3: Safety-Efficiency Pareto Frontier
# ============================================================
def fig3_pareto_frontier(compare_data, moe_data):
    print("[Fig 3] Safety-efficiency Pareto frontier...")

    all_data = compare_data + moe_data
    fig, ax = plt.subplots(figsize=(9, 7))

    markers = {"Stackelberg": "D", "PPO+RSS": "s", "MoE_Hybrid": "P", "IDM_Baseline": "X", "Random": "o"}
    sizes = {"Stackelberg": 120, "PPO+RSS": 120, "MoE_Hybrid": 180, "IDM_Baseline": 80, "Random": 60}

    for method in ["Stackelberg", "PPO+RSS", "MoE_Hybrid", "IDM_Baseline", "Random"]:
        pts_speed, pts_ttc, pts_crash, pts_density = [], [], [], []
        for r in all_data:
            if r["method"] != method:
                continue
            ttc = r.get("min_ttc", float("inf"))
            if not np.isfinite(ttc) or ttc > 50:
                continue
            pts_speed.append(r["avg_speed"])
            pts_ttc.append(ttc)
            pts_crash.append("X" if r["crashed"] else "o")
            pts_density.append(DENSITIES.index(r["density"].split("(")[0]) if "(" in r["density"] else
                                DENSITIES.index(r["density"]) if r["density"] in DENSITIES else 0)

        if not pts_speed:
            continue

        # Non-crashed as filled, crashed as open with X
        crashed_mask = [c == "X" for c in pts_crash]
        safe_mask = [not c for c in crashed_mask]
        color = METHOD_COLORS[method]

        if any(safe_mask):
            sx = [pts_speed[i] for i, m in enumerate(safe_mask) if m]
            sy = [pts_ttc[i] for i, m in enumerate(safe_mask) if m]
            ax.scatter(sx, sy, marker=markers[method], s=sizes[method], c=color,
                       alpha=0.85, edgecolors="black", linewidth=0.5,
                       label=method.replace("_", " ") if method != "IDM_Baseline" else "IDM",
                       zorder=5 if method == "MoE_Hybrid" else 3)

        if any(crashed_mask):
            cx = [pts_speed[i] for i, m in enumerate(crashed_mask) if m]
            cy = [pts_ttc[i] for i, m in enumerate(crashed_mask) if m]
            ax.scatter(cx, cy, marker=markers[method], s=sizes[method], c="none",
                       edgecolors=color, linewidth=1.5, linestyle="--", alpha=0.5, zorder=2)

    # Pareto-optimal region shading
    ax.axvspan(22, 26, alpha=0.06, color=C_PPO_RSS, label="_")
    ax.axhspan(5, 15, alpha=0.06, color=C_STACKELBERG, label="_")
    # Ideal quadrant (top-right: fast + safe)
    ax.fill_between([22, 26], 5, 15, alpha=0.10, color=C_MOE_HYBRID, label="Ideal region\n(high speed + safe)")
    ax.text(24, 13, "IDEAL", fontsize=10, ha="center", color=C_MOE_HYBRID, alpha=0.6, fontweight="bold")

    ax.set_xlabel("Average Speed (m/s)")
    ax.set_ylabel("Minimum TTC (s)")
    ax.set_title("Safety-Efficiency Pareto Frontier")
    ax.axhline(y=3.0, color=C_RANDOM, linestyle="--", linewidth=1.0, alpha=0.5)
    ax.text(18.5, 3.1, "RSS emergency threshold", fontsize=8, color=C_RANDOM, alpha=0.7)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9, ncol=1,
              title="Filled=safe, Open=crashed", title_fontsize=8)
    ax.set_xlim(16, 27)
    ax.set_ylim(0, 18)

    fig.tight_layout()
    path = SAVE_DIR / "fig3_pareto_frontier.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Figure 4: Architecture Pipeline Diagram
# ============================================================
def fig4_architecture_diagram():
    print("[Fig 4] Architecture pipeline diagram...")

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Colors
    c_perception = "#3498db"
    c_game = C_STACKELBERG
    c_gate = "#9b59b6"
    c_ppo = C_PPO_RSS
    c_rss = C_RANDOM
    c_output = "#2c3e50"
    c_arrow = "#7f8c8d"
    c_box_bg = "#f8f9fa"

    def draw_box(x, y, w, h, text, color, fontsize=10, fontcolor="white", bold=False):
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                        facecolor=color, edgecolor="white", linewidth=2, alpha=0.9)
        ax.add_patch(rect)
        weight = "bold" if bold else "normal"
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
                color=fontcolor, fontweight=weight)

    def draw_arrow(x1, y1, x2, y2, color=c_arrow, lw=2.0, style="-"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw, linestyle=style))

    def draw_label(x, y, text, fontsize=9, color="#2c3e50", bold=False):
        weight = "bold" if bold else "normal"
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=color, fontweight=weight)

    # ---- Perception ----
    draw_box(0.5, 5.5, 2.5, 1.0, "Perception\n(highway-env)", c_perception, fontsize=11, bold=True)
    draw_label(1.75, 4.5, "Observation:\n140-dim Kinematics", fontsize=8, color=c_perception)

    # ---- Stackelberg Game Solver ----
    draw_box(4.5, 5.5, 3.0, 1.0, "Stackelberg\nGame Solver", c_game, fontsize=11, bold=True)
    draw_label(6.0, 4.5, "Scene Understanding\n18-candidate enumeration\nTTC · Gap · Cost", fontsize=8, color=c_game)

    # ---- MoE Gate ----
    draw_box(9.0, 5.5, 2.5, 1.0, "MoE Gate\n(Scene Classifier)", c_gate, fontsize=11, bold=True)
    draw_label(10.25, 4.5, "3-Tier Priority:\n[1] TTC<3s -> RSS\n[2] LC beneficial -> Stackelberg\n[3] Default -> PPO+RSS", fontsize=8, color=c_gate)

    # ---- Experts (bottom row) ----
    draw_box(1.0, 1.5, 3.5, 1.5, "Stackelberg Expert\nGame + FSM Governance\n→ Lane Change Decisions", c_game, fontsize=10)
    draw_box(6.0, 1.5, 3.5, 1.5, "PPO+RSS Expert\nTrained Policy + RSS Shield\n→ Speed Optimization", c_ppo, fontsize=10)
    draw_box(11.0, 1.5, 3.5, 1.5, "RSS Emergency\nSafety Envelope\n→ Collision Avoidance", c_rss, fontsize=10)

    # ---- Output ----
    draw_box(5.0, 0.0, 5.0, 0.8, "Action Output → highway-env step()", c_output, fontsize=11, bold=True)

    # ---- Arrows ----
    draw_arrow(3.0, 6.0, 4.5, 6.0)  # Perception → Game Solver
    draw_arrow(7.5, 6.0, 9.0, 6.0)  # Game Solver → Gate

    # Gate → Experts
    draw_arrow(10.25, 5.5, 2.75, 3.0, c_gate, 1.5, "--")   # Gate → Stackelberg
    draw_arrow(10.25, 5.5, 7.75, 3.0, c_gate, 1.5, "--")   # Gate → PPO+RSS
    draw_arrow(10.25, 5.5, 12.75, 3.0, c_gate, 1.5, "--")  # Gate → RSS

    # Experts → Output
    draw_arrow(2.75, 1.5, 7.5, 0.8, c_arrow, 1.2)
    draw_arrow(7.75, 1.5, 7.5, 0.8, c_arrow, 1.2)
    draw_arrow(12.75, 1.5, 7.5, 0.8, c_arrow, 1.2)

    # Scene features feedback (dashed)
    ax.annotate("scene features", xy=(1.75, 6.5), xytext=(3.0, 7.3),
                ha="center", fontsize=8, color=c_arrow,
                arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1, linestyle=":", connectionstyle="arc3,rad=-0.3"))

    ax.set_title("MoE Highway: Mixture-of-Experts Decision Architecture", fontsize=16, fontweight="bold", pad=15)

    fig.tight_layout()
    path = SAVE_DIR / "fig4_architecture_diagram.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Figure 5: Summary Performance Table (as a matplotlib figure)
# ============================================================
def fig5_summary_table(compare_data, moe_data):
    print("[Fig 5] Summary performance table...")

    all_data = compare_data + moe_data
    methods_display = [
        ("Stackelberg", "Stackelberg"),
        ("PPO+RSS", "PPO+RSS"),
        ("MoE_Hybrid", "MoE Hybrid"),
        ("IDM_Baseline", "IDM (No Safety)"),
        ("Random", "Random"),
    ]

    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.axis("off")

    col_labels = ["Method", "Collision\nRate", "Avg Speed\n(m/s)", "Min TTC\n(s)",
                  "Min Gap\n(m)", "LC Count", "Key Characteristic"]
    rows = []

    for mkey, mlabel in methods_display:
        entries = [r for r in all_data if r["method"] == mkey]
        if not entries:
            continue
        n = len(entries)
        crash_rate = sum(1 for r in entries if r["crashed"]) / n * 100
        speed_vals = [r["avg_speed"] for r in entries]
        ttc_vals = [r.get("min_ttc", float("inf")) for r in entries if np.isfinite(r.get("min_ttc", float("inf")))]
        gap_vals = [r.get("min_gap", float("inf")) for r in entries if np.isfinite(r.get("min_gap", float("inf")))]
        lc_vals = [r.get("lc_count", 0) for r in entries]

        speed_str = f"{np.mean(speed_vals):.1f}±{np.std(speed_vals):.1f}"
        ttc_str = f"{np.mean(ttc_vals):.1f}±{np.std(ttc_vals):.1f}" if ttc_vals else "—"
        gap_str = f"{np.mean(gap_vals):.1f}±{np.std(gap_vals):.1f}" if gap_vals else "—"
        crash_str = f"{crash_rate:.0f}%"
        lc_str = f"{np.mean(lc_vals):.1f}±{np.std(lc_vals):.1f}"

        if mkey == "Stackelberg":
            char = "Strategic lane changes, very safe, conservative speed in dense traffic"
        elif mkey == "PPO+RSS":
            char = "Maximum speed within RSS bounds, never lane changes, tight following"
        elif mkey == "MoE_Hybrid":
            char = "Adaptive: LC when beneficial, speed otherwise, highest avg speed"
        elif mkey == "IDM_Baseline":
            char = "Constant speed, no collision avoidance, crashes in all but sparsest traffic"
        else:
            char = "Uniform random, high crash rate, no strategic behavior"

        rows.append([mlabel, crash_str, speed_str, ttc_str, gap_str, lc_str, char])

    table = ax.table(cellText=rows, colLabels=col_labels, cellLoc="center", loc="center",
                     colWidths=[0.12, 0.08, 0.11, 0.10, 0.10, 0.09, 0.40])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    # Style: highlight MoE Hybrid row
    for i in range(len(rows)):
        for j in range(len(col_labels)):
            cell = table[i + 1, j]
            if rows[i][0] == "MoE Hybrid":
                cell.set_facecolor("#fef3e8")
                cell.set_text_props(fontweight="bold")

    # Header styling
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    ax.set_title("Expert Method Comparison Summary", fontsize=14, fontweight="bold", pad=20)

    fig.tight_layout()
    path = SAVE_DIR / "fig5_summary_table.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Figure 6: Density-Speed-Crash 3D view
# ============================================================
def fig6_speed_profile(compare_data, moe_data):
    print("[Fig 6] Speed profile across densities...")

    all_data = compare_data + moe_data
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    methods = ["Stackelberg", "PPO+RSS", "MoE_Hybrid"]
    colors = [C_STACKELBERG, C_PPO_RSS, C_MOE_HYBRID]
    markers = ["D", "s", "P"]
    densities_display = ["sparse", "medium", "dense"]
    x_pos = [0, 1, 2]

    # --- (a) Speed degradation curve ---
    for method, color, marker in zip(methods, colors, markers):
        means, stds = [], []
        for dkey in densities_display:
            vals = [r["avg_speed"] for r in all_data if r["method"] == method and r["density"].startswith(dkey)]
            m, s = _mean_std(vals)
            means.append(m)
            stds.append(s)
        ax1.errorbar(x_pos, means, yerr=stds, color=color, marker=marker, markersize=10,
                     linewidth=2.2, capsize=5, label=method.replace("_", " "), markeredgecolor="black",
                     markeredgewidth=0.5)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(DENSITY_LABELS)
    ax1.set_ylabel("Average Speed (m/s)")
    ax1.set_title("(a) Speed Degradation with Traffic Density")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.set_ylim(15, 27)

    # --- (b) MoE Hybrid speed gain over individual experts ---
    # Compute percentage gain for each density
    for i, dkey in enumerate(densities_display):
        stack_speeds = [r["avg_speed"] for r in all_data if r["method"] == "Stackelberg" and r["density"].startswith(dkey)]
        ppo_speeds = [r["avg_speed"] for r in all_data if r["method"] == "PPO+RSS" and r["density"].startswith(dkey)]
        moe_speeds = [r["avg_speed"] for r in all_data if r["method"] == "MoE_Hybrid" and r["density"] == dkey]

        stack_m = np.mean(stack_speeds) if stack_speeds else 0
        ppo_m = np.mean(ppo_speeds) if ppo_speeds else 0
        moe_m = np.mean(moe_speeds) if moe_speeds else 0

        gain_stack = (moe_m - stack_m) / stack_m * 100 if stack_m > 0 else 0
        gain_ppo = (moe_m - ppo_m) / ppo_m * 100 if ppo_m > 0 else 0

        bar_width = 0.25
        ax2.bar(i - bar_width / 2, gain_stack, bar_width, color=C_STACKELBERG, alpha=0.85,
                label="vs Stackelberg" if i == 0 else "", edgecolor="white")
        ax2.bar(i + bar_width / 2, gain_ppo, bar_width, color=C_PPO_RSS, alpha=0.85,
                label="vs PPO+RSS" if i == 0 else "", edgecolor="white")

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(DENSITY_LABELS)
    ax2.set_ylabel("Speed Improvement (%)")
    ax2.set_title("(b) MoE Hybrid Speed Gain over Individual Experts")
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax2.legend(loc="upper left", framealpha=0.9)

    fig.suptitle("Speed Profile Analysis", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = SAVE_DIR / "fig6_speed_profile.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
def main():
    print("=" * 60)
    print("MoE Highway — SCI-Quality Result Plotting")
    print("=" * 60)

    compare_data, moe_data = load_data()
    print(f"Loaded {len(compare_data)} comparison + {len(moe_data)} MoE hybrid records")

    fig1_cross_expert_comparison(compare_data, moe_data)
    fig2_moe_gate_analysis(moe_data)
    fig3_pareto_frontier(compare_data, moe_data)
    fig4_architecture_diagram()
    fig5_summary_table(compare_data, moe_data)
    fig6_speed_profile(compare_data, moe_data)

    print()
    print("=" * 60)
    print(f"All figures saved to: {SAVE_DIR.resolve()}")
    print("Done.")


if __name__ == "__main__":
    main()
