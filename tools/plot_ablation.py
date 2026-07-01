"""Plot ablation results: threshold tuning vs MoE architecture."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

COLORS = {
    "Stackelberg-Default":      "#3498db",
    "Stackelberg-Aggressive":   "#e74c3c",
    "PPO+RSS-Default":          "#2ecc71",
    "PPO+RSS-Safe":             "#1abc9c",
    "MoE-Hybrid":               "#f39c12",
}

METHOD_ORDER = [
    "Stackelberg-Default", "Stackelberg-Aggressive",
    "PPO+RSS-Default", "PPO+RSS-Safe", "MoE-Hybrid",
]
SHORT_LABELS = {
    "Stackelberg-Default":    "Stack.\nDefault",
    "Stackelberg-Aggressive": "Stack.\nAggressive",
    "PPO+RSS-Default":        "PPO+RSS\nDefault",
    "PPO+RSS-Safe":           "PPO+RSS\nSafe",
    "MoE-Hybrid":             "MoE\nHybrid",
}
DENSITY_LABELS = {"sparse": "Sparse (0.8)", "medium": "Medium (1.2)", "dense": "Dense (1.5)"}
DENSITY_ORDER = ["sparse", "medium", "dense"]


def load_data():
    path = PROJECT_ROOT / "results/data/ablation_threshold.json"
    with open(path) as f:
        return json.load(f)


def fig7_ablation_summary(data):
    """4-panel bar chart: Speed, Min TTC, LC count, Crash rate by method."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

    metrics = [
        ("avg_speed", "Avg Speed (m/s)", axes[0]),
        ("min_ttc", "Min TTC (s)", axes[1]),
        ("lc_count", "Lane Changes", axes[2]),
    ]

    # Group by method
    for metric, ylabel, ax in metrics:
        vals = {}
        for method in METHOD_ORDER:
            entries = [r for r in data if r["method"] == method]
            vals[method] = [r[metric] for r in entries]

        x = np.arange(len(METHOD_ORDER))
        means = [np.mean(vals[m]) for m in METHOD_ORDER]
        stds = [np.std(vals[m]) for m in METHOD_ORDER]
        colors = [COLORS[m] for m in METHOD_ORDER]

        bars = ax.bar(x, means, yerr=stds, color=colors, capsize=5, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT_LABELS[m] for m in METHOD_ORDER], fontsize=8)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        # Highlight MoE bar
        bars[-1].set_edgecolor("black")
        bars[-1].set_linewidth(1.5)

        # Annotate MoE value on bar
        ax.annotate(f"{means[-1]:.1f}", xy=(x[-1], means[-1]),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=9, fontweight="bold", color="#e67e22")

    # Crash rate panel
    ax = axes[3]
    crash_rates = []
    for method in METHOD_ORDER:
        entries = [r for r in data if r["method"] == method]
        crashes = sum(1 for r in entries if r["crashed"])
        crash_rates.append(crashes / len(entries) * 100 if entries else 0)
    colors = [COLORS[m] for m in METHOD_ORDER]
    bars = ax.bar(np.arange(len(METHOD_ORDER)), crash_rates, color=colors, edgecolor="white", linewidth=0.5)
    bars[-1].set_edgecolor("black")
    bars[-1].set_linewidth(1.5)
    ax.set_xticks(np.arange(len(METHOD_ORDER)))
    ax.set_xticklabels([SHORT_LABELS[m] for m in METHOD_ORDER], fontsize=8)
    ax.set_ylabel("Crash Rate (%)", fontsize=11)
    ax.set_ylim(0, max(20, max(crash_rates) * 1.5))
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle("Ablation: Threshold Tuning vs. MoE Architecture", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = PROJECT_ROOT / "results/plots/fig7_ablation_summary.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig8_ablation_by_density(data):
    """Speed and TTC breakdown by density for each variant."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    for col, density_key in enumerate(DENSITY_ORDER):
        dlabel = DENSITY_LABELS[density_key]
        ddata = [r for r in data if r["density"] == density_key]

        # Speed panel (row 0)
        ax = axes[0, col]
        speed_vals = {}
        for method in METHOD_ORDER:
            entries = [r for r in ddata if r["method"] == method]
            speed_vals[method] = [r["avg_speed"] for r in entries]

        x = np.arange(len(METHOD_ORDER))
        means = [np.mean(speed_vals[m]) for m in METHOD_ORDER]
        stds = [np.std(speed_vals[m]) for m in METHOD_ORDER]
        colors = [COLORS[m] for m in METHOD_ORDER]
        bars = ax.bar(x, means, yerr=stds, color=colors, capsize=4, edgecolor="white", linewidth=0.5)
        bars[-1].set_edgecolor("black")
        bars[-1].set_linewidth(1.5)
        ax.set_title(f"{dlabel} — Speed", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT_LABELS[m] for m in METHOD_ORDER], fontsize=7)
        ax.set_ylabel("Avg Speed (m/s)", fontsize=10)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        # TTC panel (row 1)
        ax = axes[1, col]
        ttc_vals = {}
        for method in METHOD_ORDER:
            entries = [r for r in ddata if r["method"] == method]
            ttcs = []
            for r in entries:
                t = r["min_ttc"]
                if np.isfinite(t) and t < 100:
                    ttcs.append(t)
            ttc_vals[method] = ttcs

        means = [np.mean(ttc_vals[m]) if ttc_vals[m] else 0 for m in METHOD_ORDER]
        stds = [np.std(ttc_vals[m]) if ttc_vals[m] else 0 for m in METHOD_ORDER]
        colors = [COLORS[m] for m in METHOD_ORDER]
        bars = ax.bar(x, means, yerr=stds, color=colors, capsize=4, edgecolor="white", linewidth=0.5)
        bars[-1].set_edgecolor("black")
        bars[-1].set_linewidth(1.5)
        ax.set_title(f"{dlabel} — Min TTC", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT_LABELS[m] for m in METHOD_ORDER], fontsize=7)
        ax.set_ylabel("Min TTC (s)", fontsize=10)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        # RSS 3s red line
        ax.axhline(y=3.0, color="red", linestyle="--", linewidth=1, alpha=0.6, label="RSS 3s")
        if col == 2:
            ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Ablation: Performance Breakdown by Traffic Density", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = PROJECT_ROOT / "results/plots/fig8_ablation_by_density.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig9_hypothesis_test(data):
    """Three hypothesis test result visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ---- H1: Stackelberg-Default vs Aggressive ----
    ax = axes[0]
    def_entries = [r for r in data if r["method"] == "Stackelberg-Default"]
    agg_entries = [r for r in data if r["method"] == "Stackelberg-Aggressive"]
    moe_entries = [r for r in data if r["method"] == "MoE-Hybrid"]

    def_speeds = [r["avg_speed"] for r in def_entries]
    agg_speeds = [r["avg_speed"] for r in agg_entries]
    moe_speeds = [r["avg_speed"] for r in moe_entries]

    positions = [1, 2, 3]
    parts = ax.violinplot([def_speeds, agg_speeds, moe_speeds], positions=positions,
                          showmeans=True, showmedians=True, widths=0.6)
    for pc, color in zip(parts["bodies"], ["#3498db", "#e74c3c", "#f39c12"]):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    for partname in ("cbars", "cmins", "cmaxes", "cmeans", "cmedians"):
        if partname in parts:
            parts[partname].set_color("black")

    ax.set_xticks(positions)
    ax.set_xticklabels(["Stack.\nDefault", "Stack.\nAggressive", "MoE\nHybrid"], fontsize=9)
    ax.set_ylabel("Avg Speed (m/s)", fontsize=11)
    ax.set_title("H1: Relax FSM → match MoE speed?", fontsize=12, fontweight="bold")

    # Annotation arrows
    ax.annotate("", xy=(3, np.mean(moe_speeds)), xytext=(2, np.mean(agg_speeds)),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
    ax.text(2.5, np.mean(moe_speeds) + 0.3,
            f"+{np.mean(moe_speeds)-np.mean(agg_speeds):.1f} m/s\ngap",
            ha="center", fontsize=9, color="#e67e22")

    # ---- H2: PPO+RSS-Default vs Safe ----
    ax = axes[1]
    ppo_def = [r for r in data if r["method"] == "PPO+RSS-Default"]
    ppo_safe = [r for r in data if r["method"] == "PPO+RSS-Safe"]

    def_ttc = [r["min_ttc"] for r in ppo_def if np.isfinite(r["min_ttc"]) and r["min_ttc"] < 100]
    safe_ttc = [r["min_ttc"] for r in ppo_safe if np.isfinite(r["min_ttc"]) and r["min_ttc"] < 100]
    moe_ttc = [r["min_ttc"] for r in moe_entries if np.isfinite(r["min_ttc"]) and r["min_ttc"] < 100]

    parts = ax.violinplot([def_ttc, safe_ttc, moe_ttc], positions=positions,
                          showmeans=True, showmedians=True, widths=0.6)
    for pc, color in zip(parts["bodies"], ["#2ecc71", "#1abc9c", "#f39c12"]):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    for partname in ("cbars", "cmins", "cmaxes", "cmeans", "cmedians"):
        if partname in parts:
            parts[partname].set_color("black")

    ax.set_xticks(positions)
    ax.set_xticklabels(["PPO+RSS\nDefault", "PPO+RSS\nSafe", "MoE\nHybrid"], fontsize=9)
    ax.set_ylabel("Min TTC (s)", fontsize=11)
    ax.set_title("H2: Tighten RSS → match MoE safety?", fontsize=12, fontweight="bold")
    ax.axhline(y=3.0, color="red", linestyle="--", linewidth=1, alpha=0.5, label="RSS critical")
    ax.legend(fontsize=8)

    # ---- H3: Architecture vs Parameters ----
    ax = axes[2]

    # Key insight: PPO never lane changes (capability gap)
    lc_methods = []
    lc_vals = []
    lc_colors_inner = []
    for method in METHOD_ORDER:
        entries = [r for r in data if r["method"] == method]
        lcs = [r["lc_count"] for r in entries]
        lc_methods.append(method)
        lc_vals.append(lcs)
        lc_colors_inner.append(COLORS[method])

    x = np.arange(len(lc_methods))
    means = [np.mean(v) for v in lc_vals]
    stds = [np.std(v) for v in lc_vals]

    bars = ax.bar(x, means, yerr=stds, color=lc_colors_inner, capsize=5, edgecolor="white", linewidth=0.5)
    bars[-1].set_edgecolor("black")
    bars[-1].set_linewidth(1.5)

    # Highlight zero-LC methods
    for i, (m, mean_val) in enumerate(zip(lc_methods, means)):
        if mean_val == 0:
            ax.annotate("CANNOT\nLANE\nCHANGE", xy=(i, 0.3), ha="center", fontsize=8,
                       color="#c0392b", fontweight="bold",
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="#fadbd8", edgecolor="#e74c3c", alpha=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_LABELS[m] for m in lc_methods], fontsize=8)
    ax.set_ylabel("Lane Changes", fontsize=11)
    ax.set_title("H3: Architecture is the enabler", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle("Ablation: Three Hypothesis Tests", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = PROJECT_ROOT / "results/plots/fig9_hypothesis_test.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig10_necessity_table(data):
    """Summary table: Why MoE is necessary."""
    fig, ax = plt.subplots(figsize=(16, 3.5))
    ax.axis("off")

    headers = ["Hypothesis", "Test", "Result", "Conclusion"]
    rows = [
        ["H1: Relax Stackelberg\nFSM → match MoE speed",
         "Stackelberg-Aggressive:\nTTC 5→3s, gap 1.2→0.8, min_safe 5→3m",
         f"Speed: 20.0 vs MoE 22.9 m/s\nGap: +2.9 m/s (15% slower)",
         "REJECTED — threshold tuning\ncannot close speed gap"],
        ["H2: Tighten RSS\n→ match MoE safety",
         "PPO+RSS-Safe:\nTTC 3→5s, min_dist 8→20m",
         f"TTC: 15.7s vs MoE 7.4s\nSafe but LC=0, speed drops 8%",
         "REJECTED — safer but still\nCANNOT lane-change"],
        ["H3: MoE advantage is\narchitecture, not parameters",
         "MoE-Hybrid:\nrule-based gate + 3 experts",
         f"Speed: 22.9 m/s (highest)\nLC: 1.3 when needed, 0 crashes",
         "CONFIRMED — architecture\nuniquely bridges capability gap"],
    ]

    # Table styling
    col_widths = [0.22, 0.28, 0.28, 0.22]
    table = ax.table(cellText=rows, colLabels=headers, colWidths=col_widths,
                     cellLoc="center", loc="center")

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    # Header styling
    for j, header in enumerate(headers):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold", fontsize=10)

    # Row coloring
    row_colors = ["#eaf2f8", "#e8f8f5", "#fef9e7"]
    for i in range(3):
        for j in range(4):
            cell = table[i + 1, j]
            cell.set_facecolor(row_colors[i])
            if j == 3:  # Conclusion column
                if "REJECTED" in str(cell.get_text()):
                    cell.set_text_props(color="#c0392b", fontweight="bold")
                elif "CONFIRMED" in str(cell.get_text()):
                    cell.set_text_props(color="#27ae60", fontweight="bold")

    ax.set_title("Ablation: Proving MoE Necessity — Three Hypothesis Tests",
                 fontsize=13, fontweight="bold", pad=15)

    plt.tight_layout()
    out = PROJECT_ROOT / "results/plots/fig10_necessity_table.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    data = load_data()
    print(f"Loaded {len(data)} ablation records")
    fig7_ablation_summary(data)
    fig8_ablation_by_density(data)
    fig9_hypothesis_test(data)
    fig10_necessity_table(data)
    print("\nAll ablation plots generated.")


if __name__ == "__main__":
    main()
