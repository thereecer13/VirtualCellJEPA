"""
plot_results.py
===============
Reads CellJEPA results files and produces a grouped bar chart comparing
NMI / ARI / ASW / AvgBIO across all experimental conditions.

Usage:
    python plot_results.py [--results_dir DIR] [--out FILE]

Defaults:
    --results_dir  .   (project root)
    --out          results_comparison.png
"""

import argparse
import re
import os
import matplotlib.pyplot as plt
import numpy as np


METRICS = ["NMI", "ARI", "ASW", "AvgBIO"]

# Hard-coded same-data results from EC2 runs
SAME_DATA = {
    "JEPA on\n(same-data)":  (0.6421, 0.5200, 0.6436, 0.6019),
    "JEPA off\n(same-data)": (0.7473, 0.6040, 0.7991, 0.7168),
}


def parse_result_file(path):
    """Return (NMI, ARI, ASW, AvgBIO) from a results .txt file, or None."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            nums = re.findall(r'0\.\d+', line)
            if len(nums) == 4:
                return tuple(float(x) for x in nums)
    return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default=".")
    p.add_argument("--out", default="results_comparison.png")
    return p.parse_args()


def main():
    args = parse_args()

    # Collect all conditions
    conditions = dict(SAME_DATA)

    jepa_10k = parse_result_file(os.path.join(args.results_dir, "results_10k_jepa.txt"))
    if jepa_10k:
        conditions["JEPA on\n(pretrain 10k)"] = jepa_10k

    nojepa_10k = parse_result_file(os.path.join(args.results_dir, "results_10k_nojepa.txt"))
    if nojepa_10k:
        conditions["JEPA off\n(pretrain 10k)"] = nojepa_10k

    if not conditions:
        print("No results found. Check --results_dir.")
        return

    labels = list(conditions.keys())
    data = np.array(list(conditions.values()))  # (n_conditions, 4)

    n_conditions = len(labels)
    n_metrics = len(METRICS)
    x = np.arange(n_conditions)
    bar_width = 0.18
    offsets = np.linspace(-(n_metrics - 1) / 2, (n_metrics - 1) / 2, n_metrics) * bar_width

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(max(8, n_conditions * 2.2), 5))

    for i, (metric, color, offset) in enumerate(zip(METRICS, colors, offsets)):
        bars = ax.bar(x + offset, data[:, i], bar_width, label=metric,
                      color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("CellJEPA Clustering Performance — PBMC 3k\n(NMI / ARI / ASW / AvgBIO)")
    ax.legend(loc="upper left", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved to {args.out}")

    # Print results table
    col_w = 40
    print(f"\n{'Model':<{col_w}} {'NMI':>6} {'ARI':>6} {'ASW':>6} {'AvgBIO':>8}")
    print("-" * (col_w + 30))
    for label, vals in conditions.items():
        name = label.replace("\n", " ")
        print(f"{name:<{col_w}} {vals[0]:>6.4f} {vals[1]:>6.4f} {vals[2]:>6.4f} {vals[3]:>8.4f}")


if __name__ == "__main__":
    main()
