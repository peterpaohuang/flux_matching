"""
Plot Wasserstein distance over training steps for each run in
spring_autoregressive_experiment/runs/, then produce a comparison plot
showing the best (minimum) Wasserstein distance achieved by each method.
"""

import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")
OUT_DIR = os.path.dirname(__file__)


def load_wasserstein(csv_path):
    """Return (steps, wasserstein) lists from a train_log.csv, skipping empty values."""
    steps, values = [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            w = row.get("wasserstein", "").strip()
            if w:
                steps.append(int(row["step"]))
                values.append(float(w))
    return steps, values


def main():
    run_dirs = sorted(
        d for d in os.listdir(RUNS_DIR)
        if os.path.isdir(os.path.join(RUNS_DIR, d))
    )

    best = {}  # run_name -> best wasserstein value

    for run_name in run_dirs:
        csv_path = os.path.join(RUNS_DIR, run_name, "train_log.csv")
        if not os.path.isfile(csv_path):
            print(f"  [skip] {run_name}: no train_log.csv")
            continue

        steps, values = load_wasserstein(csv_path)
        if not steps:
            print(f"  [skip] {run_name}: no wasserstein values found")
            continue

        best[run_name] = min(values)

        # --- per-run plot ---
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(steps, values, marker="o", linewidth=1.5, markersize=4)
        ax.set_xlabel("Step")
        ax.set_ylabel("Wasserstein distance")
        ax.set_title(f"Wasserstein distance — {run_name}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out_path = os.path.join(RUNS_DIR, run_name, "wasserstein.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  saved {out_path}  (best={best[run_name]:.4f})")

    if not best:
        print("No runs with Wasserstein data found — nothing to compare.")
        return

    # --- causal vs non-causal curve comparisons ---
    all_curves = {}  # run_name -> (steps, values)
    for run_name in run_dirs:
        csv_path = os.path.join(RUNS_DIR, run_name, "train_log.csv")
        if not os.path.isfile(csv_path):
            continue
        steps, values = load_wasserstein(csv_path)
        if steps:
            all_curves[run_name] = (steps, values)

    for method, label in [("dsm", "DSM"), ("flux", "Flux")]:
        causal_key = f"{method}_causal"
        noncausal_key = f"{method}_noncausal"
        if causal_key not in all_curves and noncausal_key not in all_curves:
            continue

        fig, ax = plt.subplots(figsize=(8, 4))
        for key, name in [(causal_key, "Causal"), (noncausal_key, "Non-causal")]:
            if key in all_curves:
                s, v = all_curves[key]
                ax.plot(s, v, marker="o", linewidth=1.5, markersize=4, label=name)
        ax.set_xlabel("Step")
        ax.set_ylabel("Wasserstein distance")
        ax.set_title(f"{label}: Causal vs Non-causal Wasserstein distance")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path = os.path.join(OUT_DIR, f"wasserstein_{method}_causal_comparison.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  saved {out_path}")

    # --- comparison bar chart ---
    names = list(best.keys())
    bests = [best[n] for n in names]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.8), 5))
    bars = ax.bar(range(len(names)), bests, color="steelblue", edgecolor="black", linewidth=0.7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Best Wasserstein distance (↓ better)")
    ax.set_title("Best Wasserstein distance per method")
    ax.grid(True, axis="y", alpha=0.3)

    # annotate each bar with its value
    for bar, val in zip(bars, bests):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.01,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=8,
        )

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "wasserstein_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nsaved comparison plot → {out_path}")


if __name__ == "__main__":
    main()
