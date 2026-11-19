"""Save training curves, or replot a saved history without training again."""
import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".tools" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd


def plot_history(history, output_path):
    epochs = range(1, len(history["loss"]) + 1)
    accuracy_key = "winner_accuracy" if "winner_accuracy" in history else "accuracy"
    metrics = [
        ("loss", "Total loss", "Loss"),
        (accuracy_key, "Winner accuracy", "Accuracy"),
        ("tower_hp_loss", "Tower HP loss", "Loss"),
        ("tower_hp_mae", "Tower HP mean absolute error", "Normalized HP error"),
        ("crowns_loss", "Crown count loss", "Loss"),
        ("crowns_mae", "Crown count mean absolute error", "Mean absolute error"),
    ]
    metrics = [(key, title, ylabel) for key, title, ylabel in metrics if key in history]
    rows = (len(metrics) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(12, 4 * rows), layout="constrained", squeeze=False)

    for ax, (metric, title, ylabel) in zip(axes.flat, metrics):
        ax.plot(epochs, history[metric], marker="o", label="Training")
        if f"val_{metric}" in history:
            ax.plot(epochs, history[f"val_{metric}"], marker="o", label="Validation")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
        ax.legend()

    for ax in list(axes.flat)[len(metrics):]:
        ax.remove()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    output_dir = Path(__file__).resolve().parent / "outputs"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history_csv", nargs="?", type=Path, default=output_dir / "history.csv")
    args = parser.parse_args()
    history = pd.read_csv(args.history_csv).drop(columns="epoch").to_dict(orient="list")
    output_path = args.history_csv.with_suffix(".png")
    plot_history(history, output_path)
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
