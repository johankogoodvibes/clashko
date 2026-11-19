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
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    accuracy_key = "winner_accuracy" if "winner_accuracy" in history else "accuracy"

    for ax, metric, title in zip(axes, ["loss", accuracy_key], ["Loss", "Accuracy"]):
        ax.plot(epochs, history[metric], marker="o", label="Training")
        ax.plot(epochs, history[f"val_{metric}"], marker="o", label="Validation")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
        ax.legend()

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
