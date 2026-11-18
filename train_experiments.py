"""Train and compare models with hardcoded experiment settings."""
from datetime import datetime

import train


# Each tuple contains matchup transformer blocks and card features.
RUNS = [
    (3, 16),
    (6, 16),
    (3, 32),
    (6, 32),
    (3, 64),
    (6, 64),
]


def run_label(matchup_layers, card_features):
    return f"{matchup_layers} blocks, {card_features} features"


def plot_comparison(histories, path):
    from plot_history import MaxNLocator, plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    fig.suptitle("Training experiment comparison", fontsize=16)
    for ax, (column, title) in zip(axes.flat, [
        ("loss", "Training loss"), ("val_loss", "Validation loss"),
        ("accuracy", "Training accuracy"), ("val_accuracy", "Validation accuracy"),
    ]):
        for settings, history in histories.items():
            values = history[column] * 100 if column.endswith("accuracy") else history[column]
            ax.plot(history.index, values, label=run_label(*settings), linewidth=2)
        ax.set(title=title, xlabel="Epoch", ylabel="Accuracy (%)" if column.endswith("accuracy") else "Loss")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.25)
    axes[0, 1].legend(fontsize=9)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_accuracy_comparison(histories, path):
    from plot_history import MaxNLocator, plt

    fig, axes = plt.subplots(2, 1, figsize=(8, 7.8), sharex=True, sharey=True)
    fig.patch.set_facecolor("#f8fafc")
    fig.suptitle(f"Accuracy across {len(histories)} training runs", fontsize=20, fontweight="bold", y=0.98)
    for ax, (metric, title) in zip(axes, (("accuracy", "Training"), ("val_accuracy", "Validation"))):
        for settings, history in histories.items():
            ax.plot(history.index, history[metric] * 100, label=run_label(*settings), linewidth=2.2)
        ax.set_title(title, loc="left", fontsize=12, fontweight="semibold")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
    axes[-1].set_xlabel("Epoch")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.023))
    fig.subplots_adjust(left=0.085, right=0.975, top=0.885, bottom=0.15, hspace=0.26, wspace=0.11)
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output_dir = train.OUTPUT_DIR / "experiments" / stamp
    histories = {}
    for index, settings in enumerate(RUNS, 1):
        matchup_layers, card_features = settings
        print(f"\nRun {index}/{len(RUNS)}: {run_label(*settings)}", flush=True)
        run_dir = output_dir / f"{index:02d}_blocks-{matchup_layers}_features-{card_features}"
        histories[settings] = train.main(
            model_options={"matchup_layers": matchup_layers, "card_features": card_features},
            output_dir=run_dir,
        )
        from keras import backend

        backend.clear_session()
        plot_comparison(histories, output_dir / "comparison.png")
        plot_accuracy_comparison(histories, output_dir / "accuracy_comparison.png")
    print(f"Saved experiments to {output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
