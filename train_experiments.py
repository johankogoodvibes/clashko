"""Train and compare models with hardcoded experiment settings."""
from datetime import datetime

import train


TARGETS = ["winner", "tower_hp", "crowns"]

# Each tuple contains matchup transformer blocks, card features, and target names.
RUNS = [
    # (3, 16, TARGETS),
    # (6, 16, TARGETS),
    (3, 32, ["winner",]),
    # (3, 32, ["winner", "tower_hp"]),
    # (3, 32, ["winner", "crowns"]),
    # (3, 32, ["winner", "tower_hp", "crowns"]),
    (6, 64, ["winner", "tower_hp"]),
    # (6, 32, TARGETS),
    # (6, 32, TARGETS),
    # (3, 64, TARGETS),
    # (6, 64, TARGETS),
]


def run_label(matchup_layers, card_features, targets):
    return f"{matchup_layers} blocks, {card_features} features | {', '.join(targets)}"


def plot_comparison(histories, path):
    from plot_history import MaxNLocator, plt

    metrics = [(key, title, ylabel) for key, title, ylabel in [
        ("loss", "Total loss", "Loss"),
        ("winner_accuracy", "Winner accuracy", "Accuracy (%)"),
        ("tower_hp_mae", "Tower HP mean absolute error", "Normalized HP error"),
        ("crowns_mae", "Crown count mean absolute error", "Crowns"),
    ] if any(key in history for history in histories.values())]
    fig, axes = plt.subplots(len(metrics), 2, figsize=(14, 3.7 * len(metrics)),
                             layout="constrained", squeeze=False)
    fig.suptitle("Training experiment comparison", fontsize=16)
    for row, (metric, title, ylabel) in zip(axes, metrics):
        for ax, prefix, split in zip(row, ("", "val_"), ("Training", "Validation")):
            column = prefix + metric
            for index, (settings, history) in enumerate(histories.items()):
                if column not in history:
                    continue
                values = history[column] * 100 if metric.endswith("accuracy") else history[column]
                ax.plot(history.index, values, label=run_label(*settings), color=f"C{index % 10}", linewidth=2)
            ax.set(title=f"{split} {title.lower()}", xlabel="Epoch", ylabel=ylabel)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(alpha=0.25)
    axes[0, 1].legend(fontsize=8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_accuracy_comparison(histories, path):
    from plot_history import MaxNLocator, plt

    fig, axes = plt.subplots(2, 1, figsize=(8, 7.8), sharex=True, sharey=True)
    fig.patch.set_facecolor("#f8fafc")
    fig.suptitle(f"Accuracy across {len(histories)} training runs", fontsize=20, fontweight="bold", y=0.98)
    for ax, (metric, title) in zip(axes, (("winner_accuracy", "Training"), ("val_winner_accuracy", "Validation"))):
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
    for index, (matchup_layers, card_features, targets) in enumerate(RUNS, 1):
        settings = (matchup_layers, card_features, tuple(targets))
        print(f"\nRun {index}/{len(RUNS)}: {run_label(*settings)}", flush=True)
        run_dir = output_dir / f"{index:02d}_blocks-{matchup_layers}_features-{card_features}_{'-'.join(targets)}"
        histories[settings] = train.main(
            model_options={"matchup_layers": matchup_layers, "card_features": card_features, "targets": targets},
            output_dir=run_dir,
        )
        from keras import backend

        backend.clear_session()
        plot_comparison(histories, output_dir / "comparison.png")
        winner_histories = {key: history for key, history in histories.items() if "winner_accuracy" in history}
        if winner_histories:
            plot_accuracy_comparison(winner_histories, output_dir / "accuracy_comparison.png")
    print(f"Saved experiments to {output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
