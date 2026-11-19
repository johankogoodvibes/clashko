"""Train the Clash Royale model on the NVIDIA GPU, directly on Windows."""
from pathlib import Path

from gpu_runtime import configure_gpu

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
EPOCHS = 40
WARMUP = 15
BATCH_SIZE = 512
RANDOM_SEED = 47


def main(*, model_options=None, output_dir=OUTPUT_DIR):
    # CUDA must be configured before importing Keras or building the model.
    configure_gpu()

    import keras
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    from data import load_data
    from model import build_model
    from plot_history import plot_history

    keras.utils.set_random_seed(RANDOM_SEED)
    X, y, hp, crowns = load_data()
    train_indices, holdout_indices = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=RANDOM_SEED
    )
    val_indices, test_indices = train_test_split(
        holdout_indices, test_size=0.5, random_state=RANDOM_SEED
    )
    cnn = build_model(X.shape[1:], **(model_options or {}))
    valid = np.ones(len(y), dtype=bool)
    if "tower_hp" in cnn.output_names:
        valid &= np.isfinite(hp).all(axis=(1, 2))
    if "crowns" in cnn.output_names:
        valid &= (crowns >= 0).all(axis=1)
    train_indices = train_indices[valid[train_indices]]
    val_indices = val_indices[valid[val_indices]]
    if len(train_indices) == 0 or len(val_indices) == 0:
        raise ValueError("The selected targets need complete labels in both the training and validation sets.")
    X_train, y_train, hp_train, crowns_train = X[train_indices], y[train_indices], hp[train_indices], crowns[train_indices]
    X_val, y_val, hp_val, crowns_val = X[val_indices], y[val_indices], hp[val_indices], crowns[val_indices]
    del X, y
    print(
        f"Train: {len(train_indices):,} | Validation: {len(val_indices):,} "
        f"| Test (reserved): {len(test_indices):,}"
    )

    print(f"Model weights device: {cnn.trainable_variables[0].device}")
    y_train_one_hot = keras.utils.to_categorical((y_train == 1).astype(np.int32), num_classes=2)
    y_val_one_hot = keras.utils.to_categorical((y_val == 1).astype(np.int32), num_classes=2)

    train_targets = {
        "winner":y_train_one_hot,
        "tower_hp":hp_train,
        "crowns":crowns_train,
    }

    val_targets = {
        "winner":y_val_one_hot,
        "tower_hp":hp_val,
        "crowns":crowns_val,
    }
    train_targets = {name: train_targets[name] for name in cnn.output_names}
    val_targets = {name: val_targets[name] for name in cnn.output_names}

    fit_args = dict(
        validation_data=(X_val, val_targets),
        batch_size=BATCH_SIZE,
        verbose=2,
    )
    
    warmup = cnn.fit(
        X_train,
        train_targets,
        epochs=WARMUP,
        **fit_args
    )

    history = cnn.fit(
        X_train,
        train_targets,
        **fit_args,
        initial_epoch=WARMUP,
        epochs=EPOCHS,
        # callbacks=[
        #     keras.callbacks.EarlyStopping(
        #         monitor="val_winner_loss" if len(cnn.output_names) > 1 and "winner" in cnn.output_names else "val_loss",
        #         patience=6,
        #         min_delta=0.001,
        #         restore_best_weights=True,
        #     )
        # ],
    )

    for metric, values in warmup.history.items():
        history.history[metric] = values + history.history[metric]

    if len(cnn.output_names) == 1:
        target, = cnn.output_names
        for prefix in ("", "val_"):
            for metric in ("accuracy", "mae"):
                key = f"{prefix}{metric}"
                if key in history.history:
                    history.history[f"{prefix}{target}_{metric}"] = history.history.pop(key)

    output_dir.mkdir(parents=True, exist_ok=True)
    history_table = pd.DataFrame(history.history)
    history_table.index = np.arange(1, len(history_table) + 1)
    history_table.to_csv(output_dir / "history.csv", index_label="epoch")
    plot_history(history.history, output_dir / "history.png")
    cnn.save(output_dir / "model.h5")

    print(f"Saved model.h5, history.csv and history.png to: {output_dir.resolve()}")
    return history_table


if __name__ == "__main__":
    main()
