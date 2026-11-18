"""Train the Clash Royale model on the NVIDIA GPU, directly on Windows."""
import json
from pathlib import Path

from gpu_runtime import configure_gpu

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
EPOCHS = 40
BATCH_SIZE = 256
MAX_ROWS = None  # None = all collected matches; set an integer for a smaller run.
RANDOM_SEED = 47
EVALUATE_TEST_SET = False


def main():
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
    X, y = load_data(max_rows=MAX_ROWS)

    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=RANDOM_SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, shuffle=True, random_state=RANDOM_SEED
    )
    del X, y, X_rest, y_rest
    print(f"Train: {len(y_train):,} | Validation: {len(y_val):,} | Test: {len(y_test):,}")

    cnn = build_model(X_train.shape[1:])
    print(f"Model weights device: {cnn.trainable_variables[0].device}")
    y_train_one_hot = keras.utils.to_categorical((y_train == 1).astype(np.int32), num_classes=2)
    y_val_one_hot = keras.utils.to_categorical((y_val == 1).astype(np.int32), num_classes=2)

    


    history = cnn.fit(
        X_train,
        y_train_one_hot,
        validation_data=(X_val, y_val_one_hot),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                min_delta=0.0001,
                restore_best_weights=True,
                verbose=1,
            )
        ],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    history_table = pd.DataFrame(history.history)
    history_table.index = np.arange(1, len(history_table) + 1)
    history_table.to_csv(OUTPUT_DIR / "history.csv", index_label="epoch")
    plot_history(history.history, OUTPUT_DIR / "history.png")
    cnn.save(OUTPUT_DIR / "model.h5")

    if EVALUATE_TEST_SET:
        y_test_one_hot = keras.utils.to_categorical((y_test == 1).astype(np.int32), num_classes=2)
        metrics = cnn.evaluate(X_test, y_test_one_hot, return_dict=True, verbose=2)
        (OUTPUT_DIR / "test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"Saved model.h5, history.csv and history.png to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
