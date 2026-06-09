import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf

from src.config import MODEL_DIR, metrics_path, model_path
from src.data import load_dataset
from src.model import build_digit_cnn


def plot_history(history, out_path):
    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="validation")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="validation")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def train(dataset, epochs):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    selected_model_path = model_path(dataset)
    selected_metrics_path = metrics_path(dataset)
    train_ds, val_ds, test_ds = load_dataset(dataset)
    model = build_digit_cnn()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=7,
            restore_best_weights=True,
            mode="max",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.35,
            patience=3,
            min_lr=1e-6,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            selected_model_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
    model.save(selected_model_path)

    metrics = {
        "dataset": dataset,
        "epochs_ran": len(history.history["loss"]),
        "best_validation_accuracy": max(history.history["val_accuracy"]),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "model_path": str(selected_model_path),
    }
    selected_metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_history(history, MODEL_DIR / f"{dataset}_training_history.png")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train CNN for handwritten digit classification.")
    parser.add_argument(
        "--dataset",
        choices=["english", "devanagari"],
        default="devanagari",
        help="english uses built-in MNIST; devanagari uses only the dataset extracted from the zip.",
    )
    parser.add_argument("--epochs", type=int, default=35)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(42)
    metrics = train(args.dataset, args.epochs)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
