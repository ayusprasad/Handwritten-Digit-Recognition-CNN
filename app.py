import json
import os
import socket
from io import BytesIO

import tensorflow as tf
from flask import Flask, jsonify, render_template, request

from src.config import DATASETS, metrics_path, model_path
from src.predict import predict_digit, predict_digit_auto


app = Flask(__name__)
models = {}


def pick_port(preferred_port=5000, max_tries=25):
    for port in range(preferred_port, preferred_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(
        f"No free port found in range {preferred_port}-{preferred_port + max_tries - 1}"
    )


def get_model(dataset):
    if dataset not in models:
        path = model_path(dataset)
        if not path.exists():
            raise FileNotFoundError(
                f"{dataset} model not found. Run: python train.py --dataset {dataset}"
            )
        models[dataset] = tf.keras.models.load_model(path)
    return models[dataset]


def load_metrics(dataset):
    path = metrics_path(dataset)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


@app.get("/")
def index():
    metrics = {}
    for dataset in DATASETS:
        dataset_metrics = load_metrics(dataset)
        if dataset_metrics is not None:
            metrics[dataset] = dataset_metrics
    return render_template("index.html", metrics=metrics)


@app.post("/predict")
def predict():
    dataset = request.form.get("dataset", "auto")
    if dataset not in (*DATASETS, "auto"):
        return jsonify({"error": "Choose auto, english, or devanagari."}), 400

    upload = request.files.get("image")
    if upload is None:
        return jsonify({"error": "Upload an image or draw a digit first."}), 400
    if upload.filename == "" and getattr(upload, "content_length", None) == 0:
        return jsonify({"error": "Upload an image or draw a digit first."}), 400

    try:
        image_bytes = BytesIO(upload.read())
        if dataset == "auto":
            result = predict_digit_auto({name: get_model(name) for name in DATASETS}, image_bytes)
        else:
            result = predict_digit(get_model(dataset), image_bytes, dataset)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    return jsonify(result)


@app.get("/health")
def health():
    return jsonify({dataset: model_path(dataset).exists() for dataset in DATASETS})


if __name__ == "__main__":
    preferred_port = int(os.environ.get("PORT", "5000"))
    port = pick_port(preferred_port)
    print(f"Digit CNN Studio running at http://127.0.0.1:{port}")
    app.run(
        debug=os.environ.get("FLASK_DEBUG") == "1",
        use_reloader=False,
        host="127.0.0.1",
        port=port,
    )
