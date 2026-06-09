import json
from io import BytesIO

import tensorflow as tf
from flask import Flask, jsonify, render_template, request

from src.config import DATASETS, metrics_path, model_path
from src.predict import predict_digit


app = Flask(__name__)
models = {}


def get_model(dataset):
    if dataset not in models:
        path = model_path(dataset)
        if not path.exists():
            raise FileNotFoundError(f"{dataset} model not found. Run: python train.py --dataset {dataset}")
        models[dataset] = tf.keras.models.load_model(path)
    return models[dataset]


@app.get("/")
def index():
    metrics = {}
    for dataset in DATASETS:
        path = metrics_path(dataset)
        if path.exists():
            metrics[dataset] = json.loads(path.read_text(encoding="utf-8"))
    return render_template("index.html", metrics=metrics)


@app.post("/predict")
def predict():
    dataset = request.form.get("dataset", "devanagari")
    if dataset not in DATASETS:
        return jsonify({"error": "Choose english or devanagari."}), 400

    upload = request.files.get("image")
    if upload is None:
        return jsonify({"error": "Upload an image or draw a digit first."}), 400

    try:
        result = predict_digit(get_model(dataset), BytesIO(upload.read()), dataset)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    return jsonify(result)


@app.get("/health")
def health():
    return jsonify({dataset: model_path(dataset).exists() for dataset in DATASETS})


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
