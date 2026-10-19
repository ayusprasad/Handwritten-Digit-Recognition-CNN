# Handwritten Digit Recognition Studio

[![CI](https://github.com/ayusprasad/Handwritten-Digit-Recognition-CNN/actions/workflows/ci.yml/badge.svg)](https://github.com/ayusprasad/Handwritten-Digit-Recognition-CNN/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/Deep%20learning-TensorFlow-FF6F00?logo=tensorflow&logoColor=white)
![Flask](https://img.shields.io/badge/Interface-Flask-000000?logo=flask&logoColor=white)

A CNN-based Flask application for recognizing handwritten English and Devanagari digits from uploads or a drawing canvas.

## Features

- Train separate ten-class models for English MNIST and Devanagari digits.
- Normalize uploaded images and show the predicted label, confidence, and class probabilities.
- Report model availability through a health endpoint.
- Return a clear unavailable response when a trained model file is missing.

## Architecture

~~~text
Flask UI and JSON API → image normalization → Keras CNN
Training scripts → saved model files and metric summaries
~~~

## Local setup

### Prerequisites

- Python 3.11 or newer
- The Devanagari image dataset in project_DNN/Devnagri_dataset/Dataset for that training flow
- Network access on the first English MNIST training run so TensorFlow can download MNIST

### Install

~~~bash
git clone https://github.com/ayusprasad/Handwritten-Digit-Recognition-CNN.git
cd Handwritten-Digit-Recognition-CNN
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

Activate the environment with source .venv/bin/activate on macOS/Linux or .venv/Scripts/Activate.ps1 in PowerShell.

### Train

~~~bash
python train.py --dataset english --epochs 25
python train.py --dataset devanagari --epochs 35
~~~

### Run

~~~bash
python app.py
~~~

Open the URL printed by Flask. The local server binds to 127.0.0.1 by default. Set HOST=0.0.0.0 for container use and PORT=5000 to choose a port.

## Quality checks

~~~bash
python -m pytest -q
ruff check --select E4,E7,E9,F app.py src/predict.py tests
~~~

GitHub Actions runs lint, image-processing and API smoke tests, Python compilation, and a Docker build. Model training and dataset downloads are not part of CI.

## Model results

Accuracy depends on the selected dataset and training run. Train locally to generate model-specific metrics in models/.
