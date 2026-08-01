# Handwritten Digit CNN

Fresh CNN project for English MNIST digits and Devanagari digits.

The Devanagari data was extracted from `project_DNN (2).zip` only from:

`project_DNN/Devnagri_dataset/Dataset`

The old files inside the zip were not used.

## Features

- Built-in Keras MNIST loader for English numbers.
- Zip-provided Devanagari digit dataset for Devanagari numbers.
- CNN with convolution blocks, Batch Normalization, Dropout, Global Average Pooling, AdamW, augmentation, EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint.
- Flask frontend with drawing canvas, image upload, English/Devanagari selector, confidence score, and probability bars.

## Trained Results

- Devanagari test accuracy: 99.70%
- English MNIST test accuracy: 99.51%

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Train

Train on the provided Devanagari dataset:

```powershell
python train.py --dataset devanagari --epochs 35
```

Train on built-in English MNIST:

```powershell
python train.py --dataset english --epochs 25
```

Best models are saved to:

`models/best_devanagari_digit_cnn.keras`

`models/best_english_digit_cnn.keras`

Metrics are saved to dataset-specific JSON files, for example:

`models/devanagari_metrics.json`

## Run Frontend

```powershell
python app.py
```

Open:

`http://127.0.0.1:5000`

If port `5000` is already taken, the app automatically falls back to the next free local port and prints the exact URL in the terminal.
