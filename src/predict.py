import numpy as np
from PIL import Image, ImageOps

from .config import DEVANAGARI_LABELS, ENGLISH_LABELS, IMAGE_SIZE


def preprocess_digit_image(file_or_bytes):
    image = Image.open(file_or_bytes).convert("L")
    arr = np.asarray(image, dtype=np.uint8)
    corners = np.concatenate(
        [arr[:4, :4].ravel(), arr[:4, -4:].ravel(), arr[-4:, :4].ravel(), arr[-4:, -4:].ravel()]
    )
    if np.median(corners) > 127:
        image = ImageOps.invert(image)
    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)
    image.thumbnail((20, 20), Image.Resampling.LANCZOS)

    canvas = Image.new("L", IMAGE_SIZE, 0)
    left = (IMAGE_SIZE[0] - image.width) // 2
    top = (IMAGE_SIZE[1] - image.height) // 2
    canvas.paste(image, (left, top))

    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    return arr.reshape(1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)


def labels_for(dataset):
    return DEVANAGARI_LABELS if dataset == "devanagari" else ENGLISH_LABELS


def predict_digit(model, file_or_bytes, dataset):
    image = preprocess_digit_image(file_or_bytes)
    probs = model.predict(image, verbose=0)[0]
    label_id = int(np.argmax(probs))
    labels = labels_for(dataset)
    return {
        "digit": label_id,
        "label": labels[label_id],
        "confidence": float(probs[label_id]),
        "probabilities": [
            {"digit": i, "label": labels[i], "confidence": float(score)}
            for i, score in enumerate(probs)
        ],
    }
