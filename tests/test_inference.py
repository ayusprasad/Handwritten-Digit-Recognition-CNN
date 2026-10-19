from io import BytesIO

import numpy as np
from PIL import Image

from src.predict import predict_digit, preprocess_digit_image


def test_preprocess_returns_normalized_28_by_28_tensor() -> None:
    image = Image.new("L", (28, 28), color=255)
    image.putpixel((14, 14), 0)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    result = preprocess_digit_image(BytesIO(buffer.getvalue()))
    assert result.shape == (1, 28, 28, 1)
    assert result.dtype == np.float32
    assert 0.0 <= float(result.min()) <= float(result.max()) <= 1.0


def test_blank_image_returns_actionable_error() -> None:
    image = Image.new("L", (28, 28), color=255)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    with np.testing.assert_raises_regex(ValueError, "No visible digit"):
        predict_digit(None, BytesIO(buffer.getvalue()), "english")


def test_health_endpoint_reports_model_availability() -> None:
    from app import app
    response = app.test_client().get("/health")
    assert response.status_code == 200
    assert set(response.json) == {"english", "devanagari"}
