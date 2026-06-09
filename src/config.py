from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "project_DNN" / "Devnagri_dataset" / "Dataset"
MODEL_DIR = ROOT / "models"

IMAGE_SIZE = (28, 28)
CHANNELS = 1
NUM_CLASSES = 10
BATCH_SIZE = 128
SEED = 42

ENGLISH_LABELS = [str(i) for i in range(10)]
DEVANAGARI_LABELS = ["०", "१", "२", "३", "४", "५", "६", "७", "८", "९"]
DATASETS = ("english", "devanagari")


def model_path(dataset):
    return MODEL_DIR / f"best_{dataset}_digit_cnn.keras"


def metrics_path(dataset):
    return MODEL_DIR / f"{dataset}_metrics.json"
