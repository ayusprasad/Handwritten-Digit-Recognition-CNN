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
DEVANAGARI_LABELS = [
    "\u0966",
    "\u0967",
    "\u0968",
    "\u0969",
    "\u096a",
    "\u096b",
    "\u096c",
    "\u096d",
    "\u096e",
    "\u096f",
]
DATASETS = ("english", "devanagari")


def model_path(dataset):
    return MODEL_DIR / f"best_{dataset}_digit_cnn.keras"


def metrics_path(dataset):
    return MODEL_DIR / f"{dataset}_metrics.json"
