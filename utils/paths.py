from pathlib import Path

# Root of the project
ROOT_PATH = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = ROOT_PATH / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Specific file paths
JSON_FILE = RAW_DATA_DIR / "large_rock_dataset.json"
BASELINE_CONFIG_YAML = ROOT_PATH / "configs" / "baseline_config.yaml"
AT_CONFIG_YAML = ROOT_PATH / "configs" / "active_teacher_config.yaml"

# Training directories
TRAIN_IMAGES_DIR = PROCESSED_DATA_DIR / "images" / "train"
TRAIN_LABELS_DIR = PROCESSED_DATA_DIR / "labels" / "train"

# Validation directories
VAL_IMAGES_DIR = PROCESSED_DATA_DIR / "images" / "val"
VAL_LABELS_DIR = PROCESSED_DATA_DIR / "labels" / "val"

# Test directories
TEST_IMAGES_DIR = PROCESSED_DATA_DIR / "images" / "test"
TEST_LABELS_DIR = PROCESSED_DATA_DIR / "labels" / "test"

# Inference directories
INFERENCE_RAW_DIR = DATA_DIR / "inference"