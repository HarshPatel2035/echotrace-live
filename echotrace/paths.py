"""Project-relative locations, independent of the current working directory."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "hand_landmarker.task"
CAPTURES_DIR = PROJECT_DIR / "captures"
