"""Vertex AI entrypoint that installs deps (if requested) and runs training."""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_TRAIN_DIR = REPO_ROOT / "training" / "local-training-scripts"
COMMON_DIR = REPO_ROOT / "training" / "common"
REQUIREMENTS = LOCAL_TRAIN_DIR / "requirements.txt"
TRAIN_SCRIPT = COMMON_DIR / "train.py"


def main():
    install_flag = os.environ.get("INSTALL_DEPS", "0").lower()
    if install_flag in {"1", "true", "yes"}:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)],
            check=True,
        )
    cmd = [sys.executable, str(TRAIN_SCRIPT), *sys.argv[1:]]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
