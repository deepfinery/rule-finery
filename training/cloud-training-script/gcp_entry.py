"""Vertex AI entrypoint that installs deps (if requested) and runs training."""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_TRAIN_DIR = REPO_ROOT / "training" / "local-training-scripts"
COMMON_DIR = REPO_ROOT / "training" / "common"
REQUIREMENTS = LOCAL_TRAIN_DIR / "requirements.txt"
TRAIN_MODULE = "training.common.train"


def main():
    install_flag = os.environ.get("INSTALL_DEPS", "0").lower()
    if install_flag in {"1", "true", "yes"}:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)],
            check=True,
        )
    extra_path = str(REPO_ROOT)
    existing = os.environ.get("PYTHONPATH")
    if existing:
        os.environ["PYTHONPATH"] = f"{extra_path}:{existing}"
    else:
        os.environ["PYTHONPATH"] = extra_path

    cmd = [sys.executable, "-m", TRAIN_MODULE, *sys.argv[1:]]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
