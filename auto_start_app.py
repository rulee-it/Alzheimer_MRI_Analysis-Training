from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parent
    model_path = project_root / "models" / "alzheimer_model.h5"

    print(f"[watcher] Waiting for model: {model_path}")
    while not model_path.exists():
        time.sleep(10)

    print("[watcher] Model found. Starting Flask app...")
    # Start the Flask app in a new process; inherit environment
    python_exe = sys.executable
    subprocess.Popen([python_exe, str(project_root / "app.py")])
    print("[watcher] Flask app started.")


if __name__ == "__main__":
    main()
