# coding=utf-8
"""
The minimal way to run the app from Python directly, without the CLI
wrapper in ``run.py``. Useful as a starting point for your own script, or
for running from a notebook / IDE "run" button.

    python examples/run_demo.py
"""

import sys
from pathlib import Path

# Make the package importable when running this file directly from a clone
# (not needed if you `pip install -e .` the project).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interactive_glm.app import create_app
from interactive_glm.data.demo import build_demo_dataset

if __name__ == "__main__":
    print("Building the offline synthetic demo dataset (a few seconds)...")
    data = build_demo_dataset()

    app = create_app(data)  # ViewerSettings() defaults are used
    app.run(debug=False, port=8050)
