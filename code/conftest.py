"""
pytest configuration: adds code/ to sys.path so all test modules
can import project packages (env, models, training, baselines, experiments)
without requiring PYTHONPATH to be set externally.
"""

import sys
from pathlib import Path

# code/ directory (this file lives there)
CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
