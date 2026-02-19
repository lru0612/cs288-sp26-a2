"""
Pytest configuration for the tests package.
Adds the parent directory to sys.path so tests can import common and tokenizer.
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import common and tokenizer
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
