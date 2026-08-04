"""Pytest bootstrap: make this paper's model importable as ``import model``."""
import os
import sys

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # paper1/
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)
