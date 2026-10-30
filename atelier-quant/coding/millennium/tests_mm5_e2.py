"""Tests de la question mm5_e2 : voir _suites.py et _harnais.py. Usage autonome : python tests_mm5_e2.py reference"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _suites import SUITES  # noqa: E402

SUITE = SUITES["mm5_e2"]
evaluer = SUITE.evaluer

if __name__ == "__main__":
    SUITE.principal()
