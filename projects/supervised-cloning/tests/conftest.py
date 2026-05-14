"""Pytest setup for the unified-aux supervised-cloning tests.

- Ensures the project root (the folder containing ``bot/``) is on ``sys.path``
  and is the CWD so module-level chdir in ``collect_data.py`` / ``train.py``
  finds the right files.
- Dynamically loads ``train.py`` and ``collect_data.py`` from the
  hyphenated ``projects/supervised-cloning/`` directory (which is not a
  valid Python package name) and exposes them as the importable names
  ``bc_train`` and ``bc_collect``.
"""

from __future__ import annotations
import importlib.util
import os
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
_root = _here
while not (_root / "bot").is_dir():
    _root = _root.parent
    if _root == _root.parent:
        raise RuntimeError("Could not find project root containing 'bot/'")

os.chdir(_root)
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

PROJECT_ROOT = _root
SC_DIR = _root / "projects" / "supervised-cloning"


def _load(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# Loaded once per pytest process. After this, tests can ``import bc_train``.
bc_train = _load("bc_train", SC_DIR / "train.py")
bc_collect = _load("bc_collect", SC_DIR / "collect_data.py")
