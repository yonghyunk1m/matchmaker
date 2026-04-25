#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Matchmaker is a library for real-time music alignment
"""

from pathlib import Path

from . import dp, features, io, prob, utils
from .matchmaker import *

__all__ = ["dp", "features", "io", "prob", "utils"]

try:
    from importlib.metadata import version

    __version__ = version("pymatchmaker")
except Exception:  # pragma: no cover
    __version__ = "0.2.1"

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
EXAMPLE_SCORE = str(_ASSETS_DIR / "mozart_k265_var1.musicxml")
EXAMPLE_PERFORMANCE = str(_ASSETS_DIR / "mozart_k265_var1.mid")
EXAMPLE_MATCH = str(_ASSETS_DIR / "mozart_k265_var1.match")
EXAMPLE_AUDIO = str(_ASSETS_DIR / "mozart_k265_var1.mp3")
