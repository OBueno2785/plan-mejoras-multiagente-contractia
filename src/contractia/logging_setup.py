"""Configuración de logging (P2 6.1).

Un único entry point: `setup_logger()`. Coloca un FileHandler con timestamps
para debugging y un StreamHandler conciso para la consola del notebook.
"""
from __future__ import annotations
import logging
from pathlib import Path

from .config import CFG

_INITIALIZED = False


def setup_logger(level: int = logging.INFO, log_file: str | None = None) -> logging.Logger:
    global _INITIALIZED
    logger = logging.getLogger("contractia")
    if _INITIALIZED:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    log_path = Path(log_file or CFG.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)

    _INITIALIZED = True
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"contractia.{name}")
