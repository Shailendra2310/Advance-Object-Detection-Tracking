"""
Logging utility for Smart Vision System.
"""

import logging
import os
from datetime import datetime


def setup_logger(name: str, log_dir: str = "output/logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)

    # Console handler (WARNING and above only)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)

    formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
                                  datefmt="%H:%M:%S")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger
