import logging
import os
from datetime import datetime


def setup_logger(name: str = "contractia", log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"audit_{timestamp}.log")

    lgr = logging.getLogger(name)
    lgr.setLevel(logging.DEBUG)

    if not lgr.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)

        lgr.addHandler(fh)
        lgr.addHandler(ch)

    return lgr


logger = setup_logger()
