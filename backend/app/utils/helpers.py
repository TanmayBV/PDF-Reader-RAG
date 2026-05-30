import time
import logging
import os
from contextlib import contextmanager
from typing import Generator, Dict

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rag_app")

@contextmanager
def measure_time() -> Generator[Dict[str, float], None, None]:
    """
    Context manager to easily measure the execution time of a code block.
    Usage:
        with measure_time() as timer:
            # do work
        print(timer['elapsed'])
    """
    metrics = {"elapsed": 0.0}
    start = time.perf_counter()
    try:
        yield metrics
    finally:
        metrics["elapsed"] = time.perf_counter() - start

def ensure_directory(path: str) -> None:
    """
    Safely creates a directory path if it does not already exist.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        logger.info(f"Created directory: {path}")
