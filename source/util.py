import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


def setup_logger():
    """
    Sets up a logger that logs messages to the console.
    """
    # Initialize Logger
    global logger
    logger = logging.getLogger("experiment_logger")
    logger.setLevel(logging.INFO)

    # Clear existing handlers (prevents duplicate logs in notebooks/loops)
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create Formatter (Format: Time - Level - Message)
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Stream Handler (Writes to Console)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info(f"Logger initialized.")

    return logger

logger: logging.Logger = setup_logger()


def add_file_handler_to_logger(dir_path: str):
    """
    Adds a file handler to an existing logger to log messages to a file in dir_path.
    """
    log_file = os.path.join(dir_path, 'log.txt')

    # Create Formatter (Format: Time - Level - Message)
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Create File Handler (Writes to file)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Add File Handler to Logger
    logger.addHandler(file_handler)

    logger.info(f"File handler added. Logs will also be saved to: {log_file}")


def setup_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return device


def setup_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_directory(parent_dir: str, *subdirs: str, timestamped: bool = False) -> str:
    save_dir = Path(parent_dir).joinpath(*subdirs)
    if timestamped:
        timestamp = datetime.now().strftime("%Y_%m_%d__%H%M%S")
        save_dir = save_dir / timestamp

    os.makedirs(save_dir, exist_ok=True)

    # Convert to string with forward slashes
    save_dir_str = save_dir.as_posix()
    print(f"Created save directory at: {save_dir_str}")

    return save_dir_str


def save_as_csv(df: pd.DataFrame, parent_dir: str, filename: str):
    save_path = os.path.join(parent_dir, filename)
    df.to_csv(save_path, index=False)
    logger.info(f"Saved results to {save_path}")


def save_as_parquet(df: pd.DataFrame, parent_dir: str, filename: str):
    save_path = os.path.join(parent_dir, filename)
    df.to_parquet(save_path, index=False)
    logger.info(f"Saved results to {save_path}")


def save_as_yaml(data: dict, parent_dir: str, filename: str):
    save_path = os.path.join(parent_dir, filename)
    with open(save_path, 'w') as f:
        yaml.dump(data, f)
    logger.info(f"Saved data to {save_path}")
