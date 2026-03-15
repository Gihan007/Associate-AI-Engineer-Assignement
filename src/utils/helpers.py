"""
Utility functions for logging, config management, etc.
"""

import logging
import json
from pathlib import Path
from typing import Any, Dict


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level
        
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)


def save_config(config: Dict[str, Any], filepath: str) -> None:
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        filepath: Output file path
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)


def load_config(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        filepath: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def create_directories(path_list: list) -> None:
    """
    Create multiple directories.
    
    Args:
        path_list: List of directory paths to create
    """
    for path in path_list:
        Path(path).mkdir(parents=True, exist_ok=True)
