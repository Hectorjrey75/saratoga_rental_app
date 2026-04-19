"""
Logging configuration using loguru.
"""
import sys
from pathlib import Path
from loguru import logger
from src.config.settings import config


def setup_logging() -> None:
    """Configure logging for the application."""
    
    # Remove default handler
    logger.remove()
    
    # Add console handler with custom format
    logger.add(
        sys.stdout,
        format=config.logging.format,
        level=config.logging.level,
        colorize=True
    )
    
    # Add file handler for persistent logging
    logger.add(
        config.logging.log_file,
        format=config.logging.format,
        level=config.logging.level,
        rotation=config.logging.rotation,
        retention=config.logging.retention,
        compression="zip"
    )
    
    # Add error file handler
    error_log = config.logging.log_file.parent / "error.log"
    logger.add(
        error_log,
        format=config.logging.format,
        level="ERROR",
        rotation="5 MB",
        retention="30 days"
    )
    
    logger.info("Logging configured successfully")


def get_logger(name: str = None):
    """Get a logger instance with optional name binding."""
    if name:
        return logger.bind(name=name)
    return logger


# Auto-setup logging on import
setup_logging()