"""
Logging configuration and utilities for the course compiler.
"""
import logging
import logging.handlers
import os
import sys
from typing import Optional, Dict, Any

from .config import config

# Log format string
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = logging.getLevelName(config.LOG_LEVEL.upper())

def setup_logger(name: Optional[str] = None, level: Optional[int] = None) -> logging.Logger:
    """
    Configure and return a logger with the given name.

    Args:
        name: Logger name. If None, returns the root logger.
        level: Logging level. If None, uses the level from config.

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Don't add handlers if they're already configured
    if logger.handlers:
        return logger

    # Set log level
    logger.setLevel(level or LOG_LEVEL)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level or LOG_LEVEL)

    # Create formatter and add it to the handler
    formatter = logging.Formatter(LOG_FORMAT)
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(console_handler)

    # Configure file handler if logs directory is specified
    if hasattr(config, 'LOG_DIR') and config.LOG_DIR:
        os.makedirs(config.LOG_DIR, exist_ok=True)
        log_file = os.path.join(config.LOG_DIR, 'course_compiler.log')

        # Rotate logs (10MB per file, keep 5 backups)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setLevel(level or LOG_LEVEL)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def log_execution_time(logger: logging.Logger):
    """Decorator to log the execution time of a function."""
    import time
    from functools import wraps

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger.debug(f"Starting {func.__name__}")
            try:
                result = func(*args, **kwargs)
                logger.debug(
                    f"Completed {func.__name__} in {time.time() - start_time:.2f} seconds"
                )
                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
                raise
        return wrapper
    return decorator

def log_exceptions(logger: logging.Logger):
    """Decorator to log exceptions from a function."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Exception in {func.__name__}: {str(e)}", exc_info=True)
                raise
        return wrapper
    return decorator

# Create a default logger instance
logger = setup_logger(__name__)

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with the given name.

    This is a convenience function that wraps setup_logger() with default parameters.
    """
    return setup_logger(name)
