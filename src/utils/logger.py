"""
Logging utility module.
Configures logging from project settings and outputs to both file and console.
"""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler


_logger_instance = None


def setup_logger(name: str = "DataFactory") -> logging.Logger:

    from utils.config_loader import get_config

    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers.
    if logger.handlers:
        return logger

    # Load logging settings from config.
    log_level = get_config("logging.level", "INFO")
    log_format = get_config("logging.format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_file = get_config("logging.file_path", "logs/app.log")
    max_bytes = get_config("logging.max_file_size_mb", 10) * 1024 * 1024
    backup_count = get_config("logging.backup_count", 5)
    console_enabled = get_config("logging.console", True)

    # Set logger level.
    logger.setLevel(getattr(logging, log_level.upper()))

    # Build formatter.
    formatter = logging.Formatter(log_format)

    # Configure file handler with log rotation.
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        logger.addHandler(file_handler)

    # Configure console handler.
    if console_enabled:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        logger.addHandler(console_handler)

    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance with lazy initialization.

    Args:
        name: Logger name. If None, return the root logger.

    Returns:
        Logger instance.
    """
    global _logger_instance

    if _logger_instance is None:
        _logger_instance = setup_logger()

    if name:
        return _logger_instance.getChild(name)
    return _logger_instance


# Provide a convenient lazy logger proxy.
class LazyLogger:
    """Lazy-initialized logger wrapper."""

    def __getattr__(self, name):
        global _logger_instance
        if _logger_instance is None:
            _logger_instance = setup_logger()
        return getattr(_logger_instance, name)


logger = LazyLogger()
