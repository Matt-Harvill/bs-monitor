"""Centralized logging configuration for multi-classifier training."""

import logging
import sys
from pathlib import Path


class MultiClassifierLogger:
    """Centralized logger for multi-classifier training."""

    _instance = None
    _logger = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def setup(self, log_level: str = "INFO", log_file: str | None = None):
        """Setup logging configuration."""
        if self._logger is not None:
            return self._logger

        # Create logger
        self._logger = logging.getLogger("multi_classifier")
        self._logger.setLevel(getattr(logging, log_level.upper()))

        # Clear any existing handlers
        self._logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)

        # File handler (if specified)
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(getattr(logging, log_level.upper()))
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)

        return self._logger

    def get_logger(self):
        """Get the logger instance."""
        if self._logger is None:
            return self.setup()
        return self._logger


# Global logger instance
logger_instance = MultiClassifierLogger()


def get_logger():
    """Get the global logger instance."""
    return logger_instance.get_logger()


def setup_logging(log_level: str = "INFO", log_file: str | None = None):
    """Setup logging configuration."""
    return logger_instance.setup(log_level, log_file)
