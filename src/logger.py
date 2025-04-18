"""
Structured logging setup for the Trademark AI Agent.
"""

import logging
import os
import json
from typing import Any, Dict, Optional, Union

# Configure logging level from environment variables with default fallback
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
DEBUG = os.environ.get("DEBUG", "false").lower() == "true"

if DEBUG:
    LOG_LEVEL = "DEBUG"

# Create a custom JSON formatter for structured logging
class JsonFormatter(logging.Formatter):
    """
    Custom formatter to output logs in JSON format for structured logging.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string."""
        log_record: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        
        # Add exception info if available
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        
        # Add any additional attributes from record.__dict__
        # that don't have defaults in LogRecord
        for key, value in record.__dict__.items():
            if key not in [
                "args", "asctime", "created", "exc_info", "exc_text", "filename",
                "funcName", "id", "levelname", "levelno", "lineno", "module",
                "msecs", "message", "msg", "name", "pathname", "process",
                "processName", "relativeCreated", "stack_info", "thread", "threadName"
            ]:
                log_record[key] = value
        
        return json.dumps(log_record)

# Create a logger factory
def get_logger(name: str) -> logging.Logger:
    """
    Creates and returns a logger with the specified name,
    configured for structured logging with proper handlers.
    
    Args:
        name: The name of the logger, typically __name__.
        
    Returns:
        A configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers if they already exist
    if logger.handlers:
        return logger
    
    # Set level from environment
    level = getattr(logging, LOG_LEVEL, logging.INFO)
    logger.setLevel(level)
    
    # Create console handler with a JSON formatter
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    
    return logger

# Create a base logger for the application
logger = get_logger("trademark_ai")

# Convenience methods for adding context to logs
def log_with_context(
    level: int,
    msg: str,
    context: Optional[Dict[str, Any]] = None,
    **kwargs: Any
) -> None:
    """
    Log a message with additional context.
    
    Args:
        level: The logging level (e.g., logging.INFO).
        msg: The log message.
        context: Optional dictionary with additional context.
        **kwargs: Additional key-value pairs to include in the log.
    """
    if context:
        kwargs.update(context)
    
    extra = {"context": kwargs} if kwargs else {}
    logger.log(level, msg, extra=extra)

def debug(msg: str, context: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
    """Log a DEBUG level message with context."""
    log_with_context(logging.DEBUG, msg, context, **kwargs)

def info(msg: str, context: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
    """Log an INFO level message with context."""
    log_with_context(logging.INFO, msg, context, **kwargs)

def warning(msg: str, context: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
    """Log a WARNING level message with context."""
    log_with_context(logging.WARNING, msg, context, **kwargs)

def error(msg: str, context: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
    """Log an ERROR level message with context."""
    log_with_context(logging.ERROR, msg, context, **kwargs)

def critical(msg: str, context: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
    """Log a CRITICAL level message with context."""
    log_with_context(logging.CRITICAL, msg, context, **kwargs)

def exception(
    msg: str, 
    exc: Optional[Exception] = None, 
    context: Optional[Dict[str, Any]] = None, 
    **kwargs: Any
) -> None:
    """
    Log an exception with context.
    
    Args:
        msg: The error message.
        exc: The exception object.
        context: Additional context information.
        **kwargs: Additional key-value pairs for the log.
    """
    if exc:
        kwargs["exception_type"] = type(exc).__name__
        kwargs["exception_message"] = str(exc)
    
    log_with_context(logging.ERROR, msg, context, **kwargs)
    if exc:
        logger.exception(msg)  # This adds the traceback 