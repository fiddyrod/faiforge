import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from contextvars import ContextVar

# Context variable for request ID (thread-safe)
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)


class JSONFormatter(logging.Formatter):
    """Format logs as JSON for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        
        # Build base log entry
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add request ID if available
        request_id = request_id_var.get()
        if request_id:
            log_entry["request_id"] = request_id
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


def setup_logging(log_level: str = "INFO", log_format: str = "json") -> logging.Logger:
    """
    Setup application logging.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format type ("json" or "text")
    
    Returns:
        Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger("faiforge")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    
    # Set formatter based on format type
    if log_format == "json":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def log_with_context(
    logger: logging.Logger,
    level: str,
    message: str,
    **extra_fields
) -> None:
    """
    Log a message with additional context fields.
    
    Args:
        logger: Logger instance
        level: Log level (info, warning, error, etc.)
        message: Log message
        **extra_fields: Additional fields to include in log
    """
    
    log_method = getattr(logger, level.lower())
    
    # Create a log record with extra fields
    log_method(
        message,
        extra={'extra_fields': extra_fields}
    )


def get_logger() -> logging.Logger:
    """Get the application logger"""
    return logging.getLogger("faiforge")


def set_request_id(request_id: str) -> None:
    """Set request ID in context"""
    request_id_var.set(request_id)


def get_request_id() -> Optional[str]:
    """Get request ID from context"""
    return request_id_var.get()


# Export middleware
from .middleware import RequestLoggingMiddleware

__all__ = [
    "setup_logging",
    "get_logger",
    "set_request_id",
    "get_request_id",
    "log_with_context",
    "RequestLoggingMiddleware"
]