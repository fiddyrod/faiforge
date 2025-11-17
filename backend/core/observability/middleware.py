import uuid
import time
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .import set_request_id, get_logger, log_with_context


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all requests with correlation IDs"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.logger = get_logger()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        # Set request ID in context (available to all logs)
        set_request_id(request_id)
        
        # Start timing
        start_time = time.time()
        
        # Log incoming request
        log_with_context(
            self.logger,
            "info",
            "Request received",
            event="request_start",
            method=request.method,
            path=request.url.path,
            client_host=request.client.host if request.client else None
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Log successful response
            log_with_context(
                self.logger,
                "info",
                "Request completed",
                event="request_complete",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                latency_ms=round(latency_ms, 2)
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
        
        except Exception as e:
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Log error
            log_with_context(
                self.logger,
                "error",
                f"Request failed: {str(e)}",
                event="request_error",
                method=request.method,
                path=request.url.path,
                error=str(e),
                latency_ms=round(latency_ms, 2)
            )
            
            # Re-raise to let FastAPI handle it
            raise