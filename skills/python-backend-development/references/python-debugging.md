# Python Debugging

Logging, profiling, and error tracking for FastAPI applications.

## Structured Logging

### Setup with structlog

```python
# app/core/logging.py
"""
Structured logging configuration.
"""
import logging
import sys
from typing import Any

import structlog
from structlog.types import Processor

from app.core.config import settings


def setup_logging() -> None:
    """Configure structured logging."""
    # Shared processors
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.DEBUG:
        # Development: colorful console output
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    else:
        # Production: JSON output
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.LOG_LEVEL)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.getLevelName(settings.LOG_LEVEL),
    )


def get_logger(name: str = __name__) -> structlog.stdlib.BoundLogger:
    """Get a logger instance."""
    return structlog.get_logger(name)
```

### Request Logging Middleware

```python
# app/middleware/logging.py
"""
Request/response logging middleware.
"""
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

from app.core.logging import get_logger

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests and responses."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        # Log request
        start_time = time.perf_counter()
        logger.info(
            "request_started",
            method=request.method,
            path=request.url.path,
            query=str(request.query_params),
            client_ip=request.client.host if request.client else None,
        )

        # Process request
        try:
            response = await call_next(request)
        except Exception as e:
            logger.exception(
                "request_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

        # Log response
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "request_completed",
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
        )

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        return response


# Register in main.py
# app.add_middleware(RequestLoggingMiddleware)
```

### Context-aware Logging

```python
# Usage in services
from app.core.logging import get_logger
import structlog

logger = get_logger(__name__)


class UserService:
    async def create(self, data: UserCreate) -> UserResponse:
        # Bind context for all logs in this operation
        structlog.contextvars.bind_contextvars(
            user_email=data.email,
            operation="create_user",
        )

        logger.info("creating_user")

        try:
            user = await self.repository.create(data)
            logger.info("user_created", user_id=user.id)
            return UserResponse.model_validate(user)
        except Exception as e:
            logger.error("user_creation_failed", error=str(e))
            raise
```

## Error Tracking with Sentry

### Setup

```python
# app/core/sentry.py
"""
Sentry error tracking configuration.
"""
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
from sentry_sdk.integrations.redis import RedisIntegration

from app.core.config import settings


def setup_sentry() -> None:
    """Initialize Sentry SDK."""
    if not settings.SENTRY_DSN:
        return

    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        environment=settings.ENVIRONMENT,
        release=settings.VERSION,
        traces_sample_rate=0.1,  # 10% of transactions
        profiles_sample_rate=0.1,  # 10% of profiled transactions
        integrations=[
            FastApiIntegration(transaction_style="endpoint"),
            SqlalchemyIntegration(),
            RedisIntegration(),
        ],
        # Don't send PII
        send_default_pii=False,
        # Filter sensitive data
        before_send=filter_sensitive_data,
    )


def filter_sensitive_data(event, hint):
    """Filter sensitive data before sending to Sentry."""
    # Remove sensitive headers
    if "request" in event and "headers" in event["request"]:
        headers = event["request"]["headers"]
        sensitive_headers = ["authorization", "x-api-key", "cookie"]
        for header in sensitive_headers:
            if header in headers:
                headers[header] = "[FILTERED]"

    return event
```

### Custom Error Context

```python
# app/api/deps.py
"""
Add user context to Sentry.
"""
import sentry_sdk


async def get_current_user(...) -> UserResponse:
    user = ...  # Get user

    # Set Sentry user context
    sentry_sdk.set_user({
        "id": str(user.id),
        "email": user.email,
        "username": user.username,
    })

    return user
```

## Profiling

### cProfile for CPU Profiling

```python
# scripts/profile_endpoint.py
"""
Profile specific endpoints.
"""
import cProfile
import pstats
import io
from functools import wraps


def profile(func):
    """Decorator to profile a function."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()

        result = await func(*args, **kwargs)

        profiler.disable()

        # Print stats
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats("cumulative")
        stats.print_stats(20)
        print(stream.getvalue())

        return result
    return wrapper


# Usage
@profile
async def slow_endpoint():
    ...
```

### Memory Profiling

```python
# scripts/memory_profile.py
"""
Memory profiling utilities.
"""
import tracemalloc
from functools import wraps


def memory_profile(func):
    """Decorator to profile memory usage."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        tracemalloc.start()

        result = await func(*args, **kwargs)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"Current memory: {current / 1024 / 1024:.2f} MB")
        print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")

        return result
    return wrapper
```

### SQL Query Logging

```python
# app/db/session.py
"""
Enable SQL query logging.
"""
import logging
from sqlalchemy import event
from sqlalchemy.engine import Engine

# Enable SQL logging in debug mode
if settings.DEBUG:
    logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)


# Query timing
@event.listens_for(Engine, "before_cursor_execute")
def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    conn.info.setdefault("query_start_time", []).append(time.perf_counter())


@event.listens_for(Engine, "after_cursor_execute")
def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    total = time.perf_counter() - conn.info["query_start_time"].pop(-1)
    if total > 0.1:  # Log slow queries (>100ms)
        logger.warning(
            "slow_query",
            duration_ms=round(total * 1000, 2),
            query=statement[:200],
        )
```

## Debug Endpoints (Development Only)

```python
# app/api/v1/endpoints/debug.py
"""
Debug endpoints (development only).
"""
from fastapi import APIRouter, Depends, HTTPException
import gc
import sys

from app.core.config import settings

router = APIRouter(prefix="/debug", tags=["Debug"])


def require_debug_mode():
    """Only allow in debug mode."""
    if not settings.DEBUG:
        raise HTTPException(status_code=404, detail="Not found")


@router.get("/memory", dependencies=[Depends(require_debug_mode)])
async def memory_stats():
    """Get memory statistics."""
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()

    return {
        "rss_mb": memory_info.rss / 1024 / 1024,
        "vms_mb": memory_info.vms / 1024 / 1024,
        "python_objects": len(gc.get_objects()),
    }


@router.get("/routes", dependencies=[Depends(require_debug_mode)])
async def list_routes():
    """List all registered routes."""
    from app.main import app

    routes = []
    for route in app.routes:
        if hasattr(route, "methods"):
            routes.append({
                "path": route.path,
                "methods": list(route.methods),
                "name": route.name,
            })

    return {"routes": routes}


@router.post("/gc", dependencies=[Depends(require_debug_mode)])
async def force_gc():
    """Force garbage collection."""
    collected = gc.collect()
    return {"collected_objects": collected}
```

## Exception Handling with Context

```python
# app/core/exceptions.py
"""
Exception classes with debugging context.
"""
from typing import Any, Optional


class AppException(Exception):
    """Base application exception with context."""

    def __init__(
        self,
        message: str,
        code: str = "APP_ERROR",
        context: Optional[dict[str, Any]] = None,
    ):
        self.message = message
        self.code = code
        self.context = context or {}
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "code": self.code,
            "message": self.message,
            "context": self.context,
        }


class NotFoundError(AppException):
    def __init__(self, resource: str, identifier: Any):
        super().__init__(
            message=f"{resource} not found",
            code="NOT_FOUND",
            context={"resource": resource, "identifier": identifier},
        )


# Usage
raise NotFoundError("User", user_id)
# Logs: {"code": "NOT_FOUND", "message": "User not found", "context": {"resource": "User", "identifier": 123}}
```

## Debugging Checklist

- [ ] Structured logging with structlog
- [ ] Request/response logging middleware
- [ ] Request ID tracking
- [ ] Sentry integration for error tracking
- [ ] SQL query logging (slow queries)
- [ ] Memory profiling utilities
- [ ] Debug endpoints (dev only)
- [ ] Context-rich exceptions
- [ ] Log levels properly configured
- [ ] Sensitive data filtered from logs
