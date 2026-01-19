# Python Performance

Caching, query optimization, and async patterns for FastAPI.

## Redis Caching

### Setup

```python
# app/core/cache.py
"""
Redis caching utilities.
"""
from typing import Any, Optional
import json
from datetime import timedelta

import redis.asyncio as redis

from app.core.config import settings

redis_client: Optional[redis.Redis] = None


async def get_redis() -> redis.Redis:
    """Get Redis connection."""
    global redis_client
    if redis_client is None:
        redis_client = redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
        )
    return redis_client


async def cache_get(key: str) -> Optional[Any]:
    """Get value from cache."""
    client = await get_redis()
    value = await client.get(key)
    if value:
        return json.loads(value)
    return None


async def cache_set(
    key: str,
    value: Any,
    ttl: int = 300,  # 5 minutes default
) -> None:
    """Set value in cache."""
    client = await get_redis()
    await client.setex(key, ttl, json.dumps(value))


async def cache_delete(key: str) -> None:
    """Delete value from cache."""
    client = await get_redis()
    await client.delete(key)


async def cache_delete_pattern(pattern: str) -> None:
    """Delete all keys matching pattern."""
    client = await get_redis()
    keys = await client.keys(pattern)
    if keys:
        await client.delete(*keys)
```

### Caching Decorator

```python
# app/core/cache.py (continued)
"""
Caching decorator for services.
"""
from functools import wraps
from typing import Callable
import hashlib


def cached(ttl: int = 300, prefix: str = "cache"):
    """Decorator to cache function results."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Build cache key from function name and arguments
            key_parts = [prefix, func.__name__]
            key_parts.extend(str(arg) for arg in args[1:])  # Skip self
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = ":".join(key_parts)

            # Try cache first
            cached_value = await cache_get(cache_key)
            if cached_value is not None:
                return cached_value

            # Call function and cache result
            result = await func(*args, **kwargs)

            # Convert Pydantic models to dict for caching
            if hasattr(result, "model_dump"):
                await cache_set(cache_key, result.model_dump(), ttl)
            else:
                await cache_set(cache_key, result, ttl)

            return result
        return wrapper
    return decorator


# Usage in service
class UserService:
    @cached(ttl=300, prefix="user")
    async def get(self, id: int) -> UserResponse:
        user = await self.repository.get_by_id(id)
        if not user:
            raise NotFoundError(f"User {id} not found")
        return UserResponse.model_validate(user)
```

### Cache Invalidation

```python
# app/services/user.py
"""
Service with cache invalidation.
"""

class UserService:
    async def update(self, id: int, data: UserUpdate) -> UserResponse:
        user = await self.repository.update(id, data)

        # Invalidate cache
        await cache_delete(f"user:get:{id}")
        await cache_delete_pattern(f"user:list:*")

        return UserResponse.model_validate(user)

    async def delete(self, id: int) -> None:
        await self.repository.delete(id)

        # Invalidate cache
        await cache_delete(f"user:get:{id}")
        await cache_delete_pattern(f"user:list:*")
```

## Database Query Optimization

### Select Only Needed Columns

```python
# BAD: Select all columns
result = await session.execute(select(User))

# GOOD: Select only needed columns
from sqlalchemy import select
from sqlalchemy.orm import load_only

result = await session.execute(
    select(User).options(load_only(User.id, User.email, User.username))
)
```

### Eager Loading (Avoid N+1)

```python
# BAD: N+1 queries
users = await session.execute(select(User))
for user in users.scalars():
    print(user.api_keys)  # Each access = 1 query!

# GOOD: Eager loading with selectinload
from sqlalchemy.orm import selectinload

result = await session.execute(
    select(User).options(selectinload(User.api_keys))
)
users = result.scalars().all()
for user in users:
    print(user.api_keys)  # No additional queries
```

### Joinedload vs Selectinload

```python
# joinedload: Single query with JOIN (good for one-to-one)
from sqlalchemy.orm import joinedload

result = await session.execute(
    select(ApiKey).options(joinedload(ApiKey.user))
)

# selectinload: Separate query with IN clause (good for one-to-many)
from sqlalchemy.orm import selectinload

result = await session.execute(
    select(User).options(selectinload(User.api_keys))
)
```

### Pagination with Count

```python
# BAD: Two separate queries
items = await session.execute(
    select(User).offset(skip).limit(limit)
)
count = await session.execute(
    select(func.count()).select_from(User)
)

# GOOD: Use window function for single query (PostgreSQL)
from sqlalchemy import func, over

subq = (
    select(
        User,
        func.count().over().label("total_count")
    )
    .offset(skip)
    .limit(limit)
    .subquery()
)
result = await session.execute(select(subq))
```

### Index Usage

```python
# Ensure indexes on frequently queried columns
class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        Index("ix_users_email", "email"),
        Index("ix_users_username", "username"),
        Index("ix_users_created_at", "created_at"),
        Index("ix_users_active_created", "is_active", "created_at"),  # Composite
    )
```

## Connection Pooling

### SQLAlchemy Async Pool

```python
# app/db/session.py
"""
Database session configuration with connection pooling.
"""
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from app.core.config import settings

engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=10,           # Number of connections to keep open
    max_overflow=20,        # Additional connections when pool is full
    pool_timeout=30,        # Seconds to wait for connection
    pool_recycle=1800,      # Recycle connections after 30 minutes
    pool_pre_ping=True,     # Check connection health before use
    echo=settings.DEBUG,    # Log SQL in debug mode
)

async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db_session():
    """Dependency for database session."""
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()
```

## Async Patterns

### Concurrent Requests

```python
# BAD: Sequential requests
user = await user_service.get(user_id)
orders = await order_service.get_by_user(user_id)
stats = await stats_service.get_user_stats(user_id)

# GOOD: Concurrent requests with asyncio.gather
import asyncio

user, orders, stats = await asyncio.gather(
    user_service.get(user_id),
    order_service.get_by_user(user_id),
    stats_service.get_user_stats(user_id),
)
```

### Background Tasks

```python
# app/api/v1/endpoints/users.py
"""
Background tasks for non-blocking operations.
"""
from fastapi import BackgroundTasks

async def send_welcome_email(email: str, name: str):
    """Background task to send email."""
    # Simulate email sending
    await asyncio.sleep(2)
    print(f"Email sent to {email}")


@router.post("/users", response_model=UserResponse)
async def create_user(
    data: UserCreate,
    background_tasks: BackgroundTasks,
    service: Annotated[UserService, Depends(get_user_service)],
):
    user = await service.create(data)

    # Non-blocking email send
    background_tasks.add_task(send_welcome_email, user.email, user.full_name)

    return user
```

### Async HTTP Client

```python
# app/core/http_client.py
"""
Async HTTP client with connection pooling.
"""
import httpx
from typing import Any

# Reusable client with connection pooling
http_client: httpx.AsyncClient = None


async def get_http_client() -> httpx.AsyncClient:
    """Get HTTP client singleton."""
    global http_client
    if http_client is None:
        http_client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )
    return http_client


async def fetch_external_api(url: str) -> dict[str, Any]:
    """Fetch data from external API."""
    client = await get_http_client()
    response = await client.get(url)
    response.raise_for_status()
    return response.json()
```

## Response Caching with ETags

```python
# app/api/deps.py
"""
ETag caching for GET endpoints.
"""
import hashlib
from fastapi import Request, Response, HTTPException, status


def generate_etag(data: Any) -> str:
    """Generate ETag from response data."""
    content = json.dumps(data, sort_keys=True, default=str)
    return hashlib.md5(content.encode()).hexdigest()


async def check_etag(
    request: Request,
    response: Response,
    data: Any,
) -> Any:
    """Check If-None-Match header and set ETag."""
    etag = generate_etag(data)
    response.headers["ETag"] = f'"{etag}"'

    if_none_match = request.headers.get("If-None-Match")
    if if_none_match and if_none_match.strip('"') == etag:
        raise HTTPException(status_code=status.HTTP_304_NOT_MODIFIED)

    return data


# Usage in endpoint
@router.get("/users/{user_id}")
async def get_user(
    user_id: int,
    request: Request,
    response: Response,
    service: Annotated[UserService, Depends(get_user_service)],
):
    user = await service.get(user_id)
    return await check_etag(request, response, user.model_dump())
```

## Rate Limiting

```python
# app/core/rate_limit.py
"""
Rate limiting with Redis.
"""
from fastapi import HTTPException, Request, status


async def rate_limit(
    request: Request,
    key_prefix: str,
    limit: int,
    window: int = 60,  # seconds
) -> None:
    """Check rate limit for request."""
    client = await get_redis()
    client_ip = request.client.host
    key = f"rate_limit:{key_prefix}:{client_ip}"

    current = await client.incr(key)
    if current == 1:
        await client.expire(key, window)

    if current > limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Max {limit} requests per {window}s",
        )


# Dependency for rate limiting
async def rate_limit_dependency(request: Request):
    await rate_limit(request, "api", limit=100, window=60)
```

## Performance Checklist

- [ ] Redis caching for frequently accessed data
- [ ] Cache invalidation on updates
- [ ] Eager loading to avoid N+1 queries
- [ ] Select only needed columns
- [ ] Proper indexes on query columns
- [ ] Connection pooling configured
- [ ] asyncio.gather for concurrent operations
- [ ] Background tasks for non-blocking operations
- [ ] Rate limiting on API endpoints
- [ ] ETags for response caching
