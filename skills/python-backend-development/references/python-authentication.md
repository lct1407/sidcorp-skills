# Python Authentication

JWT, API Key, and OAuth 2.0 implementation for FastAPI.

## JWT Authentication

### Security Module

```python
# app/core/security.py
"""
Security utilities for authentication.
"""
import secrets
from datetime import datetime, timedelta
from typing import Any, Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import settings

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

ALGORITHM = "HS256"


def hash_password(password: str) -> str:
    """Hash password using Argon2id."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(
    subject: int | str,
    expires_delta: Optional[timedelta] = None,
    extra_claims: Optional[dict[str, Any]] = None,
) -> str:
    """Create JWT access token."""
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode = {
        "sub": str(subject),
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access",
    }

    if extra_claims:
        to_encode.update(extra_claims)

    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(subject: int | str) -> str:
    """Create JWT refresh token."""
    expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)

    to_encode = {
        "sub": str(subject),
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh",
    }

    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[dict[str, Any]]:
    """Decode and validate JWT token."""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


def generate_api_key() -> str:
    """Generate secure API key."""
    return f"sk_{secrets.token_urlsafe(32)}"
```

### Auth Schemas

```python
# app/schemas/auth.py
"""
Authentication schemas.
"""
from pydantic import BaseModel, EmailStr, Field


class TokenPayload(BaseModel):
    """JWT token payload."""
    sub: str
    exp: int
    iat: int
    type: str


class TokenResponse(BaseModel):
    """Token response for login."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class LoginRequest(BaseModel):
    """Login request body."""
    email: EmailStr
    password: str = Field(..., min_length=8)


class RefreshTokenRequest(BaseModel):
    """Refresh token request."""
    refresh_token: str


class PasswordChangeRequest(BaseModel):
    """Password change request."""
    current_password: str
    new_password: str = Field(..., min_length=8)
```

### Auth Router

```python
# app/api/v1/endpoints/auth.py
"""
Authentication routes.
"""
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db_session, get_current_user
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    verify_password,
    hash_password,
)
from app.services.user import UserService
from app.schemas.auth import (
    LoginRequest,
    TokenResponse,
    RefreshTokenRequest,
    PasswordChangeRequest,
)
from app.schemas.user import UserResponse

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/login", response_model=TokenResponse)
async def login(
    data: LoginRequest,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> TokenResponse:
    """
    Login with email and password.

    Returns access and refresh tokens.
    """
    user_service = UserService(session)

    try:
        user = await user_service.authenticate(data.email, data.password)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    return TokenResponse(
        access_token=create_access_token(user.id),
        refresh_token=create_refresh_token(user.id),
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    data: RefreshTokenRequest,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> TokenResponse:
    """
    Refresh access token using refresh token.
    """
    payload = decode_token(data.refresh_token)

    if not payload or payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )

    user_id = payload.get("sub")
    user_service = UserService(session)

    # Verify user still exists and is active
    try:
        user = await user_service.get(int(user_id))
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    return TokenResponse(
        access_token=create_access_token(user.id),
        refresh_token=create_refresh_token(user.id),
    )


@router.post("/logout")
async def logout(
    current_user: Annotated[UserResponse, Depends(get_current_user)],
) -> dict:
    """
    Logout current user.

    Note: For stateless JWT, client should discard tokens.
    For token revocation, implement token blacklist with Redis.
    """
    # Optional: Add token to blacklist in Redis
    # await redis.setex(f"blacklist:{token}", ttl, "1")
    return {"message": "Successfully logged out"}


@router.post("/change-password")
async def change_password(
    data: PasswordChangeRequest,
    current_user: Annotated[UserResponse, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> dict:
    """
    Change current user's password.
    """
    user_service = UserService(session)

    # Verify current password
    user = await user_service.repository.get_by_id(current_user.id)
    if not verify_password(data.current_password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        )

    # Update password
    user.hashed_password = hash_password(data.new_password)
    await session.commit()

    return {"message": "Password changed successfully"}
```

## API Key Authentication

### Dependency for API Key Auth

```python
# app/api/deps.py (additional)
"""
API Key authentication dependency.
"""
from typing import Annotated, Optional
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.services.api_key import ApiKeyService

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_current_user_from_api_key(
    api_key: Annotated[Optional[str], Security(api_key_header)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> Optional[UserResponse]:
    """Get user from API key if provided."""
    if not api_key:
        return None

    api_key_service = ApiKeyService(session)
    key_data = await api_key_service.validate_key(api_key)

    if not key_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    user_service = UserService(session)
    return await user_service.get(key_data.user_id)
```

### Combined Auth (JWT or API Key)

```python
# app/api/deps.py
"""
Combined authentication: JWT or API Key.
"""
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader

http_bearer = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_current_user(
    bearer_token: Annotated[Optional[HTTPAuthorizationCredentials], Depends(http_bearer)],
    api_key: Annotated[Optional[str], Security(api_key_header)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> UserResponse:
    """
    Get current user from JWT Bearer token or API Key.

    Priority: Bearer token > API Key
    """
    # Try Bearer token first
    if bearer_token:
        payload = decode_token(bearer_token.credentials)
        if payload and payload.get("type") == "access":
            user_service = UserService(session)
            return await user_service.get(int(payload["sub"]))

    # Try API Key
    if api_key:
        api_key_service = ApiKeyService(session)
        key_data = await api_key_service.validate_key(api_key)
        if key_data:
            user_service = UserService(session)
            return await user_service.get(key_data.user_id)

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )
```

## Role-Based Access Control (RBAC)

### Permission Decorator

```python
# app/core/permissions.py
"""
Permission checking utilities.
"""
from functools import wraps
from typing import Callable, List

from fastapi import HTTPException, status

from app.schemas.user import UserResponse


def require_roles(allowed_roles: List[str]):
    """Decorator to check user roles."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, current_user: UserResponse, **kwargs):
            if not any(role in current_user.roles for role in allowed_roles):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions",
                )
            return await func(*args, current_user=current_user, **kwargs)
        return wrapper
    return decorator


def require_scopes(required_scopes: List[str]):
    """Check API key scopes."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, api_key_scopes: List[str] = None, **kwargs):
            if api_key_scopes is None:
                api_key_scopes = []
            if not all(scope in api_key_scopes for scope in required_scopes):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Required scopes: {required_scopes}",
                )
            return await func(*args, **kwargs)
        return wrapper
    return decorator
```

### Permission Dependency

```python
# app/api/deps.py
"""
Permission dependencies.
"""
from typing import List


class PermissionChecker:
    """Dependency for checking permissions."""

    def __init__(self, required_roles: List[str] = None, required_scopes: List[str] = None):
        self.required_roles = required_roles or []
        self.required_scopes = required_scopes or []

    async def __call__(
        self,
        current_user: Annotated[UserResponse, Depends(get_current_user)],
    ) -> UserResponse:
        # Check roles
        if self.required_roles:
            if not hasattr(current_user, 'roles') or not any(
                role in current_user.roles for role in self.required_roles
            ):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions",
                )

        return current_user


# Usage in router
@router.get("/admin/users")
async def list_all_users(
    current_user: Annotated[UserResponse, Depends(PermissionChecker(required_roles=["admin"]))],
    service: Annotated[UserService, Depends(get_user_service)],
):
    """Admin only endpoint."""
    return await service.list()
```

## Token Blacklist (Redis)

```python
# app/core/token_blacklist.py
"""
Token blacklist using Redis.
"""
from typing import Optional
import redis.asyncio as redis

from app.core.config import settings

redis_client: Optional[redis.Redis] = None


async def get_redis() -> redis.Redis:
    """Get Redis connection."""
    global redis_client
    if redis_client is None:
        redis_client = redis.from_url(settings.REDIS_URL)
    return redis_client


async def blacklist_token(token: str, ttl_seconds: int) -> None:
    """Add token to blacklist."""
    client = await get_redis()
    await client.setex(f"blacklist:{token}", ttl_seconds, "1")


async def is_token_blacklisted(token: str) -> bool:
    """Check if token is blacklisted."""
    client = await get_redis()
    return await client.exists(f"blacklist:{token}") > 0
```

## Configuration

```python
# app/core/config.py
"""
Application configuration.
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # JWT
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Database
    DATABASE_URL: str

    # Redis (for token blacklist)
    REDIS_URL: str = "redis://localhost:6379"

    class Config:
        env_file = ".env"


settings = Settings()
```

## Security Best Practices

1. **Use Argon2id** for password hashing (not bcrypt)
2. **Short-lived access tokens** (15-30 minutes)
3. **Rotate refresh tokens** on each use
4. **Implement token blacklist** for logout
5. **Use HTTPS only** in production
6. **Rate limit auth endpoints** (prevent brute force)
7. **Log authentication events** for audit trail
