# Backend API Design

Comprehensive guide to designing RESTful APIs with Python FastAPI.

## REST API Design

### Resource-Based URLs

**Good:**

```
GET    /api/v1/users              # List users
GET    /api/v1/users/{id}         # Get specific user
POST   /api/v1/users              # Create user
PATCH  /api/v1/users/{id}         # Update user (partial)
DELETE /api/v1/users/{id}         # Delete user

GET    /api/v1/users/{id}/api-keys    # Get user's API keys
POST   /api/v1/users/{id}/api-keys    # Create API key for user
```

**Bad (Avoid):**

```
GET /api/v1/getUser?id=123        # RPC-style, not RESTful
POST /api/v1/createUser           # Verb in URL
GET /api/v1/user-api-keys         # Unclear relationship
```

### HTTP Status Codes

**Success:**

- `200 OK` - Successful GET, PATCH
- `201 Created` - Successful POST (resource created)
- `204 No Content` - Successful DELETE

**Client Errors:**

- `400 Bad Request` - Invalid input/validation error
- `401 Unauthorized` - Missing or invalid authentication
- `403 Forbidden` - Authenticated but not authorized
- `404 Not Found` - Resource doesn't exist
- `409 Conflict` - Resource conflict (duplicate email)
- `422 Unprocessable Entity` - Validation error (detailed)
- `429 Too Many Requests` - Rate limit exceeded

**Server Errors:**

- `500 Internal Server Error` - Generic server error
- `502 Bad Gateway` - Upstream service error
- `503 Service Unavailable` - Temporary downtime

## FastAPI Implementation

### User Router

```python
# app/api/v1/endpoints/users.py
"""
API routes for User management.
"""
from typing import Annotated
from fastapi import APIRouter, Depends, Query, status

from app.api.deps import get_db_session, get_current_user
from app.services.user import UserService
from app.schemas.user import (
    UserCreate,
    UserUpdate,
    UserResponse,
    UserList,
)
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/users", tags=["Users"])


def get_user_service(
    session: Annotated[AsyncSession, Depends(get_db_session)]
) -> UserService:
    return UserService(session)


@router.get("", response_model=UserList)
async def list_users(
    service: Annotated[UserService, Depends(get_user_service)],
    search: str | None = Query(None, max_length=100),
    is_active: bool | None = None,
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
) -> UserList:
    """
    Get paginated list of users.

    - **search**: Filter by email, username, or full name
    - **is_active**: Filter by active status
    - **page**: Page number (default: 1)
    - **size**: Items per page (default: 20, max: 100)
    """
    return await service.list(
        search=search,
        is_active=is_active,
        page=page,
        size=size,
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    current_user: Annotated[UserResponse, Depends(get_current_user)],
) -> UserResponse:
    """Get current authenticated user's profile."""
    return current_user


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    service: Annotated[UserService, Depends(get_user_service)],
) -> UserResponse:
    """Get user by ID."""
    return await service.get(user_id)


@router.post("", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    data: UserCreate,
    service: Annotated[UserService, Depends(get_user_service)],
) -> UserResponse:
    """
    Create new user.

    - **email**: Valid email address (unique)
    - **username**: 3-50 alphanumeric characters (unique)
    - **password**: Minimum 8 characters
    """
    return await service.create(data)


@router.patch("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    data: UserUpdate,
    service: Annotated[UserService, Depends(get_user_service)],
    current_user: Annotated[UserResponse, Depends(get_current_user)],
) -> UserResponse:
    """Update user. Only the user themselves or admin can update."""
    return await service.update(user_id, data, current_user)


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: int,
    service: Annotated[UserService, Depends(get_user_service)],
    current_user: Annotated[UserResponse, Depends(get_current_user)],
) -> None:
    """Delete user. Only the user themselves or admin can delete."""
    await service.delete(user_id, current_user)
```

### API Key Router

```python
# app/api/v1/endpoints/api_keys.py
"""
API routes for API Key management.
"""
from typing import Annotated
from fastapi import APIRouter, Depends, Query, status

from app.api.deps import get_db_session, get_current_user
from app.services.api_key import ApiKeyService
from app.schemas.api_key import (
    ApiKeyCreate,
    ApiKeyResponse,
    ApiKeyCreateResponse,
    ApiKeyList,
)
from app.schemas.user import UserResponse
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/api-keys", tags=["API Keys"])


def get_api_key_service(
    session: Annotated[AsyncSession, Depends(get_db_session)]
) -> ApiKeyService:
    return ApiKeyService(session)


@router.get("", response_model=ApiKeyList)
async def list_api_keys(
    service: Annotated[ApiKeyService, Depends(get_api_key_service)],
    current_user: Annotated[UserResponse, Depends(get_current_user)],
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
) -> ApiKeyList:
    """
    List current user's API keys.

    Note: The full key is only shown when created.
    """
    return await service.list(current_user.id, page=page, size=size)


@router.get("/{key_id}", response_model=ApiKeyResponse)
async def get_api_key(
    key_id: int,
    service: Annotated[ApiKeyService, Depends(get_api_key_service)],
    current_user: Annotated[UserResponse, Depends(get_current_user)],
) -> ApiKeyResponse:
    """Get API key details by ID."""
    return await service.get(key_id, current_user.id)


@router.post("", response_model=ApiKeyCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    data: ApiKeyCreate,
    service: Annotated[ApiKeyService, Depends(get_api_key_service)],
    current_user: Annotated[UserResponse, Depends(get_current_user)],
) -> ApiKeyCreateResponse:
    """
    Create new API key.

    **Important**: The full API key is only shown once in the response.
    Store it securely as it cannot be retrieved later.

    - **name**: Friendly name for the key
    - **scopes**: List of permission scopes
    - **expires_at**: Optional expiration datetime
    """
    return await service.create(current_user.id, data)


@router.post("/{key_id}/revoke", response_model=ApiKeyResponse)
async def revoke_api_key(
    key_id: int,
    service: Annotated[ApiKeyService, Depends(get_api_key_service)],
    current_user: Annotated[UserResponse, Depends(get_current_user)],
) -> ApiKeyResponse:
    """Revoke (deactivate) an API key."""
    return await service.revoke(key_id, current_user.id)


@router.delete("/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_api_key(
    key_id: int,
    service: Annotated[ApiKeyService, Depends(get_api_key_service)],
    current_user: Annotated[UserResponse, Depends(get_current_user)],
) -> None:
    """Permanently delete an API key."""
    await service.delete(key_id, current_user.id)
```

### Dependencies

```python
# app/api/deps.py
"""
FastAPI dependencies for injection.
"""
from typing import Annotated, AsyncGenerator
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import async_session_maker
from app.core.security import decode_token
from app.services.user import UserService
from app.services.api_key import ApiKeyService
from app.schemas.user import UserResponse

security = HTTPBearer()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session."""
    async with async_session_maker() as session:
        yield session


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> UserResponse:
    """Get current authenticated user from JWT or API key."""
    token = credentials.credentials

    # Try JWT first
    if token.startswith("eyJ"):
        payload = decode_token(token)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
            )
        user_service = UserService(session)
        return await user_service.get(payload["sub"])

    # Try API key (sk_...)
    if token.startswith("sk_"):
        api_key_service = ApiKeyService(session)
        api_key = await api_key_service.validate_key(token)
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired API key",
            )
        user_service = UserService(session)
        return await user_service.get(api_key.user_id)

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
    )
```

### Error Handling

```python
# app/core/exceptions.py
"""
Custom exceptions for the application.
"""


class AppException(Exception):
    """Base application exception."""

    def __init__(self, message: str, code: str = "APP_ERROR"):
        self.message = message
        self.code = code
        super().__init__(message)


class NotFoundError(AppException):
    """Resource not found."""

    def __init__(self, message: str = "Resource not found"):
        super().__init__(message, "NOT_FOUND")


class ConflictError(AppException):
    """Resource conflict (e.g., duplicate)."""

    def __init__(self, message: str = "Resource already exists"):
        super().__init__(message, "CONFLICT")


class UnauthorizedError(AppException):
    """Authentication failed."""

    def __init__(self, message: str = "Unauthorized"):
        super().__init__(message, "UNAUTHORIZED")


class ForbiddenError(AppException):
    """Access denied."""

    def __init__(self, message: str = "Access denied"):
        super().__init__(message, "FORBIDDEN")
```

```python
# app/core/exception_handlers.py
"""
Exception handlers for FastAPI.
"""
from fastapi import Request, status
from fastapi.responses import JSONResponse

from app.core.exceptions import (
    AppException,
    NotFoundError,
    ConflictError,
    UnauthorizedError,
    ForbiddenError,
)


async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    """Handle application exceptions."""
    status_map = {
        NotFoundError: status.HTTP_404_NOT_FOUND,
        ConflictError: status.HTTP_409_CONFLICT,
        UnauthorizedError: status.HTTP_401_UNAUTHORIZED,
        ForbiddenError: status.HTTP_403_FORBIDDEN,
    }

    status_code = status_map.get(type(exc), status.HTTP_400_BAD_REQUEST)

    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
            }
        },
    )
```

```python
# app/main.py (register handlers)
from fastapi import FastAPI
from app.core.exceptions import AppException
from app.core.exception_handlers import app_exception_handler

app = FastAPI(title="AI Services API", version="1.0.0")

app.add_exception_handler(AppException, app_exception_handler)
```

### Router Registration

```python
# app/api/v1/router.py
"""
API v1 router aggregation.
"""
from fastapi import APIRouter

from app.api.v1.endpoints import auth, users, api_keys

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(auth.router)
api_router.include_router(users.router)
api_router.include_router(api_keys.router)
```

```python
# app/main.py
from fastapi import FastAPI
from app.api.v1.router import api_router

app = FastAPI(title="AI Services API", version="1.0.0")
app.include_router(api_router)
```

## Response Format

### Success Response

```json
{
  "id": 123,
  "email": "user@example.com",
  "username": "johndoe",
  "full_name": "John Doe",
  "is_active": true,
  "created_at": "2026-01-17T12:00:00Z",
  "updated_at": "2026-01-17T12:00:00Z"
}
```

### Paginated Response

```json
{
  "items": [...],
  "total": 1234,
  "page": 2,
  "size": 20,
  "pages": 62
}
```

### Error Response

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": [
      {
        "field": "email",
        "message": "Invalid email format"
      }
    ]
  }
}
```

## API Security Checklist

- [ ] HTTPS/TLS only
- [ ] Authentication (JWT + API keys)
- [ ] Authorization (check permissions)
- [ ] Rate limiting
- [ ] Input validation (Pydantic)
- [ ] CORS configured
- [ ] Error messages don't leak system info
- [ ] Audit logging

## API Versioning

Use URL versioning:

```
/api/v1/users
/api/v2/users
```

## OpenAPI Documentation

FastAPI auto-generates OpenAPI docs at:

- Swagger UI: `/docs`
- ReDoc: `/redoc`
- OpenAPI JSON: `/openapi.json`
