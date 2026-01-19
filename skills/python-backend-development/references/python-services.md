# Service Layer

## Purpose

- Contains business logic
- Orchestrates repository calls
- Handles validation and authorization
- Raises domain exceptions

## User Service

```python
# app/services/user.py
"""
Service layer for User business logic.
"""
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import hash_password, verify_password
from app.core.exceptions import NotFoundError, ConflictError, UnauthorizedError
from app.repositories.user import UserRepository
from app.schemas.user import (
    UserCreate,
    UserUpdate,
    UserResponse,
    UserList,
)


class UserService:
    """Service for User business logic."""

    def __init__(self, session: AsyncSession):
        self.repository = UserRepository(session)

    async def get(self, id: int) -> UserResponse:
        """Get user by ID."""
        user = await self.repository.get_by_id(id)
        if not user:
            raise NotFoundError(f"User with id {id} not found")
        return UserResponse.model_validate(user)

    async def get_by_email(self, email: str) -> UserResponse:
        """Get user by email."""
        user = await self.repository.get_by_email(email)
        if not user:
            raise NotFoundError(f"User with email {email} not found")
        return UserResponse.model_validate(user)

    async def list(
        self,
        *,
        search: Optional[str] = None,
        is_active: Optional[bool] = None,
        page: int = 1,
        size: int = 20,
    ) -> UserList:
        """Get paginated list of users."""
        skip = (page - 1) * size

        items = await self.repository.get_all(
            skip=skip,
            limit=size,
            search=search,
            is_active=is_active,
        )
        total = await self.repository.count(search=search, is_active=is_active)
        pages = (total + size - 1) // size

        return UserList(
            items=[UserResponse.model_validate(item) for item in items],
            total=total,
            page=page,
            size=size,
            pages=pages,
        )

    async def create(self, data: UserCreate) -> UserResponse:
        """Create new user."""
        # Check email uniqueness
        existing = await self.repository.get_by_email(data.email)
        if existing:
            raise ConflictError(f"Email {data.email} already registered")

        # Check username uniqueness
        existing = await self.repository.get_by_username(data.username)
        if existing:
            raise ConflictError(f"Username {data.username} already taken")

        # Hash password and create
        hashed_password = hash_password(data.password)
        user = await self.repository.create(data, hashed_password)
        return UserResponse.model_validate(user)

    async def update(
        self,
        id: int,
        data: UserUpdate,
        current_user: UserResponse,
    ) -> UserResponse:
        """Update user."""
        # Authorization check
        if current_user.id != id and not current_user.is_superuser:
            raise ForbiddenError("Not authorized to update this user")

        user = await self.repository.get_by_id(id)
        if not user:
            raise NotFoundError(f"User with id {id} not found")

        # Check email uniqueness if changing
        if data.email and data.email != user.email:
            existing = await self.repository.get_by_email(data.email)
            if existing:
                raise ConflictError(f"Email {data.email} already registered")

        # Hash new password if provided
        if data.password:
            data.password = hash_password(data.password)

        user = await self.repository.update(user, data)
        return UserResponse.model_validate(user)

    async def delete(self, id: int, current_user: UserResponse) -> None:
        """Delete user."""
        if current_user.id != id and not current_user.is_superuser:
            raise ForbiddenError("Not authorized to delete this user")

        user = await self.repository.get_by_id(id)
        if not user:
            raise NotFoundError(f"User with id {id} not found")
        await self.repository.delete(user)

    async def authenticate(self, email: str, password: str) -> UserResponse:
        """Authenticate user by email and password."""
        user = await self.repository.get_by_email(email)
        if not user:
            raise UnauthorizedError("Invalid email or password")

        if not verify_password(password, user.hashed_password):
            raise UnauthorizedError("Invalid email or password")

        if not user.is_active:
            raise UnauthorizedError("User account is inactive")

        return UserResponse.model_validate(user)
```

## API Key Service

```python
# app/services/api_key.py
"""
Service layer for API Key business logic.
"""
import secrets
import hashlib
from datetime import datetime
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError, ForbiddenError
from app.repositories.api_key import ApiKeyRepository
from app.schemas.api_key import (
    ApiKeyCreate,
    ApiKeyResponse,
    ApiKeyCreateResponse,
    ApiKeyList,
)


class ApiKeyService:
    """Service for API Key business logic."""

    def __init__(self, session: AsyncSession):
        self.repository = ApiKeyRepository(session)

    def _generate_api_key(self) -> tuple[str, str, str]:
        """Generate API key, hash, and prefix."""
        raw_key = secrets.token_urlsafe(32)
        api_key = f"sk_{raw_key}"
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        key_prefix = api_key[:12]
        return api_key, key_hash, key_prefix

    async def get(self, id: int, user_id: int) -> ApiKeyResponse:
        """Get API key by ID (must belong to user)."""
        api_key = await self.repository.get_by_id(id)
        if not api_key:
            raise NotFoundError(f"API key with id {id} not found")
        if api_key.user_id != user_id:
            raise ForbiddenError("Access denied to this API key")
        return ApiKeyResponse.model_validate(api_key)

    async def list(
        self,
        user_id: int,
        *,
        page: int = 1,
        size: int = 20,
    ) -> ApiKeyList:
        """Get paginated list of user's API keys."""
        skip = (page - 1) * size

        items = await self.repository.get_by_user_id(user_id, skip=skip, limit=size)
        total = await self.repository.count_by_user_id(user_id)
        pages = (total + size - 1) // size

        return ApiKeyList(
            items=[ApiKeyResponse.model_validate(item) for item in items],
            total=total,
            page=page,
            size=size,
            pages=pages,
        )

    async def create(self, user_id: int, data: ApiKeyCreate) -> ApiKeyCreateResponse:
        """Create new API key for user."""
        api_key, key_hash, key_prefix = self._generate_api_key()

        db_obj = await self.repository.create(
            user_id=user_id,
            data=data,
            key_hash=key_hash,
            key_prefix=key_prefix,
        )

        response = ApiKeyCreateResponse.model_validate(db_obj)
        response.api_key = api_key  # Only shown once
        return response

    async def revoke(self, id: int, user_id: int) -> ApiKeyResponse:
        """Revoke API key."""
        api_key = await self.repository.get_by_id(id)
        if not api_key:
            raise NotFoundError(f"API key with id {id} not found")
        if api_key.user_id != user_id:
            raise ForbiddenError("Access denied to this API key")

        api_key = await self.repository.revoke(api_key)
        return ApiKeyResponse.model_validate(api_key)

    async def delete(self, id: int, user_id: int) -> None:
        """Delete API key."""
        api_key = await self.repository.get_by_id(id)
        if not api_key:
            raise NotFoundError(f"API key with id {id} not found")
        if api_key.user_id != user_id:
            raise ForbiddenError("Access denied to this API key")
        await self.repository.delete(api_key)

    async def validate_key(self, raw_key: str) -> Optional[ApiKeyResponse]:
        """Validate API key and return if valid."""
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        api_key = await self.repository.get_by_key_hash(key_hash)

        if not api_key:
            return None
        if not api_key.is_active:
            return None
        if api_key.expires_at and api_key.expires_at < datetime.utcnow():
            return None

        await self.repository.update_last_used(api_key)
        return ApiKeyResponse.model_validate(api_key)
```

## Service Rules

1. **Validate input** - Check business rules before operations
2. **Raise exceptions** - Use domain exceptions, not HTTP errors
3. **Return schemas** - Convert models to response schemas
4. **Check authorization** - Verify user has access to resources
5. **Orchestrate** - Coordinate multiple repositories if needed
