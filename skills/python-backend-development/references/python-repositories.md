# Repository Pattern

## Purpose
- Encapsulate all database operations
- One repository per model
- Pure data access, no business logic

## User Repository Example

```python
"""
Repository for User database operations.
"""
from typing import Optional, Sequence
from sqlalchemy import select, func, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.schemas.user import UserCreate, UserUpdate


class UserRepository:
    """Repository for User CRUD operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, id: int) -> Optional[User]:
        """Get user by ID."""
        result = await self.session.execute(
            select(User).where(User.id == id)
        )
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        result = await self.session.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()

    async def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        result = await self.session.execute(
            select(User).where(User.username == username)
        )
        return result.scalar_one_or_none()

    async def get_all(
        self,
        *,
        skip: int = 0,
        limit: int = 100,
        search: Optional[str] = None,
        is_active: Optional[bool] = None,
    ) -> Sequence[User]:
        """Get all users with filters and pagination."""
        query = select(User)

        if search:
            query = query.where(
                or_(
                    User.email.ilike(f"%{search}%"),
                    User.username.ilike(f"%{search}%"),
                    User.full_name.ilike(f"%{search}%"),
                )
            )

        if is_active is not None:
            query = query.where(User.is_active == is_active)

        query = query.offset(skip).limit(limit).order_by(User.id.desc())
        result = await self.session.execute(query)
        return result.scalars().all()

    async def count(
        self,
        *,
        search: Optional[str] = None,
        is_active: Optional[bool] = None,
    ) -> int:
        """Count users with filters."""
        query = select(func.count()).select_from(User)

        if search:
            query = query.where(
                or_(
                    User.email.ilike(f"%{search}%"),
                    User.username.ilike(f"%{search}%"),
                )
            )

        if is_active is not None:
            query = query.where(User.is_active == is_active)

        result = await self.session.execute(query)
        return result.scalar_one()

    async def create(self, data: UserCreate, hashed_password: str) -> User:
        """Create new user."""
        db_obj = User(
            email=data.email,
            username=data.username,
            full_name=data.full_name,
            hashed_password=hashed_password,
            is_active=data.is_active,
        )
        self.session.add(db_obj)
        await self.session.commit()
        await self.session.refresh(db_obj)
        return db_obj

    async def update(self, user: User, data: UserUpdate) -> User:
        """Update user."""
        update_data = data.model_dump(exclude_unset=True)

        for field, value in update_data.items():
            setattr(user, field, value)

        await self.session.commit()
        await self.session.refresh(user)
        return user

    async def delete(self, user: User) -> None:
        """Delete user."""
        await self.session.delete(user)
        await self.session.commit()
```

## API Key Repository Example

```python
"""
Repository for API Key database operations.
"""
from typing import Optional, Sequence
from datetime import datetime
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.api_key import ApiKey
from app.schemas.api_key import ApiKeyCreate


class ApiKeyRepository:
    """Repository for API Key CRUD operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, id: int) -> Optional[ApiKey]:
        """Get API key by ID."""
        result = await self.session.execute(
            select(ApiKey).where(ApiKey.id == id)
        )
        return result.scalar_one_or_none()

    async def get_by_key_hash(self, key_hash: str) -> Optional[ApiKey]:
        """Get API key by hashed key value."""
        result = await self.session.execute(
            select(ApiKey).where(ApiKey.key_hash == key_hash)
        )
        return result.scalar_one_or_none()

    async def get_by_user_id(
        self,
        user_id: int,
        *,
        skip: int = 0,
        limit: int = 100,
    ) -> Sequence[ApiKey]:
        """Get all API keys for a user."""
        result = await self.session.execute(
            select(ApiKey)
            .where(ApiKey.user_id == user_id)
            .offset(skip)
            .limit(limit)
            .order_by(ApiKey.created_at.desc())
        )
        return result.scalars().all()

    async def count_by_user_id(self, user_id: int) -> int:
        """Count API keys for a user."""
        result = await self.session.execute(
            select(func.count())
            .select_from(ApiKey)
            .where(ApiKey.user_id == user_id)
        )
        return result.scalar_one()

    async def create(
        self,
        user_id: int,
        data: ApiKeyCreate,
        key_hash: str,
        key_prefix: str,
    ) -> ApiKey:
        """Create new API key."""
        db_obj = ApiKey(
            user_id=user_id,
            name=data.name,
            description=data.description,
            key_hash=key_hash,
            key_prefix=key_prefix,
            scopes=data.scopes,
            expires_at=data.expires_at,
            is_active=data.is_active,
        )
        self.session.add(db_obj)
        await self.session.commit()
        await self.session.refresh(db_obj)
        return db_obj

    async def update_last_used(self, api_key: ApiKey) -> None:
        """Update last used timestamp."""
        api_key.last_used_at = datetime.utcnow()
        await self.session.commit()

    async def delete(self, api_key: ApiKey) -> None:
        """Delete API key."""
        await self.session.delete(api_key)
        await self.session.commit()

    async def revoke(self, api_key: ApiKey) -> ApiKey:
        """Revoke (deactivate) API key."""
        api_key.is_active = False
        await self.session.commit()
        await self.session.refresh(api_key)
        return api_key
```

## Repository Rules

1. **No business logic** - Only database operations
2. **Return models** - Not schemas
3. **Accept schemas** - For create/update input
4. **Handle None** - Return Optional for single items
5. **Pagination** - Always support skip/limit
6. **Commit in repository** - Each operation commits its changes
