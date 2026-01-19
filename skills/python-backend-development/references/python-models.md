# SQLAlchemy Models

## Base Model Setup

```python
# app/db/base.py
from datetime import datetime
from typing import Optional
from sqlalchemy import DateTime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


class TimestampMixin:
    """Mixin for created_at and updated_at fields."""
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, onupdate=datetime.utcnow, nullable=True
    )
```

## User Model

```python
# app/models/user.py
"""
User model for authentication and user management.
"""
from datetime import datetime
from typing import Optional, TYPE_CHECKING
from sqlalchemy import String, Integer, DateTime, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.api_key import ApiKey


class User(Base, TimestampMixin):
    """SQLAlchemy model for users table."""

    __tablename__ = "users"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Required fields
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)

    # Optional fields
    full_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False)

    # Relationships
    api_keys: Mapped[list["ApiKey"]] = relationship(
        "ApiKey",
        back_populates="user",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email})>"
```

## API Key Model

```python
# app/models/api_key.py
"""
API Key model for API authentication.
"""
from datetime import datetime
from typing import Optional, TYPE_CHECKING
from sqlalchemy import String, Integer, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.user import User


class ApiKey(Base, TimestampMixin):
    """SQLAlchemy model for api_keys table."""

    __tablename__ = "api_keys"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Foreign key
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Key fields
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    key_prefix: Mapped[str] = mapped_column(String(12), nullable=False)

    # Permissions and status
    scopes: Mapped[list[str]] = mapped_column(JSON, default=list)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Expiration and usage
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="api_keys")

    def __repr__(self) -> str:
        return f"<ApiKey(id={self.id}, name={self.name}, prefix={self.key_prefix})>"
```

## Field Type Mapping

| Python Type | SQLAlchemy Type | Notes |
|-------------|-----------------|-------|
| `str` | `String(n)` | Specify max length |
| `str` (long) | `Text` | Unlimited text |
| `int` | `Integer` | Standard integer |
| `int` (big) | `BigInteger` | Large numbers |
| `float` | `Float` | Floating point |
| `Decimal` | `Numeric(p, s)` | Precision decimals |
| `bool` | `Boolean` | True/False |
| `datetime` | `DateTime` | Timestamps |
| `date` | `Date` | Date only |
| `UUID` | `UUID` | Unique identifiers |
| `dict`/`list` | `JSON` | JSON data |

## Relationships

### One-to-Many (User → ApiKeys)
```python
# Parent side (User)
api_keys: Mapped[list["ApiKey"]] = relationship(
    "ApiKey",
    back_populates="user",
    cascade="all, delete-orphan",
)

# Child side (ApiKey)
user_id: Mapped[int] = mapped_column(
    Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
)
user: Mapped["User"] = relationship("User", back_populates="api_keys")
```

### Many-to-Many (User → Roles)
```python
from sqlalchemy import Table, Column

user_roles = Table(
    "user_roles",
    Base.metadata,
    Column("user_id", ForeignKey("users.id"), primary_key=True),
    Column("role_id", ForeignKey("roles.id"), primary_key=True),
)

# On User model
roles: Mapped[list["Role"]] = relationship(
    "Role", secondary=user_roles, back_populates="users"
)
```

## Indexes

```python
from sqlalchemy import Index

class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        Index("ix_users_email_active", "email", "is_active"),
    )
```

## Soft Delete Pattern

```python
class SoftDeleteMixin:
    """Mixin for soft delete functionality."""
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True, default=None
    )

    @property
    def is_deleted(self) -> bool:
        return self.deleted_at is not None
```

## Model Registry

```python
# app/models/__init__.py
from app.models.user import User
from app.models.api_key import ApiKey

__all__ = ["User", "ApiKey"]
```
