# Pydantic Schemas

## Schema Types

| Schema | Purpose | Usage |
|--------|---------|-------|
| `Base` | Common fields | Internal inheritance |
| `Create` | POST request body | Creating new records |
| `Update` | PATCH request body | Partial updates |
| `Response` | API response | Single item response |
| `List` | Paginated response | List with metadata |

## User Schemas Example

```python
"""
Pydantic schemas for User.
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, EmailStr


class UserBase(BaseModel):
    """Base schema with common fields."""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    full_name: Optional[str] = Field(None, max_length=255)
    is_active: bool = True


class UserCreate(UserBase):
    """Schema for creating a new user."""
    password: str = Field(..., min_length=8)


class UserUpdate(BaseModel):
    """Schema for updating user. All fields optional."""
    email: Optional[EmailStr] = None
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    full_name: Optional[str] = Field(None, max_length=255)
    is_active: Optional[bool] = None
    password: Optional[str] = Field(None, min_length=8)


class UserResponse(UserBase):
    """Schema for user API response."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None


class UserList(BaseModel):
    """Paginated list of users."""
    items: list[UserResponse]
    total: int
    page: int
    size: int
    pages: int
```

## API Key Schemas Example

```python
"""
Pydantic schemas for API Key.
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class ApiKeyBase(BaseModel):
    """Base schema for API Key."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    is_active: bool = True


class ApiKeyCreate(ApiKeyBase):
    """Schema for creating API Key."""
    expires_at: Optional[datetime] = None
    scopes: list[str] = Field(default_factory=list)


class ApiKeyUpdate(BaseModel):
    """Schema for updating API Key."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    is_active: Optional[bool] = None
    scopes: Optional[list[str]] = None


class ApiKeyResponse(ApiKeyBase):
    """Schema for API Key response (without secret)."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    key_prefix: str  # First 8 chars of key for identification
    scopes: list[str]
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    created_at: datetime


class ApiKeyCreateResponse(ApiKeyResponse):
    """Response when creating API Key (includes full key, shown once)."""
    api_key: str  # Full key, only shown on creation


class ApiKeyList(BaseModel):
    """Paginated list of API Keys."""
    items: list[ApiKeyResponse]
    total: int
    page: int
    size: int
    pages: int
```

## Nested Response Example

```python
class UserWithApiKeysResponse(UserResponse):
    """User with their API Keys."""
    api_keys: list[ApiKeyResponse] = []
```

## Query Parameters

```python
class UserFilter(BaseModel):
    """Query parameters for filtering users."""
    search: Optional[str] = None
    is_active: Optional[bool] = None


class PaginationParams(BaseModel):
    """Pagination parameters."""
    page: int = Field(1, ge=1)
    size: int = Field(20, ge=1, le=100)
```
