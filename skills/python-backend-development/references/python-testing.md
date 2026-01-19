# Python Testing

pytest testing patterns for FastAPI applications.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── unit/                    # Unit tests (70%)
│   ├── test_services.py
│   └── test_utils.py
├── integration/             # Integration tests (20%)
│   ├── test_repositories.py
│   └── test_api.py
└── e2e/                     # End-to-end tests (10%)
    └── test_workflows.py
```

## Fixtures (conftest.py)

```python
# tests/conftest.py
"""
Shared test fixtures.
"""
import asyncio
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.db.base import Base
from app.api.deps import get_db_session
from app.models.user import User
from app.core.security import hash_password, create_access_token


# Test database URL (SQLite in memory)
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create test database session."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    async with async_session() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Create test HTTP client."""

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db_session] = override_get_db

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def test_user(db_session: AsyncSession) -> User:
    """Create test user."""
    user = User(
        email="test@example.com",
        username="testuser",
        hashed_password=hash_password("testpassword123"),
        full_name="Test User",
        is_active=True,
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest_asyncio.fixture
async def auth_headers(test_user: User) -> dict[str, str]:
    """Create authenticated headers."""
    token = create_access_token(test_user.id)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def user_data() -> dict:
    """Sample user data for testing."""
    return {
        "email": "newuser@example.com",
        "username": "newuser",
        "password": "securepassword123",
        "full_name": "New User",
    }
```

## Unit Tests

### Service Tests

```python
# tests/unit/test_user_service.py
"""
Unit tests for UserService.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.services.user import UserService
from app.schemas.user import UserCreate, UserUpdate
from app.core.exceptions import NotFoundError, ConflictError


class TestUserService:
    """Test cases for UserService."""

    @pytest.fixture
    def mock_session(self):
        """Create mock database session."""
        return AsyncMock()

    @pytest.fixture
    def user_service(self, mock_session):
        """Create UserService with mock session."""
        return UserService(mock_session)

    @pytest.mark.asyncio
    async def test_get_user_success(self, user_service):
        """Test getting user by ID."""
        # Arrange
        mock_user = MagicMock()
        mock_user.id = 1
        mock_user.email = "test@example.com"
        user_service.repository.get_by_id = AsyncMock(return_value=mock_user)

        # Act
        result = await user_service.get(1)

        # Assert
        assert result.id == 1
        user_service.repository.get_by_id.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_get_user_not_found(self, user_service):
        """Test getting non-existent user."""
        # Arrange
        user_service.repository.get_by_id = AsyncMock(return_value=None)

        # Act & Assert
        with pytest.raises(NotFoundError):
            await user_service.get(999)

    @pytest.mark.asyncio
    async def test_create_user_success(self, user_service):
        """Test creating new user."""
        # Arrange
        user_data = UserCreate(
            email="new@example.com",
            username="newuser",
            password="password123",
        )
        user_service.repository.get_by_email = AsyncMock(return_value=None)
        user_service.repository.get_by_username = AsyncMock(return_value=None)

        mock_user = MagicMock()
        mock_user.id = 1
        mock_user.email = user_data.email
        user_service.repository.create = AsyncMock(return_value=mock_user)

        # Act
        result = await user_service.create(user_data)

        # Assert
        assert result.email == user_data.email
        user_service.repository.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_user_duplicate_email(self, user_service):
        """Test creating user with duplicate email."""
        # Arrange
        user_data = UserCreate(
            email="existing@example.com",
            username="newuser",
            password="password123",
        )
        user_service.repository.get_by_email = AsyncMock(return_value=MagicMock())

        # Act & Assert
        with pytest.raises(ConflictError):
            await user_service.create(user_data)
```

### Utility Tests

```python
# tests/unit/test_security.py
"""
Unit tests for security utilities.
"""
import pytest
from datetime import timedelta

from app.core.security import (
    hash_password,
    verify_password,
    create_access_token,
    decode_token,
)


class TestPasswordHashing:
    """Test password hashing functions."""

    def test_hash_password(self):
        """Test password hashing."""
        password = "testpassword123"
        hashed = hash_password(password)

        assert hashed != password
        assert hashed.startswith("$argon2")

    def test_verify_password_correct(self):
        """Test verifying correct password."""
        password = "testpassword123"
        hashed = hash_password(password)

        assert verify_password(password, hashed) is True

    def test_verify_password_incorrect(self):
        """Test verifying incorrect password."""
        hashed = hash_password("correct_password")

        assert verify_password("wrong_password", hashed) is False


class TestJWT:
    """Test JWT token functions."""

    def test_create_access_token(self):
        """Test creating access token."""
        token = create_access_token(subject=123)

        assert token is not None
        assert isinstance(token, str)

    def test_decode_token_valid(self):
        """Test decoding valid token."""
        token = create_access_token(subject=123)
        payload = decode_token(token)

        assert payload is not None
        assert payload["sub"] == "123"
        assert payload["type"] == "access"

    def test_decode_token_expired(self):
        """Test decoding expired token."""
        token = create_access_token(
            subject=123,
            expires_delta=timedelta(seconds=-1),
        )
        payload = decode_token(token)

        assert payload is None

    def test_decode_token_invalid(self):
        """Test decoding invalid token."""
        payload = decode_token("invalid.token.here")

        assert payload is None
```

## Integration Tests

### API Tests

```python
# tests/integration/test_users_api.py
"""
Integration tests for Users API.
"""
import pytest
from httpx import AsyncClient

from app.models.user import User


class TestUsersAPI:
    """Test cases for Users API endpoints."""

    @pytest.mark.asyncio
    async def test_create_user(self, client: AsyncClient, user_data: dict):
        """Test creating a new user."""
        response = await client.post("/api/v1/users", json=user_data)

        assert response.status_code == 201
        data = response.json()
        assert data["email"] == user_data["email"]
        assert data["username"] == user_data["username"]
        assert "id" in data
        assert "password" not in data

    @pytest.mark.asyncio
    async def test_create_user_duplicate_email(
        self,
        client: AsyncClient,
        test_user: User,
    ):
        """Test creating user with existing email."""
        response = await client.post("/api/v1/users", json={
            "email": test_user.email,
            "username": "different_username",
            "password": "password123",
        })

        assert response.status_code == 409

    @pytest.mark.asyncio
    async def test_get_user(
        self,
        client: AsyncClient,
        test_user: User,
        auth_headers: dict,
    ):
        """Test getting user by ID."""
        response = await client.get(
            f"/api/v1/users/{test_user.id}",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == test_user.id
        assert data["email"] == test_user.email

    @pytest.mark.asyncio
    async def test_get_user_not_found(
        self,
        client: AsyncClient,
        auth_headers: dict,
    ):
        """Test getting non-existent user."""
        response = await client.get(
            "/api/v1/users/99999",
            headers=auth_headers,
        )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_list_users(
        self,
        client: AsyncClient,
        test_user: User,
        auth_headers: dict,
    ):
        """Test listing users with pagination."""
        response = await client.get(
            "/api/v1/users?page=1&size=10",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert "page" in data
        assert len(data["items"]) >= 1

    @pytest.mark.asyncio
    async def test_update_user(
        self,
        client: AsyncClient,
        test_user: User,
        auth_headers: dict,
    ):
        """Test updating user."""
        response = await client.patch(
            f"/api/v1/users/{test_user.id}",
            headers=auth_headers,
            json={"full_name": "Updated Name"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["full_name"] == "Updated Name"

    @pytest.mark.asyncio
    async def test_delete_user(
        self,
        client: AsyncClient,
        test_user: User,
        auth_headers: dict,
    ):
        """Test deleting user."""
        response = await client.delete(
            f"/api/v1/users/{test_user.id}",
            headers=auth_headers,
        )

        assert response.status_code == 204

        # Verify deleted
        response = await client.get(
            f"/api/v1/users/{test_user.id}",
            headers=auth_headers,
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_unauthorized_access(self, client: AsyncClient):
        """Test accessing protected endpoint without auth."""
        response = await client.get("/api/v1/users")

        assert response.status_code == 401
```

### Auth API Tests

```python
# tests/integration/test_auth_api.py
"""
Integration tests for Auth API.
"""
import pytest
from httpx import AsyncClient

from app.models.user import User


class TestAuthAPI:
    """Test cases for Auth API endpoints."""

    @pytest.mark.asyncio
    async def test_login_success(
        self,
        client: AsyncClient,
        test_user: User,
    ):
        """Test successful login."""
        response = await client.post("/api/v1/auth/login", json={
            "email": test_user.email,
            "password": "testpassword123",
        })

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_login_invalid_password(
        self,
        client: AsyncClient,
        test_user: User,
    ):
        """Test login with wrong password."""
        response = await client.post("/api/v1/auth/login", json={
            "email": test_user.email,
            "password": "wrongpassword",
        })

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_login_invalid_email(self, client: AsyncClient):
        """Test login with non-existent email."""
        response = await client.post("/api/v1/auth/login", json={
            "email": "nonexistent@example.com",
            "password": "password123",
        })

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_refresh_token(
        self,
        client: AsyncClient,
        test_user: User,
    ):
        """Test refreshing access token."""
        # Login first
        login_response = await client.post("/api/v1/auth/login", json={
            "email": test_user.email,
            "password": "testpassword123",
        })
        refresh_token = login_response.json()["refresh_token"]

        # Refresh
        response = await client.post("/api/v1/auth/refresh", json={
            "refresh_token": refresh_token,
        })

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific file
pytest tests/integration/test_users_api.py

# Run specific test
pytest tests/integration/test_users_api.py::TestUsersAPI::test_create_user

# Run with verbose output
pytest -v --tb=short

# Run only unit tests
pytest tests/unit/

# Run async tests
pytest -v --asyncio-mode=auto
```

## pytest.ini Configuration

```ini
# pytest.ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
asyncio_mode = auto
addopts = -v --tb=short --strict-markers
markers =
    slow: marks tests as slow
    integration: marks tests as integration tests
```

## Test Rules

1. **70-20-10** - Unit (70%), Integration (20%), E2E (10%)
2. **Arrange-Act-Assert** - Structure all tests clearly
3. **One assertion per test** when possible
4. **Use fixtures** for common setup
5. **Mock external dependencies** in unit tests
6. **Use real database** in integration tests (SQLite in-memory)
7. **Test edge cases** - empty, null, duplicates, errors
