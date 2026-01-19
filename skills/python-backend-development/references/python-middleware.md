# Python Middleware

FastAPI middleware patterns for authentication, rate limiting, and credit management.

## Middleware Architecture

```
Request → Auth Middleware → Rate Limit → Credit Check → Router → Response
              ↓                ↓             ↓
         API Key/JWT      Redis Counter   DB Credit
```

## Dual Authentication (API Key + JWT)

### Auth Dependencies

```python
# app/api/deps.py
"""
Authentication dependencies supporting both API Key and JWT.
"""
from typing import Annotated, Optional

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.security import decode_token
from app.models.user import User
from app.models.api_key import ApiKey
from app.repositories.user import UserRepository
from app.repositories.api_key import ApiKeyRepository
from app.db.session import get_db_session

security = HTTPBearer(auto_error=False)


async def get_current_user_from_token(
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(security)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> Optional[User]:
    """Extract user from JWT token."""
    if not credentials:
        return None

    payload = decode_token(credentials.credentials)
    if not payload:
        return None

    user_id = payload.get("sub")
    if not user_id:
        return None

    repo = UserRepository(session)
    return await repo.get_by_id(int(user_id))


async def get_current_user_from_api_key(
    x_api_key: Annotated[Optional[str], Header(alias="X-API-Key")] = None,
    session: Annotated[AsyncSession, Depends(get_db_session)] = None,
) -> Optional[tuple[User, ApiKey]]:
    """Extract user from API Key."""
    if not x_api_key:
        return None

    repo = ApiKeyRepository(session)
    api_key = await repo.get_by_key(x_api_key)

    if not api_key or not api_key.is_active:
        return None

    # Check expiration
    if api_key.expires_at and api_key.expires_at < datetime.utcnow():
        return None

    return api_key.user, api_key


async def get_current_user(
    token_user: Annotated[Optional[User], Depends(get_current_user_from_token)],
    api_key_result: Annotated[Optional[tuple[User, ApiKey]], Depends(get_current_user_from_api_key)],
) -> User:
    """
    Get current user from either JWT or API Key.
    Priority: JWT > API Key
    """
    if token_user:
        return token_user

    if api_key_result:
        user, _ = api_key_result
        return user

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_api_key(
    api_key_result: Annotated[Optional[tuple[User, ApiKey]], Depends(get_current_user_from_api_key)],
) -> ApiKey:
    """Get current API key (required for API key-only endpoints)."""
    if not api_key_result:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key required",
        )
    _, api_key = api_key_result
    return api_key


async def get_optional_user(
    token_user: Annotated[Optional[User], Depends(get_current_user_from_token)],
    api_key_result: Annotated[Optional[tuple[User, ApiKey]], Depends(get_current_user_from_api_key)],
) -> Optional[User]:
    """Get current user if authenticated, None otherwise."""
    if token_user:
        return token_user
    if api_key_result:
        return api_key_result[0]
    return None
```

### Usage in Endpoints

```python
# app/api/v1/endpoints/chat.py
"""
Chat endpoints with dual authentication.
"""
from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.deps import get_current_user, get_current_api_key
from app.models.user import User
from app.models.api_key import ApiKey
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/completions", response_model=ChatResponse)
async def create_completion(
    request: ChatRequest,
    current_user: Annotated[User, Depends(get_current_user)],
):
    """
    Create chat completion.
    Accepts both JWT token and API Key.
    """
    # current_user is available from either auth method
    pass


@router.post("/completions/stream")
async def create_completion_stream(
    request: ChatRequest,
    api_key: Annotated[ApiKey, Depends(get_current_api_key)],
):
    """
    Create streaming chat completion.
    API Key only (for usage tracking).
    """
    # api_key contains the specific key used
    pass
```

## Rate Limiting Middleware

### Redis-Based Rate Limiter

```python
# app/middleware/rate_limit.py
"""
Rate limiting middleware with Redis.
"""
from datetime import datetime
from typing import Optional

from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.cache import get_redis


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limit middleware using sliding window."""

    def __init__(
        self,
        app,
        default_limit: int = 100,
        window_seconds: int = 60,
    ):
        super().__init__(app)
        self.default_limit = default_limit
        self.window_seconds = window_seconds

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/health/ready"]:
            return await call_next(request)

        # Get identifier (API key or IP)
        identifier = await self._get_identifier(request)
        limit = await self._get_limit(request)

        # Check rate limit
        is_allowed, remaining, reset_at = await self._check_rate_limit(
            identifier, limit
        )

        if not is_allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Rate limit exceeded",
                    "limit": limit,
                    "reset_at": reset_at.isoformat(),
                },
                headers={
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(reset_at.timestamp())),
                    "Retry-After": str(self.window_seconds),
                },
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(reset_at.timestamp()))

        return response

    async def _get_identifier(self, request: Request) -> str:
        """Get rate limit identifier from API key or IP."""
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api_key:{api_key[:16]}"  # Use prefix for privacy
        return f"ip:{request.client.host}"

    async def _get_limit(self, request: Request) -> int:
        """Get rate limit based on API key tier."""
        api_key = request.headers.get("X-API-Key")
        if api_key:
            # TODO: Look up API key tier and return appropriate limit
            # tier_limits = {"free": 60, "pro": 600, "enterprise": 6000}
            return 600  # Default for API key users
        return self.default_limit  # Default for unauthenticated

    async def _check_rate_limit(
        self,
        identifier: str,
        limit: int,
    ) -> tuple[bool, int, datetime]:
        """Check rate limit using sliding window."""
        redis = await get_redis()
        now = datetime.utcnow()
        window_start = now.timestamp() - self.window_seconds
        key = f"rate_limit:{identifier}"

        # Remove old entries
        await redis.zremrangebyscore(key, 0, window_start)

        # Count current requests
        current_count = await redis.zcard(key)

        if current_count >= limit:
            # Get oldest entry to calculate reset time
            oldest = await redis.zrange(key, 0, 0, withscores=True)
            if oldest:
                reset_timestamp = oldest[0][1] + self.window_seconds
                reset_at = datetime.fromtimestamp(reset_timestamp)
            else:
                reset_at = now
            return False, 0, reset_at

        # Add current request
        await redis.zadd(key, {str(now.timestamp()): now.timestamp()})
        await redis.expire(key, self.window_seconds)

        remaining = limit - current_count - 1
        reset_at = datetime.fromtimestamp(now.timestamp() + self.window_seconds)

        return True, remaining, reset_at
```

### Per-Endpoint Rate Limiting

```python
# app/api/deps.py (continued)
"""
Per-endpoint rate limiting dependency.
"""
from functools import wraps


def rate_limit(limit: int = 10, window: int = 60):
    """Decorator for per-endpoint rate limiting."""
    async def dependency(
        request: Request,
        current_user: Annotated[Optional[User], Depends(get_optional_user)],
    ):
        redis = await get_redis()

        # Build key
        if current_user:
            identifier = f"user:{current_user.id}"
        else:
            identifier = f"ip:{request.client.host}"

        key = f"rate:{request.url.path}:{identifier}"

        # Increment counter
        current = await redis.incr(key)
        if current == 1:
            await redis.expire(key, window)

        if current > limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Max {limit} requests per {window}s",
            )

    return Depends(dependency)


# Usage
@router.post("/auth/login")
async def login(
    data: LoginRequest,
    _: Annotated[None, rate_limit(limit=5, window=60)],  # 5 per minute
):
    pass
```

## Credit/Usage Middleware

### Credit Check Dependency

```python
# app/api/deps.py (continued)
"""
Credit checking for usage-based billing.
"""
from app.repositories.credit import CreditRepository
from app.core.exceptions import InsufficientCreditsError


async def check_credits(
    required_credits: int = 1,
):
    """Dependency factory for credit checking."""
    async def dependency(
        current_user: Annotated[User, Depends(get_current_user)],
        session: Annotated[AsyncSession, Depends(get_db_session)],
    ):
        repo = CreditRepository(session)
        balance = await repo.get_balance(current_user.id)

        if balance < required_credits:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail={
                    "error": "Insufficient credits",
                    "required": required_credits,
                    "balance": balance,
                },
            )

        return balance

    return Depends(dependency)


# Usage
@router.post("/chat/completions")
async def create_completion(
    request: ChatRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    balance: Annotated[int, check_credits(required_credits=10)],
):
    # User has sufficient credits
    pass
```

### Credit Model and Repository

```python
# app/models/credit.py
"""
Credit/usage tracking model.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, ForeignKey, DateTime, String, Numeric
from sqlalchemy.orm import relationship

from app.db.base import Base


class CreditBalance(Base):
    """User credit balance."""

    __tablename__ = "credit_balances"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    balance = Column(Numeric(precision=10, scale=2), default=0)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="credit_balance")


class CreditTransaction(Base):
    """Credit transaction history."""

    __tablename__ = "credit_transactions"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    amount = Column(Numeric(precision=10, scale=2), nullable=False)  # + or -
    transaction_type = Column(String(50), nullable=False)  # purchase, usage, refund
    description = Column(String(255))
    reference_id = Column(String(100))  # API call ID, order ID, etc.
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="credit_transactions")
```

```python
# app/repositories/credit.py
"""
Credit repository for balance and transactions.
"""
from decimal import Decimal
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.credit import CreditBalance, CreditTransaction


class CreditRepository:
    """Repository for credit operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_balance(self, user_id: int) -> Decimal:
        """Get user's credit balance."""
        result = await self.session.execute(
            select(CreditBalance.balance).where(CreditBalance.user_id == user_id)
        )
        balance = result.scalar_one_or_none()
        return balance or Decimal("0")

    async def deduct(
        self,
        user_id: int,
        amount: Decimal,
        transaction_type: str,
        description: str = None,
        reference_id: str = None,
    ) -> Decimal:
        """Deduct credits and record transaction."""
        # Update balance atomically
        result = await self.session.execute(
            update(CreditBalance)
            .where(CreditBalance.user_id == user_id)
            .where(CreditBalance.balance >= amount)
            .values(balance=CreditBalance.balance - amount)
            .returning(CreditBalance.balance)
        )
        new_balance = result.scalar_one_or_none()

        if new_balance is None:
            raise InsufficientCreditsError("Insufficient credits")

        # Record transaction
        transaction = CreditTransaction(
            user_id=user_id,
            amount=-amount,
            transaction_type=transaction_type,
            description=description,
            reference_id=reference_id,
        )
        self.session.add(transaction)
        await self.session.commit()

        return new_balance

    async def add(
        self,
        user_id: int,
        amount: Decimal,
        transaction_type: str,
        description: str = None,
        reference_id: str = None,
    ) -> Decimal:
        """Add credits and record transaction."""
        # Upsert balance
        balance = await self.session.execute(
            select(CreditBalance).where(CreditBalance.user_id == user_id)
        )
        credit_balance = balance.scalar_one_or_none()

        if credit_balance:
            credit_balance.balance += amount
        else:
            credit_balance = CreditBalance(user_id=user_id, balance=amount)
            self.session.add(credit_balance)

        # Record transaction
        transaction = CreditTransaction(
            user_id=user_id,
            amount=amount,
            transaction_type=transaction_type,
            description=description,
            reference_id=reference_id,
        )
        self.session.add(transaction)
        await self.session.commit()

        return credit_balance.balance
```

## Combined Middleware Setup

```python
# app/main.py
"""
Application setup with all middleware.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.middleware.rate_limit import RateLimitMiddleware
from app.middleware.security_headers import SecurityHeadersMiddleware
from app.core.config import settings

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    docs_url="/docs" if settings.DEBUG else None,
)

# Order matters: first added = last executed
# Security headers (outermost)
app.add_middleware(SecurityHeadersMiddleware)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)

# Rate limiting
app.add_middleware(
    RateLimitMiddleware,
    default_limit=100,
    window_seconds=60,
)
```

## Middleware Checklist

- [ ] Dual auth (API Key + JWT) with priority handling
- [ ] Global rate limiting with Redis sliding window
- [ ] Per-endpoint rate limiting for sensitive endpoints
- [ ] Credit balance checking before expensive operations
- [ ] Credit deduction after successful operations
- [ ] Rate limit headers in responses (X-RateLimit-*)
- [ ] Proper error responses (401, 402, 429)
- [ ] Health check endpoints excluded from rate limiting
- [ ] API key tier-based limits
