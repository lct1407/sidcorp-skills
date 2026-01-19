# Python Security

OWASP Top 10 protection and security best practices for FastAPI.

## OWASP Top 10 (2025)

### 1. Broken Access Control

```python
# BAD: No ownership check
@router.delete("/api-keys/{key_id}")
async def delete_api_key(key_id: int, service: ApiKeyService):
    await service.delete(key_id)  # Anyone can delete any key!

# GOOD: Check ownership
@router.delete("/api-keys/{key_id}")
async def delete_api_key(
    key_id: int,
    service: Annotated[ApiKeyService, Depends(get_api_key_service)],
    current_user: Annotated[UserResponse, Depends(get_current_user)],
):
    api_key = await service.get(key_id)
    if api_key.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    await service.delete(key_id, current_user.id)
```

### 2. Cryptographic Failures

```python
# BAD: Weak hashing
import hashlib
hashed = hashlib.md5(password.encode()).hexdigest()  # Never use MD5/SHA1!

# GOOD: Use Argon2id
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)
```

```python
# BAD: Hardcoded secrets
SECRET_KEY = "my-secret-key-123"

# GOOD: Environment variables
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    SECRET_KEY: str
    DATABASE_URL: str

    class Config:
        env_file = ".env"

settings = Settings()
```

### 3. Injection (SQL Injection)

```python
# BAD: String concatenation
async def get_user(email: str):
    query = f"SELECT * FROM users WHERE email = '{email}'"  # SQL Injection!
    return await db.execute(query)

# GOOD: Parameterized queries with SQLAlchemy
from sqlalchemy import select

async def get_user(email: str) -> Optional[User]:
    result = await session.execute(
        select(User).where(User.email == email)
    )
    return result.scalar_one_or_none()
```

```python
# GOOD: Use ORM methods
async def search_users(search: str) -> Sequence[User]:
    result = await session.execute(
        select(User).where(
            User.email.ilike(f"%{search}%")  # SQLAlchemy handles escaping
        )
    )
    return result.scalars().all()
```

### 4. Insecure Design

```python
# BAD: No rate limiting on sensitive endpoints
@router.post("/auth/login")
async def login(data: LoginRequest):
    return await auth_service.login(data)

# GOOD: Rate limiting with slowapi
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@router.post("/auth/login")
@limiter.limit("5/minute")  # Max 5 attempts per minute
async def login(request: Request, data: LoginRequest):
    return await auth_service.login(data)
```

### 5. Security Misconfiguration

```python
# app/main.py - Security headers and CORS
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware

app = FastAPI(
    title="AI Services API",
    docs_url="/docs" if settings.DEBUG else None,  # Disable in production
    redoc_url="/redoc" if settings.DEBUG else None,
)

# HTTPS redirect in production
if not settings.DEBUG:
    app.add_middleware(HTTPSRedirectMiddleware)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,  # Never use ["*"] in production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)
```

```python
# Security headers middleware
from starlette.middleware.base import BaseHTTPMiddleware

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        return response

app.add_middleware(SecurityHeadersMiddleware)
```

### 6. Vulnerable Components

```python
# requirements.txt - Pin versions and audit regularly
fastapi==0.109.0
sqlalchemy==2.0.25
pydantic==2.5.3
python-jose[cryptography]==3.3.0
passlib[argon2]==1.7.4

# Run security audit
# pip install pip-audit
# pip-audit
```

### 7. Authentication Failures

```python
# BAD: No account lockout
async def login(email: str, password: str):
    user = await get_user(email)
    if not verify_password(password, user.hashed_password):
        raise UnauthorizedError("Invalid credentials")
    return user

# GOOD: Account lockout after failed attempts
from datetime import datetime, timedelta

MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_DURATION = timedelta(minutes=15)

async def login(email: str, password: str):
    user = await get_user(email)

    # Check if locked out
    if user.locked_until and user.locked_until > datetime.utcnow():
        raise HTTPException(
            status_code=429,
            detail=f"Account locked. Try again after {user.locked_until}",
        )

    if not verify_password(password, user.hashed_password):
        user.failed_login_attempts += 1

        if user.failed_login_attempts >= MAX_LOGIN_ATTEMPTS:
            user.locked_until = datetime.utcnow() + LOCKOUT_DURATION

        await session.commit()
        raise UnauthorizedError("Invalid credentials")

    # Reset on successful login
    user.failed_login_attempts = 0
    user.locked_until = None
    await session.commit()

    return user
```

### 8. Data Integrity Failures

```python
# BAD: No input validation
@router.post("/users")
async def create_user(data: dict):
    return await user_service.create(data)

# GOOD: Pydantic validation
from pydantic import BaseModel, EmailStr, Field, field_validator
import re

class UserCreate(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_]+$", v):
            raise ValueError("Username must be alphanumeric")
        return v.lower()

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain uppercase letter")
        if not re.search(r"[0-9]", v):
            raise ValueError("Password must contain digit")
        if not re.search(r"[!@#$%^&*]", v):
            raise ValueError("Password must contain special character")
        return v

@router.post("/users")
async def create_user(data: UserCreate):  # Validated automatically
    return await user_service.create(data)
```

### 9. Security Logging Failures

```python
# app/core/logging.py
"""
Security event logging.
"""
import logging
from datetime import datetime
from typing import Optional

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

security_logger = logging.getLogger("security")


def log_auth_event(
    event_type: str,
    user_id: Optional[int],
    ip_address: str,
    success: bool,
    details: Optional[str] = None,
):
    """Log authentication events."""
    security_logger.info(
        f"AUTH_EVENT | type={event_type} | user_id={user_id} | "
        f"ip={ip_address} | success={success} | details={details}"
    )


def log_access_denied(
    user_id: int,
    resource: str,
    action: str,
    ip_address: str,
):
    """Log access denied events."""
    security_logger.warning(
        f"ACCESS_DENIED | user_id={user_id} | resource={resource} | "
        f"action={action} | ip={ip_address}"
    )


# Usage in auth
@router.post("/auth/login")
async def login(request: Request, data: LoginRequest):
    ip = request.client.host

    try:
        user = await auth_service.authenticate(data.email, data.password)
        log_auth_event("LOGIN", user.id, ip, success=True)
        return create_tokens(user)
    except UnauthorizedError as e:
        log_auth_event("LOGIN", None, ip, success=False, details=str(e))
        raise
```

### 10. Server-Side Request Forgery (SSRF)

```python
# BAD: Unrestricted URL fetching
import httpx

@router.post("/fetch-url")
async def fetch_url(url: str):
    response = await httpx.get(url)  # SSRF vulnerability!
    return response.text

# GOOD: URL validation and allowlist
from urllib.parse import urlparse

ALLOWED_DOMAINS = ["api.example.com", "cdn.example.com"]

def validate_url(url: str) -> bool:
    """Validate URL against allowlist."""
    try:
        parsed = urlparse(url)

        # Must be HTTPS
        if parsed.scheme != "https":
            return False

        # Check domain allowlist
        if parsed.netloc not in ALLOWED_DOMAINS:
            return False

        # Block internal IPs
        import ipaddress
        try:
            ip = ipaddress.ip_address(parsed.hostname)
            if ip.is_private or ip.is_loopback:
                return False
        except ValueError:
            pass  # Not an IP address

        return True
    except Exception:
        return False

@router.post("/fetch-url")
async def fetch_url(url: str):
    if not validate_url(url):
        raise HTTPException(status_code=400, detail="Invalid or disallowed URL")

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url)
    return response.text
```

## Input Sanitization

```python
# app/core/sanitize.py
"""
Input sanitization utilities.
"""
import html
import re
from typing import Any


def sanitize_html(text: str) -> str:
    """Escape HTML entities."""
    return html.escape(text)


def sanitize_filename(filename: str) -> str:
    """Remove dangerous characters from filename."""
    # Remove path separators and null bytes
    filename = filename.replace("/", "").replace("\\", "").replace("\x00", "")
    # Remove leading dots
    filename = filename.lstrip(".")
    # Only allow safe characters
    filename = re.sub(r"[^a-zA-Z0-9_.-]", "_", filename)
    return filename[:255]  # Limit length


def sanitize_search_query(query: str) -> str:
    """Sanitize search query."""
    # Remove SQL wildcards that could cause issues
    query = query.replace("%", "").replace("_", " ")
    # Limit length
    return query[:100].strip()
```

## API Key Security

```python
# Secure API key handling
import secrets
import hashlib

def generate_api_key() -> tuple[str, str]:
    """Generate API key and its hash."""
    # Generate secure random key
    raw_key = secrets.token_urlsafe(32)
    api_key = f"sk_{raw_key}"

    # Hash for storage (never store plain key)
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()

    return api_key, key_hash


def verify_api_key(provided_key: str, stored_hash: str) -> bool:
    """Verify API key against stored hash."""
    provided_hash = hashlib.sha256(provided_key.encode()).hexdigest()
    # Use constant-time comparison to prevent timing attacks
    return secrets.compare_digest(provided_hash, stored_hash)
```

## Security Checklist

- [ ] Argon2id for password hashing
- [ ] Parameterized queries (SQLAlchemy ORM)
- [ ] Input validation (Pydantic)
- [ ] Rate limiting on auth endpoints
- [ ] HTTPS only in production
- [ ] Security headers configured
- [ ] CORS properly restricted
- [ ] Secrets in environment variables
- [ ] Account lockout after failed attempts
- [ ] Security event logging
- [ ] Regular dependency audits
- [ ] API key hashing (never store plain)
- [ ] Ownership checks on resources
