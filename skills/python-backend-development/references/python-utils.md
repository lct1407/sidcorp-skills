# Python Utils

Utility functions và helper modules cho FastAPI applications.

## Cấu trúc thư mục

```
app/
├── core/           # Framework-level (config, security, exceptions)
│   ├── config.py
│   ├── security.py
│   ├── exceptions.py
│   └── ...
│
└── utils/          # Application-level helpers (reusable functions)
    ├── __init__.py
    ├── datetime.py
    ├── string.py
    ├── validation.py
    ├── pagination.py
    ├── file.py
    └── ...
```

## Core vs Utils

| `app/core/` | `app/utils/` |
|-------------|--------------|
| Framework configuration | Pure utility functions |
| Security (JWT, hashing) | String manipulation |
| Exception classes | Date/time helpers |
| Database session | Validation helpers |
| Middleware | Pagination |
| Logging setup | File operations |

**Rule:** `utils` chứa các hàm thuần túy, không phụ thuộc vào framework hay database.

## DateTime Utils

```python
# app/utils/datetime.py
"""
DateTime utility functions.
"""
from datetime import datetime, timedelta, timezone
from typing import Optional


def utc_now() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)


def to_utc(dt: datetime) -> datetime:
    """Convert datetime to UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def from_timestamp(ts: float) -> datetime:
    """Create UTC datetime from Unix timestamp."""
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def to_timestamp(dt: datetime) -> float:
    """Convert datetime to Unix timestamp."""
    return dt.timestamp()


def format_iso(dt: datetime) -> str:
    """Format datetime as ISO 8601 string."""
    return dt.isoformat()


def parse_iso(iso_string: str) -> datetime:
    """Parse ISO 8601 string to datetime."""
    return datetime.fromisoformat(iso_string.replace("Z", "+00:00"))


def add_days(dt: datetime, days: int) -> datetime:
    """Add days to datetime."""
    return dt + timedelta(days=days)


def add_hours(dt: datetime, hours: int) -> datetime:
    """Add hours to datetime."""
    return dt + timedelta(hours=hours)


def add_minutes(dt: datetime, minutes: int) -> datetime:
    """Add minutes to datetime."""
    return dt + timedelta(minutes=minutes)


def start_of_day(dt: datetime) -> datetime:
    """Get start of day (00:00:00)."""
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def end_of_day(dt: datetime) -> datetime:
    """Get end of day (23:59:59)."""
    return dt.replace(hour=23, minute=59, second=59, microsecond=999999)


def days_between(start: datetime, end: datetime) -> int:
    """Calculate days between two dates."""
    return (end.date() - start.date()).days


def is_expired(expiry: datetime, now: Optional[datetime] = None) -> bool:
    """Check if datetime has expired."""
    now = now or utc_now()
    return expiry < now


def time_ago(dt: datetime, now: Optional[datetime] = None) -> str:
    """Human-readable time ago string."""
    now = now or utc_now()
    diff = now - dt

    seconds = diff.total_seconds()
    if seconds < 60:
        return "just now"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    else:
        days = int(seconds / 86400)
        return f"{days} day{'s' if days > 1 else ''} ago"
```

## String Utils

```python
# app/utils/string.py
"""
String utility functions.
"""
import re
import unicodedata
from typing import Optional
import secrets
import string


def slugify(text: str) -> str:
    """Convert text to URL-safe slug."""
    # Normalize unicode
    text = unicodedata.normalize("NFKD", text)
    # Remove non-ASCII
    text = text.encode("ascii", "ignore").decode("ascii")
    # Lowercase
    text = text.lower()
    # Replace spaces and special chars with hyphens
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "-", text)
    return text.strip("-")


def truncate(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to max length."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def mask_email(email: str) -> str:
    """Mask email for display (j***@example.com)."""
    if "@" not in email:
        return email
    local, domain = email.split("@", 1)
    if len(local) <= 2:
        masked = local[0] + "***"
    else:
        masked = local[0] + "***" + local[-1]
    return f"{masked}@{domain}"


def mask_api_key(key: str, visible_chars: int = 8) -> str:
    """Mask API key (sk_live_abc...xyz)."""
    if len(key) <= visible_chars * 2:
        return key
    return f"{key[:visible_chars]}...{key[-visible_chars:]}"


def generate_random_string(length: int = 32) -> str:
    """Generate cryptographically secure random string."""
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def generate_token(prefix: str = "", length: int = 32) -> str:
    """Generate token with optional prefix (sk_live_xxx)."""
    token = secrets.token_urlsafe(length)
    if prefix:
        return f"{prefix}_{token}"
    return token


def camel_to_snake(name: str) -> str:
    """Convert CamelCase to snake_case."""
    pattern = re.compile(r"(?<!^)(?=[A-Z])")
    return pattern.sub("_", name).lower()


def snake_to_camel(name: str) -> str:
    """Convert snake_case to camelCase."""
    components = name.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


def pluralize(word: str, count: int) -> str:
    """Simple pluralization."""
    if count == 1:
        return word
    # Simple rules
    if word.endswith("y"):
        return word[:-1] + "ies"
    elif word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    return word + "s"


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace (collapse multiple spaces, trim)."""
    return " ".join(text.split())


def is_valid_uuid(value: str) -> bool:
    """Check if string is valid UUID."""
    pattern = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        re.IGNORECASE,
    )
    return bool(pattern.match(value))
```

## Validation Utils

```python
# app/utils/validation.py
"""
Validation utility functions.
"""
import re
from typing import Optional


def is_valid_email(email: str) -> bool:
    """Validate email format."""
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


def is_valid_phone(phone: str) -> bool:
    """Validate phone number (E.164 format)."""
    pattern = r"^\+[1-9]\d{1,14}$"
    return bool(re.match(pattern, phone))


def is_valid_url(url: str) -> bool:
    """Validate URL format."""
    pattern = r"^https?://[^\s/$.?#].[^\s]*$"
    return bool(re.match(pattern, url, re.IGNORECASE))


def is_strong_password(password: str) -> tuple[bool, Optional[str]]:
    """
    Check password strength.
    Returns (is_valid, error_message).
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain uppercase letter"
    if not re.search(r"[a-z]", password):
        return False, "Password must contain lowercase letter"
    if not re.search(r"\d", password):
        return False, "Password must contain digit"
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False, "Password must contain special character"
    return True, None


def is_valid_username(username: str) -> tuple[bool, Optional[str]]:
    """
    Validate username.
    Returns (is_valid, error_message).
    """
    if len(username) < 3:
        return False, "Username must be at least 3 characters"
    if len(username) > 30:
        return False, "Username must be at most 30 characters"
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", username):
        return False, "Username must start with letter, contain only letters, numbers, underscores"
    return True, None


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage."""
    # Remove path separators
    filename = filename.replace("/", "").replace("\\", "")
    # Remove null bytes
    filename = filename.replace("\x00", "")
    # Remove leading dots
    filename = filename.lstrip(".")
    # Only allow safe characters
    filename = re.sub(r"[^\w\s.-]", "_", filename)
    # Limit length
    return filename[:255]


def clamp(value: int | float, min_val: int | float, max_val: int | float) -> int | float:
    """Clamp value between min and max."""
    return max(min_val, min(value, max_val))
```

## Pagination Utils

```python
# app/utils/pagination.py
"""
Pagination utility functions.
"""
from typing import TypeVar, Generic, Sequence
from pydantic import BaseModel

T = TypeVar("T")


class PaginationParams(BaseModel):
    """Pagination parameters."""
    page: int = 1
    size: int = 20

    @property
    def skip(self) -> int:
        """Calculate offset."""
        return (self.page - 1) * self.size

    @property
    def limit(self) -> int:
        """Get limit."""
        return self.size


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated response wrapper."""
    items: Sequence[T]
    total: int
    page: int
    size: int
    pages: int

    @classmethod
    def create(
        cls,
        items: Sequence[T],
        total: int,
        page: int,
        size: int,
    ) -> "PaginatedResponse[T]":
        """Create paginated response."""
        pages = (total + size - 1) // size if size > 0 else 0
        return cls(
            items=items,
            total=total,
            page=page,
            size=size,
            pages=pages,
        )


def paginate(
    items: Sequence[T],
    page: int = 1,
    size: int = 20,
) -> PaginatedResponse[T]:
    """Paginate a sequence in memory."""
    total = len(items)
    start = (page - 1) * size
    end = start + size
    return PaginatedResponse.create(
        items=items[start:end],
        total=total,
        page=page,
        size=size,
    )


# FastAPI dependency
from fastapi import Query


def get_pagination(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(20, ge=1, le=100, description="Items per page"),
) -> PaginationParams:
    """Pagination dependency for FastAPI."""
    return PaginationParams(page=page, size=size)
```

## File Utils

```python
# app/utils/file.py
"""
File utility functions.
"""
import os
import hashlib
import mimetypes
from pathlib import Path
from typing import BinaryIO


def get_file_hash(file: BinaryIO, algorithm: str = "sha256") -> str:
    """Calculate file hash."""
    hasher = hashlib.new(algorithm)
    for chunk in iter(lambda: file.read(8192), b""):
        hasher.update(chunk)
    file.seek(0)  # Reset position
    return hasher.hexdigest()


def get_file_size(file: BinaryIO) -> int:
    """Get file size in bytes."""
    current = file.tell()
    file.seek(0, 2)  # End
    size = file.tell()
    file.seek(current)  # Reset
    return size


def get_mime_type(filename: str) -> str:
    """Get MIME type from filename."""
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type or "application/octet-stream"


def get_file_extension(filename: str) -> str:
    """Get file extension (lowercase, without dot)."""
    return Path(filename).suffix.lower().lstrip(".")


def is_allowed_extension(filename: str, allowed: set[str]) -> bool:
    """Check if file extension is allowed."""
    ext = get_file_extension(filename)
    return ext in allowed


def human_readable_size(size_bytes: int) -> str:
    """Convert bytes to human readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def ensure_directory(path: str | Path) -> Path:
    """Ensure directory exists, create if not."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_join(base: str | Path, *paths: str) -> Path:
    """Safely join paths, preventing directory traversal."""
    base = Path(base).resolve()
    result = base.joinpath(*paths).resolve()
    if not str(result).startswith(str(base)):
        raise ValueError("Path traversal detected")
    return result
```

## JSON Utils

```python
# app/utils/json.py
"""
JSON utility functions.
"""
import json
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for common Python types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, date):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, bytes):
            return obj.decode("utf-8")
        if hasattr(obj, "model_dump"):  # Pydantic v2
            return obj.model_dump()
        if hasattr(obj, "dict"):  # Pydantic v1
            return obj.dict()
        return super().default(obj)


def dumps(obj: Any, **kwargs) -> str:
    """Serialize object to JSON string."""
    return json.dumps(obj, cls=CustomJSONEncoder, **kwargs)


def loads(s: str, **kwargs) -> Any:
    """Deserialize JSON string to object."""
    return json.loads(s, **kwargs)


def pretty_print(obj: Any) -> str:
    """Pretty print JSON."""
    return dumps(obj, indent=2, ensure_ascii=False)
```

## Retry Utils

```python
# app/utils/retry.py
"""
Retry utility functions.
"""
import asyncio
from functools import wraps
from typing import Callable, TypeVar, Any
import random

T = TypeVar("T")


class RetryError(Exception):
    """Raised when all retries are exhausted."""
    pass


async def retry_async(
    func: Callable[..., T],
    *args,
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    jitter: bool = True,
    exceptions: tuple = (Exception,),
    **kwargs,
) -> T:
    """
    Retry async function with exponential backoff.
    """
    last_exception = None

    for attempt in range(max_attempts):
        try:
            return await func(*args, **kwargs)
        except exceptions as e:
            last_exception = e

            if attempt == max_attempts - 1:
                break

            wait_time = delay * (backoff ** attempt)
            if jitter:
                wait_time *= (0.5 + random.random())

            await asyncio.sleep(wait_time)

    raise RetryError(f"Failed after {max_attempts} attempts") from last_exception


def with_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """Decorator for retry with exponential backoff."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await retry_async(
                func,
                *args,
                max_attempts=max_attempts,
                delay=delay,
                backoff=backoff,
                exceptions=exceptions,
                **kwargs,
            )
        return wrapper
    return decorator


# Usage
@with_retry(max_attempts=3, delay=1.0, exceptions=(ConnectionError,))
async def fetch_data(url: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()
```

## Usage in Project

```python
# Importing utils
from app.utils.datetime import utc_now, is_expired
from app.utils.string import slugify, mask_email, generate_token
from app.utils.validation import is_valid_email, is_strong_password
from app.utils.pagination import PaginationParams, get_pagination
from app.utils.file import get_file_hash, human_readable_size
from app.utils.json import dumps, loads
from app.utils.retry import retry_async, with_retry


# Example in service
class UserService:
    async def create(self, data: UserCreate) -> User:
        # Validate
        if not is_valid_email(data.email):
            raise ValidationError("Invalid email")

        is_valid, error = is_strong_password(data.password)
        if not is_valid:
            raise ValidationError(error)

        # Generate
        user = User(
            email=data.email,
            username=slugify(data.username),
            api_key=generate_token(prefix="sk"),
            created_at=utc_now(),
        )

        return await self.repository.create(user)
```

## S3 Storage Utils

```python
# app/utils/storage.py
"""
S3/Cloud storage utility functions.
"""
import uuid
from typing import BinaryIO, Optional
from datetime import datetime

import boto3
from botocore.exceptions import ClientError

from app.core.config import settings


class S3Client:
    """S3 storage client wrapper."""

    def __init__(self):
        self.client = boto3.client(
            "s3",
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION,
        )
        self.bucket = settings.S3_BUCKET

    def upload_file(
        self,
        file: BinaryIO,
        filename: str,
        folder: str = "uploads",
        content_type: Optional[str] = None,
        public: bool = False,
    ) -> str:
        """
        Upload file to S3.
        Returns the S3 key (path).
        """
        # Generate unique key
        ext = filename.rsplit(".", 1)[-1] if "." in filename else ""
        unique_name = f"{uuid.uuid4().hex}.{ext}" if ext else uuid.uuid4().hex
        key = f"{folder}/{datetime.utcnow():%Y/%m/%d}/{unique_name}"

        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type
        if public:
            extra_args["ACL"] = "public-read"

        self.client.upload_fileobj(
            file,
            self.bucket,
            key,
            ExtraArgs=extra_args,
        )

        return key

    def get_url(self, key: str, expires_in: int = 3600) -> str:
        """Generate presigned URL for file access."""
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires_in,
        )

    def get_public_url(self, key: str) -> str:
        """Get public URL (for public files only)."""
        return f"https://{self.bucket}.s3.{settings.AWS_REGION}.amazonaws.com/{key}"

    def delete_file(self, key: str) -> bool:
        """Delete file from S3."""
        try:
            self.client.delete_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False

    def file_exists(self, key: str) -> bool:
        """Check if file exists in S3."""
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False

    def copy_file(self, source_key: str, dest_key: str) -> bool:
        """Copy file within S3."""
        try:
            self.client.copy_object(
                Bucket=self.bucket,
                CopySource={"Bucket": self.bucket, "Key": source_key},
                Key=dest_key,
            )
            return True
        except ClientError:
            return False


# Singleton instance
_s3_client: Optional[S3Client] = None


def get_s3_client() -> S3Client:
    """Get S3 client singleton."""
    global _s3_client
    if _s3_client is None:
        _s3_client = S3Client()
    return _s3_client


# Convenience functions
async def upload_to_s3(
    file: BinaryIO,
    filename: str,
    folder: str = "uploads",
    content_type: Optional[str] = None,
) -> str:
    """Upload file and return URL."""
    client = get_s3_client()
    key = client.upload_file(file, filename, folder, content_type)
    return client.get_url(key)


async def delete_from_s3(key: str) -> bool:
    """Delete file from S3."""
    return get_s3_client().delete_file(key)
```

## Email Utils

```python
# app/utils/email.py
"""
Email utility functions.
"""
from typing import Optional, List
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiosmtplib

from app.core.config import settings


async def send_email(
    to: str | list[str],
    subject: str,
    body: str,
    html: Optional[str] = None,
    from_email: Optional[str] = None,
    reply_to: Optional[str] = None,
) -> bool:
    """
    Send email via SMTP.
    Returns True if sent successfully.
    """
    if isinstance(to, str):
        to = [to]

    from_email = from_email or settings.EMAIL_FROM

    # Create message
    if html:
        message = MIMEMultipart("alternative")
        message.attach(MIMEText(body, "plain"))
        message.attach(MIMEText(html, "html"))
    else:
        message = MIMEText(body, "plain")

    message["Subject"] = subject
    message["From"] = from_email
    message["To"] = ", ".join(to)

    if reply_to:
        message["Reply-To"] = reply_to

    try:
        await aiosmtplib.send(
            message,
            hostname=settings.SMTP_HOST,
            port=settings.SMTP_PORT,
            username=settings.SMTP_USER,
            password=settings.SMTP_PASSWORD,
            use_tls=settings.SMTP_TLS,
        )
        return True
    except Exception:
        return False


async def send_template_email(
    to: str,
    template_name: str,
    context: dict,
    subject: Optional[str] = None,
) -> bool:
    """
    Send email using template.
    Templates stored in app/templates/email/
    """
    from jinja2 import Environment, FileSystemLoader

    env = Environment(loader=FileSystemLoader("app/templates/email"))

    # Load template
    template = env.get_template(f"{template_name}.html")
    html = template.render(**context)

    # Plain text version
    text_template = env.get_template(f"{template_name}.txt")
    text = text_template.render(**context)

    # Subject from template or parameter
    subject = subject or context.get("subject", template_name)

    return await send_email(to, subject, text, html=html)


# Pre-built email functions
async def send_welcome_email(email: str, name: str) -> bool:
    """Send welcome email to new user."""
    return await send_template_email(
        to=email,
        template_name="welcome",
        context={"name": name, "subject": "Welcome to AI Services!"},
    )


async def send_password_reset_email(email: str, reset_link: str) -> bool:
    """Send password reset email."""
    return await send_template_email(
        to=email,
        template_name="password_reset",
        context={
            "reset_link": reset_link,
            "subject": "Reset Your Password",
        },
    )


async def send_verification_email(email: str, code: str) -> bool:
    """Send email verification code."""
    return await send_template_email(
        to=email,
        template_name="verification",
        context={
            "code": code,
            "subject": "Verify Your Email",
        },
    )
```

## Text Formatting Utils

```python
# app/utils/text.py
"""
Text formatting utility functions.
"""
import re
import html
from typing import Optional


def strip_html(text: str) -> str:
    """Remove HTML tags from text."""
    clean = re.compile("<.*?>")
    return re.sub(clean, "", text)


def escape_html(text: str) -> str:
    """Escape HTML entities."""
    return html.escape(text)


def unescape_html(text: str) -> str:
    """Unescape HTML entities."""
    return html.unescape(text)


def nl2br(text: str) -> str:
    """Convert newlines to <br> tags."""
    return text.replace("\n", "<br>")


def br2nl(text: str) -> str:
    """Convert <br> tags to newlines."""
    return re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)


def excerpt(text: str, length: int = 150, suffix: str = "...") -> str:
    """Create excerpt from text, breaking at word boundary."""
    if len(text) <= length:
        return text
    # Find last space before length
    truncated = text[:length].rsplit(" ", 1)[0]
    return truncated + suffix


def word_count(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def reading_time(text: str, wpm: int = 200) -> int:
    """Estimate reading time in minutes."""
    words = word_count(text)
    return max(1, round(words / wpm))


def highlight_search(text: str, query: str, tag: str = "mark") -> str:
    """Highlight search terms in text."""
    if not query:
        return text
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    return pattern.sub(f"<{tag}>\\g<0></{tag}>", text)


def format_number(number: int | float, decimals: int = 0) -> str:
    """Format number with thousand separators."""
    if decimals > 0:
        return f"{number:,.{decimals}f}"
    return f"{int(number):,}"


def format_currency(
    amount: float,
    currency: str = "USD",
    locale: str = "en_US",
) -> str:
    """Format currency amount."""
    symbols = {"USD": "$", "EUR": "€", "GBP": "£", "VND": "₫"}
    symbol = symbols.get(currency, currency)

    if currency == "VND":
        return f"{int(amount):,} {symbol}"
    return f"{symbol}{amount:,.2f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format as percentage."""
    return f"{value * 100:.{decimals}f}%"


def ordinal(n: int) -> str:
    """Convert number to ordinal (1st, 2nd, 3rd, etc.)."""
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = ["th", "st", "nd", "rd", "th"][min(n % 10, 4)]
    return f"{n}{suffix}"


def initials(name: str) -> str:
    """Get initials from name."""
    words = name.split()
    return "".join(word[0].upper() for word in words if word)


def title_case(text: str) -> str:
    """Convert to title case (respecting small words)."""
    small_words = {"a", "an", "and", "as", "at", "but", "by", "for", "in", "of", "on", "or", "the", "to", "with"}
    words = text.lower().split()
    result = []
    for i, word in enumerate(words):
        if i == 0 or word not in small_words:
            result.append(word.capitalize())
        else:
            result.append(word)
    return " ".join(result)


def remove_extra_whitespace(text: str) -> str:
    """Remove extra whitespace, keeping single spaces."""
    return " ".join(text.split())


def extract_urls(text: str) -> list[str]:
    """Extract all URLs from text."""
    pattern = r"https?://[^\s<>\"{}|\\^`\[\]]+"
    return re.findall(pattern, text)


def extract_emails(text: str) -> list[str]:
    """Extract all email addresses from text."""
    pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    return re.findall(pattern, text)


def mask_sensitive(text: str, visible: int = 4) -> str:
    """Mask sensitive text, showing only last N characters."""
    if len(text) <= visible:
        return "*" * len(text)
    return "*" * (len(text) - visible) + text[-visible:]
```

## Number Utils

```python
# app/utils/number.py
"""
Number utility functions.
"""
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional


def round_decimal(value: Decimal, places: int = 2) -> Decimal:
    """Round decimal to specified places."""
    quantize = Decimal(10) ** -places
    return value.quantize(quantize, rounding=ROUND_HALF_UP)


def to_decimal(value: int | float | str, places: int = 2) -> Decimal:
    """Convert to decimal with specified precision."""
    return round_decimal(Decimal(str(value)), places)


def percentage_of(part: float, whole: float) -> float:
    """Calculate what percentage part is of whole."""
    if whole == 0:
        return 0.0
    return (part / whole) * 100


def percentage_change(old: float, new: float) -> float:
    """Calculate percentage change between two values."""
    if old == 0:
        return 0.0 if new == 0 else float("inf")
    return ((new - old) / abs(old)) * 100


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(value, max_val))


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b."""
    return a + (b - a) * t


def is_between(value: float, min_val: float, max_val: float) -> bool:
    """Check if value is between min and max (inclusive)."""
    return min_val <= value <= max_val


def safe_divide(
    numerator: float,
    denominator: float,
    default: float = 0.0,
) -> float:
    """Safely divide, returning default if denominator is 0."""
    if denominator == 0:
        return default
    return numerator / denominator
```

## Crypto Utils

```python
# app/utils/crypto.py
"""
Cryptography utility functions.
"""
import hashlib
import hmac
import secrets
import base64
from typing import Optional

from cryptography.fernet import Fernet


def generate_secret_key() -> str:
    """Generate a secure secret key."""
    return secrets.token_urlsafe(32)


def generate_api_key(prefix: str = "sk") -> str:
    """Generate API key with prefix."""
    key = secrets.token_urlsafe(32)
    return f"{prefix}_{key}"


def hash_sha256(data: str | bytes) -> str:
    """SHA256 hash."""
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()


def hash_sha512(data: str | bytes) -> str:
    """SHA512 hash."""
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha512(data).hexdigest()


def hmac_sign(message: str, secret: str) -> str:
    """Create HMAC signature."""
    return hmac.new(
        secret.encode(),
        message.encode(),
        hashlib.sha256,
    ).hexdigest()


def hmac_verify(message: str, signature: str, secret: str) -> bool:
    """Verify HMAC signature (timing-safe)."""
    expected = hmac_sign(message, secret)
    return hmac.compare_digest(expected, signature)


def encrypt(data: str, key: str) -> str:
    """Encrypt data using Fernet symmetric encryption."""
    # Key must be 32 url-safe base64-encoded bytes
    key_bytes = base64.urlsafe_b64encode(key.encode()[:32].ljust(32, b"\0"))
    f = Fernet(key_bytes)
    encrypted = f.encrypt(data.encode())
    return base64.urlsafe_b64encode(encrypted).decode()


def decrypt(encrypted_data: str, key: str) -> str:
    """Decrypt Fernet encrypted data."""
    key_bytes = base64.urlsafe_b64encode(key.encode()[:32].ljust(32, b"\0"))
    f = Fernet(key_bytes)
    decrypted = f.decrypt(base64.urlsafe_b64decode(encrypted_data))
    return decrypted.decode()


def generate_otp(length: int = 6) -> str:
    """Generate numeric OTP."""
    return "".join(secrets.choice("0123456789") for _ in range(length))


def constant_time_compare(a: str, b: str) -> bool:
    """Constant-time string comparison (timing-safe)."""
    return hmac.compare_digest(a, b)
```

## Usage in Project

**QUAN TRỌNG:** Chỉ import những gì cần dùng, không import tất cả.

```python
# ✅ ĐÚNG: Import chỉ những gì cần cho file này
from app.utils.datetime import utc_now
from app.utils.string import slugify

# ❌ SAI: Không import hết
from app.utils import *
from app.utils.datetime import *
```

**Lazy initialization:** Các client (S3, Redis, etc.) chỉ được tạo khi gọi lần đầu:

```python
# Lần đầu gọi -> tạo S3Client
url = await upload_to_s3(file, filename)

# Lần sau -> dùng lại client đã tạo
url2 = await upload_to_s3(file2, filename2)
```

### Example: Upload avatar
@router.post("/users/{user_id}/avatar")
async def upload_avatar(
    user_id: int,
    file: UploadFile,
    service: UserService = Depends(),
):
    # Validate
    if not is_allowed_extension(file.filename, {"jpg", "jpeg", "png"}):
        raise HTTPException(400, "Invalid file type")

    # Upload to S3
    url = await upload_to_s3(
        file.file,
        file.filename,
        folder=f"avatars/{user_id}",
        content_type=file.content_type,
    )

    # Update user
    await service.update_avatar(user_id, url)

    return {"avatar_url": url}


# Example: Format response
def format_order_summary(order):
    return {
        "id": order.id,
        "total": format_currency(order.total, "USD"),
        "created": time_ago(order.created_at),
        "items_count": format_number(order.items_count),
    }
```

## Utils Checklist

- [ ] Pure functions (no side effects, no dependencies)
- [ ] Type hints on all functions
- [ ] Docstrings with examples
- [ ] Unit tests for each utility
- [ ] No framework dependencies (FastAPI, SQLAlchemy)
- [ ] Reusable across projects
- [ ] S3 client with singleton pattern
- [ ] Email templates in separate folder
- [ ] Crypto functions use secure libraries
