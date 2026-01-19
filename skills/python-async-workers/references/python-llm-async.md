# Python LLM Async Patterns

Handling long-running LLM API calls that exceed HTTP timeout limits.

## The Problem

```
Client → API Gateway (30s timeout) → FastAPI → LLM API (60-300s)
                    ↑
              TIMEOUT ERROR
```

LLM APIs (OpenAI, Anthropic, etc.) can take 30-300+ seconds for:
- Long context completions
- Complex reasoning tasks
- Large batch processing
- Image/audio generation

## Solution Patterns

### Pattern 1: Webhook/Callback (Recommended)

```
1. Client POST /completions → Returns task_id immediately
2. Worker processes LLM request asynchronously
3. Worker calls webhook URL when complete
4. Client receives result via webhook
```

### Pattern 2: Polling

```
1. Client POST /completions → Returns task_id immediately
2. Client polls GET /tasks/{task_id} periodically
3. Eventually receives completed result
```

### Pattern 3: Server-Sent Events (SSE) / Streaming

```
1. Client connects to streaming endpoint
2. Server streams tokens as they arrive
3. Connection stays open until complete
```

## Implementation

### Request Schema

```python
# app/schemas/completion.py
"""
Completion request/response schemas.
"""
from typing import Optional
from pydantic import BaseModel, HttpUrl


class CompletionRequest(BaseModel):
    """Async completion request."""
    prompt: str
    model: str = "gpt-4"
    max_tokens: int = 1000
    temperature: float = 0.7

    # Async options
    webhook_url: Optional[HttpUrl] = None  # For callback pattern
    metadata: Optional[dict] = None  # Pass-through data


class CompletionTask(BaseModel):
    """Task tracking response."""
    task_id: str
    status: str  # pending, processing, completed, failed
    created_at: datetime
    estimated_seconds: Optional[int] = None


class CompletionResult(BaseModel):
    """Completed result."""
    task_id: str
    status: str
    content: Optional[str] = None
    usage: Optional[dict] = None
    error: Optional[str] = None
    metadata: Optional[dict] = None
    completed_at: Optional[datetime] = None
```

### Task Model

```python
# app/models/completion_task.py
"""
Completion task model for tracking.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship

from app.db.base import Base


class CompletionTask(Base):
    """Track async completion tasks."""

    __tablename__ = "completion_tasks"

    id = Column(Integer, primary_key=True)
    task_id = Column(String(36), unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    api_key_id = Column(Integer, ForeignKey("api_keys.id"), nullable=True)

    # Request
    prompt = Column(Text, nullable=False)
    model = Column(String(50), nullable=False)
    parameters = Column(JSON, default={})
    webhook_url = Column(String(500), nullable=True)
    metadata = Column(JSON, default={})

    # Status
    status = Column(String(20), default="pending", index=True)
    # pending, processing, completed, failed, cancelled

    # Result
    content = Column(Text, nullable=True)
    usage = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", back_populates="completion_tasks")
```

### API Endpoints

```python
# app/api/v1/endpoints/completions.py
"""
Async completion endpoints.
"""
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse

from app.api.deps import get_current_user, get_current_api_key
from app.models.user import User
from app.models.api_key import ApiKey
from app.schemas.completion import (
    CompletionRequest,
    CompletionTask,
    CompletionResult,
)
from app.services.completion import CompletionService
from app.tasks.llm import process_completion

router = APIRouter(prefix="/completions", tags=["Completions"])


@router.post("", response_model=CompletionTask, status_code=202)
async def create_completion(
    request: CompletionRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    api_key: Annotated[ApiKey, Depends(get_current_api_key)],
    service: Annotated[CompletionService, Depends()],
):
    """
    Create async completion request.

    Returns task_id immediately. Result delivered via:
    - Webhook (if webhook_url provided)
    - Polling GET /completions/{task_id}
    """
    # Create task record
    task = await service.create_task(
        user_id=current_user.id,
        api_key_id=api_key.id,
        request=request,
    )

    # Dispatch to Celery worker
    process_completion.delay(task.task_id)

    return CompletionTask(
        task_id=task.task_id,
        status="pending",
        created_at=task.created_at,
        estimated_seconds=estimate_completion_time(request),
    )


@router.get("/{task_id}", response_model=CompletionResult)
async def get_completion(
    task_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    service: Annotated[CompletionService, Depends()],
):
    """Get completion task status and result."""
    task = await service.get_task(task_id, current_user.id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return CompletionResult(
        task_id=task.task_id,
        status=task.status,
        content=task.content,
        usage=task.usage,
        error=task.error,
        metadata=task.metadata,
        completed_at=task.completed_at,
    )


@router.delete("/{task_id}")
async def cancel_completion(
    task_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    service: Annotated[CompletionService, Depends()],
):
    """Cancel a pending/processing completion task."""
    success = await service.cancel_task(task_id, current_user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Task not found or not cancellable")

    return {"message": "Task cancelled"}


@router.post("/stream", response_class=StreamingResponse)
async def create_streaming_completion(
    request: CompletionRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    service: Annotated[CompletionService, Depends()],
):
    """
    Create streaming completion (SSE).

    For clients that can handle long connections.
    """
    async def generate():
        async for chunk in service.stream_completion(request, current_user.id):
            yield f"data: {chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
```

### Celery Task

```python
# app/tasks/llm.py
"""
LLM processing tasks.
"""
import httpx
import structlog
from datetime import datetime

from app.core.celery import celery_app
from app.db.session import sync_session_maker
from app.repositories.completion import CompletionRepository
from app.services.llm import LLMService

logger = structlog.get_logger()


@celery_app.task(bind=True, max_retries=2, soft_time_limit=600)
def process_completion(self, task_id: str):
    """Process LLM completion asynchronously."""
    logger.info("Processing completion", task_id=task_id)

    with sync_session_maker() as session:
        repo = CompletionRepository(session)
        llm_service = LLMService()

        # Get task
        task = repo.get_by_task_id(task_id)
        if not task:
            logger.error("Task not found", task_id=task_id)
            return

        if task.status == "cancelled":
            logger.info("Task cancelled, skipping", task_id=task_id)
            return

        try:
            # Update status
            task.status = "processing"
            task.started_at = datetime.utcnow()
            session.commit()

            # Call LLM API (this can take minutes)
            response = llm_service.create_completion(
                prompt=task.prompt,
                model=task.model,
                **task.parameters,
            )

            # Update with result
            task.status = "completed"
            task.content = response.content
            task.usage = response.usage.dict()
            task.completed_at = datetime.utcnow()
            session.commit()

            logger.info(
                "Completion processed",
                task_id=task_id,
                tokens=response.usage.total_tokens,
            )

            # Send webhook if configured
            if task.webhook_url:
                send_webhook(task)

        except Exception as exc:
            logger.error(
                "Completion failed",
                task_id=task_id,
                error=str(exc),
            )

            task.status = "failed"
            task.error = str(exc)
            task.completed_at = datetime.utcnow()
            session.commit()

            # Send webhook with error
            if task.webhook_url:
                send_webhook(task)

            # Retry for transient errors
            if is_transient_error(exc):
                raise self.retry(exc=exc, countdown=60)


def send_webhook(task):
    """Send webhook notification."""
    try:
        payload = {
            "task_id": task.task_id,
            "status": task.status,
            "content": task.content,
            "usage": task.usage,
            "error": task.error,
            "metadata": task.metadata,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        }

        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                task.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()

        logger.info("Webhook sent", task_id=task.task_id, url=task.webhook_url)

    except Exception as e:
        logger.error(
            "Webhook failed",
            task_id=task.task_id,
            url=task.webhook_url,
            error=str(e),
        )


def is_transient_error(exc: Exception) -> bool:
    """Check if error is transient (should retry)."""
    transient_errors = [
        "rate_limit",
        "timeout",
        "connection",
        "503",
        "502",
        "429",
    ]
    error_str = str(exc).lower()
    return any(err in error_str for err in transient_errors)
```

### Streaming Service

```python
# app/services/completion.py
"""
Completion service with streaming support.
"""
import uuid
from typing import AsyncGenerator

import openai
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.completion_task import CompletionTask
from app.schemas.completion import CompletionRequest, StreamChunk


class CompletionService:
    """Service for LLM completions."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.client = openai.AsyncOpenAI()

    async def create_task(
        self,
        user_id: int,
        api_key_id: int,
        request: CompletionRequest,
    ) -> CompletionTask:
        """Create completion task record."""
        task = CompletionTask(
            task_id=str(uuid.uuid4()),
            user_id=user_id,
            api_key_id=api_key_id,
            prompt=request.prompt,
            model=request.model,
            parameters={
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
            },
            webhook_url=str(request.webhook_url) if request.webhook_url else None,
            metadata=request.metadata or {},
            status="pending",
        )
        self.session.add(task)
        await self.session.commit()
        return task

    async def stream_completion(
        self,
        request: CompletionRequest,
        user_id: int,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream completion tokens."""
        stream = await self.client.chat.completions.create(
            model=request.model,
            messages=[{"role": "user", "content": request.prompt}],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield StreamChunk(
                    content=chunk.choices[0].delta.content,
                    finish_reason=chunk.choices[0].finish_reason,
                )
```

## Client Examples

### Webhook Pattern (Python)

```python
# Client code
import httpx

# 1. Create completion request
response = httpx.post(
    "https://api.example.com/completions",
    json={
        "prompt": "Write a long story...",
        "model": "gpt-4",
        "webhook_url": "https://myapp.com/webhook/completions",
        "metadata": {"request_id": "abc123"},
    },
    headers={"X-API-Key": "sk_..."},
)
task = response.json()
print(f"Task created: {task['task_id']}")

# 2. Webhook endpoint on client side
@app.post("/webhook/completions")
async def handle_completion_webhook(data: dict):
    if data["status"] == "completed":
        print(f"Got result: {data['content']}")
    else:
        print(f"Task failed: {data['error']}")
```

### Polling Pattern (JavaScript)

```javascript
// 1. Create request
const response = await fetch('/api/completions', {
  method: 'POST',
  body: JSON.stringify({ prompt: 'Write a story...' }),
  headers: { 'X-API-Key': apiKey },
});
const { task_id } = await response.json();

// 2. Poll for result
async function pollForResult(taskId, maxAttempts = 60) {
  for (let i = 0; i < maxAttempts; i++) {
    const res = await fetch(`/api/completions/${taskId}`);
    const task = await res.json();

    if (task.status === 'completed') {
      return task.content;
    }
    if (task.status === 'failed') {
      throw new Error(task.error);
    }

    // Wait before next poll (exponential backoff)
    await new Promise(r => setTimeout(r, Math.min(1000 * (i + 1), 10000)));
  }
  throw new Error('Timeout waiting for completion');
}

const result = await pollForResult(task_id);
```

### Streaming Pattern (JavaScript)

```javascript
// SSE streaming
const eventSource = new EventSource('/api/completions/stream?prompt=...');

eventSource.onmessage = (event) => {
  if (event.data === '[DONE]') {
    eventSource.close();
    return;
  }
  const chunk = JSON.parse(event.data);
  console.log(chunk.content); // Print tokens as they arrive
};

eventSource.onerror = (error) => {
  console.error('Stream error:', error);
  eventSource.close();
};
```

## Timeout Configuration

### API Gateway (nginx)

```nginx
location /api/completions/stream {
    proxy_pass http://backend;
    proxy_http_version 1.1;
    proxy_set_header Connection "";

    # Long timeout for streaming
    proxy_read_timeout 600s;
    proxy_send_timeout 600s;
}

location /api/completions {
    proxy_pass http://backend;

    # Short timeout - just queue the task
    proxy_read_timeout 30s;
}
```

### FastAPI Settings

```python
# For streaming endpoints
@app.middleware("http")
async def add_timeout(request: Request, call_next):
    if "/stream" in request.url.path:
        # Long timeout for streaming
        return await asyncio.wait_for(call_next(request), timeout=600)
    else:
        # Normal timeout
        return await asyncio.wait_for(call_next(request), timeout=30)
```

## Best Practices

1. **Always return task_id immediately** - Never block on LLM call
2. **Provide estimated time** - Help clients set expectations
3. **Support cancellation** - Users should be able to cancel pending tasks
4. **Implement webhook retries** - Webhooks can fail too
5. **Track usage/costs** - Record tokens for billing
6. **Set hard timeouts** - Prevent runaway tasks
7. **Use idempotency keys** - Prevent duplicate processing

## LLM Async Checklist

- [ ] Async endpoint returns task_id immediately (202 Accepted)
- [ ] Webhook callback support
- [ ] Polling endpoint for status check
- [ ] Streaming endpoint (SSE) for supported clients
- [ ] Task cancellation support
- [ ] Progress tracking for long tasks
- [ ] Webhook retry logic
- [ ] Proper timeout configuration at all layers
- [ ] Usage/cost tracking per task
- [ ] Idempotency for retries
