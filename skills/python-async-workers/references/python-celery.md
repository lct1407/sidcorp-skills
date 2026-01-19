# Python Celery

Distributed task queue with Celery for FastAPI applications.

## Setup

### Installation

```bash
pip install celery[redis]
# Or with RabbitMQ
pip install celery[rabbitmq]
```

### Celery Configuration

```python
# app/core/celery.py
"""
Celery configuration.
"""
from celery import Celery

from app.core.config import settings

celery_app = Celery(
    "ai-services",
    broker=settings.CELERY_BROKER_URL,  # redis://localhost:6379/0
    backend=settings.CELERY_RESULT_BACKEND,  # redis://localhost:6379/1
    include=[
        "app.tasks.email",
        "app.tasks.ai",
        "app.tasks.export",
    ],
)

# Configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    # Task execution
    task_acks_late=True,  # Acknowledge after task completes
    task_reject_on_worker_lost=True,
    task_time_limit=3600,  # 1 hour hard limit
    task_soft_time_limit=3000,  # 50 min soft limit

    # Worker settings
    worker_prefetch_multiplier=1,  # One task at a time per worker
    worker_concurrency=4,

    # Result settings
    result_expires=86400,  # Results expire after 1 day

    # Retry settings
    task_default_retry_delay=60,  # 1 minute
    task_max_retries=3,
)

# Celery Beat schedule (periodic tasks)
celery_app.conf.beat_schedule = {
    "cleanup-every-hour": {
        "task": "app.tasks.cleanup.cleanup_expired_data",
        "schedule": 3600.0,  # Every hour
    },
    "daily-report": {
        "task": "app.tasks.reports.generate_daily_report",
        "schedule": {
            "hour": 2,
            "minute": 0,
        },
    },
}
```

## Task Definitions

### Basic Task

```python
# app/tasks/email.py
"""
Email tasks.
"""
import structlog
from celery import shared_task

from app.core.celery import celery_app
from app.services.email import EmailService

logger = structlog.get_logger()


@celery_app.task(bind=True, max_retries=3)
def send_welcome_email(self, user_id: int, email: str):
    """Send welcome email to new user."""
    logger.info("Sending welcome email", user_id=user_id, email=email)

    try:
        service = EmailService()
        service.send_welcome(email)

        logger.info("Welcome email sent", user_id=user_id)

    except Exception as exc:
        logger.error(
            "Failed to send welcome email",
            user_id=user_id,
            error=str(exc),
        )
        raise self.retry(exc=exc, countdown=60 * (self.request.retries + 1))


@celery_app.task(bind=True, max_retries=3)
def send_notification_email(self, user_id: int, subject: str, body: str):
    """Send notification email."""
    try:
        service = EmailService()
        service.send_notification(user_id, subject, body)
    except Exception as exc:
        raise self.retry(exc=exc, countdown=60)
```

### Async Task (with Database)

```python
# app/tasks/ai.py
"""
AI processing tasks.
"""
import asyncio
import structlog
from celery import shared_task

from app.core.celery import celery_app
from app.db.session import sync_session_maker  # Sync session for Celery
from app.services.ai import AIService
from app.repositories.chat import ChatRepository

logger = structlog.get_logger()


def run_async(coro):
    """Run async function in sync context."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(bind=True, max_retries=3, soft_time_limit=300)
def process_ai_completion(self, request_id: str, prompt: str, user_id: int):
    """Process AI completion request."""
    logger.info("Processing AI completion", request_id=request_id)

    try:
        with sync_session_maker() as session:
            ai_service = AIService()
            chat_repo = ChatRepository(session)

            # Call AI API
            response = ai_service.generate_completion(prompt)

            # Save to database
            chat_repo.save_completion(
                request_id=request_id,
                user_id=user_id,
                prompt=prompt,
                response=response,
            )
            session.commit()

            logger.info(
                "AI completion processed",
                request_id=request_id,
                tokens=response.usage.total_tokens,
            )

            return {
                "request_id": request_id,
                "status": "completed",
                "tokens": response.usage.total_tokens,
            }

    except Exception as exc:
        logger.error(
            "AI completion failed",
            request_id=request_id,
            error=str(exc),
        )
        raise self.retry(exc=exc, countdown=30)
```

### Task with Progress Tracking

```python
# app/tasks/export.py
"""
Export tasks with progress tracking.
"""
import structlog
from celery import shared_task

from app.core.celery import celery_app
from app.core.cache import cache_set

logger = structlog.get_logger()


@celery_app.task(bind=True)
def export_user_data(self, user_id: int, export_format: str):
    """Export user data with progress updates."""
    task_id = self.request.id
    total_steps = 4

    def update_progress(step: int, message: str):
        progress = {
            "current": step,
            "total": total_steps,
            "percent": int((step / total_steps) * 100),
            "message": message,
        }
        # Store progress in Redis
        cache_set(f"export_progress:{task_id}", progress, ttl=3600)
        # Also update task state
        self.update_state(state="PROGRESS", meta=progress)

    try:
        update_progress(1, "Fetching user data...")
        user_data = fetch_user_data(user_id)

        update_progress(2, "Fetching activity history...")
        activity_data = fetch_activity_data(user_id)

        update_progress(3, "Generating export file...")
        file_path = generate_export(user_data, activity_data, export_format)

        update_progress(4, "Complete!")

        return {
            "status": "completed",
            "file_path": file_path,
        }

    except Exception as exc:
        logger.error("Export failed", user_id=user_id, error=str(exc))
        raise
```

## Calling Tasks

### From FastAPI Endpoints

```python
# app/api/v1/endpoints/users.py
"""
User endpoints with task dispatching.
"""
from fastapi import APIRouter, BackgroundTasks

from app.tasks.email import send_welcome_email
from app.tasks.export import export_user_data
from app.schemas.user import UserCreate, UserResponse

router = APIRouter(prefix="/users", tags=["Users"])


@router.post("", response_model=UserResponse, status_code=201)
async def create_user(data: UserCreate):
    """Create user and send welcome email."""
    user = await user_service.create(data)

    # Dispatch Celery task (non-blocking)
    send_welcome_email.delay(user.id, user.email)

    return user


@router.post("/{user_id}/export")
async def request_export(user_id: int, format: str = "json"):
    """Request data export (async)."""
    # Dispatch export task
    task = export_user_data.delay(user_id, format)

    return {
        "task_id": task.id,
        "status": "pending",
        "check_url": f"/api/v1/tasks/{task.id}",
    }
```

### Task Status Endpoint

```python
# app/api/v1/endpoints/tasks.py
"""
Task status endpoints.
"""
from fastapi import APIRouter, HTTPException

from app.core.celery import celery_app

router = APIRouter(prefix="/tasks", tags=["Tasks"])


@router.get("/{task_id}")
async def get_task_status(task_id: str):
    """Get task status and result."""
    result = celery_app.AsyncResult(task_id)

    response = {
        "task_id": task_id,
        "status": result.status,
    }

    if result.status == "PENDING":
        response["message"] = "Task is waiting to be processed"

    elif result.status == "PROGRESS":
        response["progress"] = result.info

    elif result.status == "SUCCESS":
        response["result"] = result.result

    elif result.status == "FAILURE":
        response["error"] = str(result.result)

    return response
```

## Task Chains & Groups

### Chain (Sequential)

```python
from celery import chain

# Run tasks in sequence, passing result to next
workflow = chain(
    fetch_data.s(user_id),
    process_data.s(),
    save_results.s(),
)
result = workflow.apply_async()
```

### Group (Parallel)

```python
from celery import group

# Run tasks in parallel
workflow = group(
    send_email.s(user1_id),
    send_email.s(user2_id),
    send_email.s(user3_id),
)
result = workflow.apply_async()
```

### Chord (Parallel + Callback)

```python
from celery import chord

# Run parallel tasks, then callback with all results
workflow = chord(
    [process_item.s(item) for item in items],
    aggregate_results.s(),
)
result = workflow.apply_async()
```

## Running Workers

### Development

```bash
# Start worker
celery -A app.core.celery worker --loglevel=info

# Start beat (scheduler)
celery -A app.core.celery beat --loglevel=info

# Start both (dev only)
celery -A app.core.celery worker --beat --loglevel=info
```

### Production (systemd)

```ini
# /etc/systemd/system/celery-worker.service
[Unit]
Description=Celery Worker
After=network.target

[Service]
Type=forking
User=appuser
Group=appgroup
WorkingDirectory=/opt/ai-services
ExecStart=/opt/ai-services/.venv/bin/celery -A app.core.celery worker --loglevel=info --concurrency=4
ExecStop=/bin/kill -TERM $MAINPID
Restart=always

[Install]
WantedBy=multi-user.target
```

### Docker

```dockerfile
# Dockerfile.worker
FROM python:3.14-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY ./app /app/app

CMD ["celery", "-A", "app.core.celery", "worker", "--loglevel=info"]
```

```yaml
# docker-compose.yml
services:
  worker:
    build:
      context: .
      dockerfile: Dockerfile.worker
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CELERY_RESULT_BACKEND=redis://redis:6379/1
    depends_on:
      - redis
    deploy:
      replicas: 2

  beat:
    build:
      context: .
      dockerfile: Dockerfile.worker
    command: celery -A app.core.celery beat --loglevel=info
    depends_on:
      - redis
```

## Monitoring

### Flower (Web UI)

```bash
pip install flower
celery -A app.core.celery flower --port=5555
```

### Prometheus Metrics

```python
# app/core/celery.py (add)
from celery.signals import task_prerun, task_postrun, task_failure

@task_prerun.connect
def task_prerun_handler(task_id, task, *args, **kwargs):
    # Increment task started counter
    pass

@task_postrun.connect
def task_postrun_handler(task_id, task, *args, retval=None, state=None, **kwargs):
    # Record task duration, increment success counter
    pass

@task_failure.connect
def task_failure_handler(task_id, exception, *args, **kwargs):
    # Increment failure counter
    pass
```

## Celery Checklist

- [ ] Task serializer set to JSON
- [ ] `task_acks_late=True` for reliability
- [ ] `bind=True` for task retry access
- [ ] `max_retries` and `countdown` configured
- [ ] `soft_time_limit` and `time_limit` set
- [ ] Result backend configured
- [ ] Progress tracking for long tasks
- [ ] Proper error handling with logging
- [ ] Separate worker and beat processes in production
- [ ] Monitoring with Flower or Prometheus
