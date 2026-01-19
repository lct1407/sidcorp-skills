# Python Cron Jobs

Scheduled tasks with APScheduler for FastAPI applications.

## APScheduler Setup

### Installation

```bash
pip install apscheduler
```

### Basic Configuration

```python
# app/core/scheduler.py
"""
APScheduler configuration for cron jobs.
"""
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.executors.asyncio import AsyncIOExecutor

from app.core.config import settings

# Job stores for persistence
jobstores = {
    "default": SQLAlchemyJobStore(url=settings.DATABASE_URL.replace("+asyncpg", "")),
}

# Executors
executors = {
    "default": AsyncIOExecutor(),
}

# Job defaults
job_defaults = {
    "coalesce": True,  # Combine missed runs into one
    "max_instances": 1,  # Only one instance at a time
    "misfire_grace_time": 60,  # Seconds to still run if missed
}

scheduler = AsyncIOScheduler(
    jobstores=jobstores,
    executors=executors,
    job_defaults=job_defaults,
    timezone="UTC",
)
```

### Integration with FastAPI

```python
# app/main.py
"""
FastAPI app with scheduler lifecycle.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.scheduler import scheduler
from app.jobs import register_jobs


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - start/stop scheduler."""
    # Startup
    register_jobs(scheduler)
    scheduler.start()
    yield
    # Shutdown
    scheduler.shutdown(wait=True)


app = FastAPI(lifespan=lifespan)
```

## Job Definitions

### Basic Job Structure

```python
# app/jobs/__init__.py
"""
Job registration.
"""
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.jobs.cleanup import cleanup_expired_tokens
from app.jobs.reports import generate_daily_report
from app.jobs.sync import sync_external_data


def register_jobs(scheduler: AsyncIOScheduler):
    """Register all cron jobs."""

    # Every hour
    scheduler.add_job(
        cleanup_expired_tokens,
        "interval",
        hours=1,
        id="cleanup_expired_tokens",
        replace_existing=True,
    )

    # Daily at 2 AM UTC
    scheduler.add_job(
        generate_daily_report,
        "cron",
        hour=2,
        minute=0,
        id="generate_daily_report",
        replace_existing=True,
    )

    # Every 5 minutes
    scheduler.add_job(
        sync_external_data,
        "interval",
        minutes=5,
        id="sync_external_data",
        replace_existing=True,
    )
```

### Job Implementation

```python
# app/jobs/cleanup.py
"""
Cleanup jobs.
"""
import structlog

from app.db.session import async_session_maker
from app.services.token import TokenService

logger = structlog.get_logger()


async def cleanup_expired_tokens():
    """Remove expired tokens from database."""
    logger.info("Starting cleanup_expired_tokens job")

    try:
        async with async_session_maker() as session:
            service = TokenService(session)
            deleted_count = await service.delete_expired()

            logger.info(
                "cleanup_expired_tokens completed",
                deleted_count=deleted_count,
            )
    except Exception as e:
        logger.error(
            "cleanup_expired_tokens failed",
            error=str(e),
            exc_info=True,
        )
        raise  # Re-raise for APScheduler to handle
```

```python
# app/jobs/reports.py
"""
Report generation jobs.
"""
import structlog
from datetime import datetime, timedelta

from app.db.session import async_session_maker
from app.services.analytics import AnalyticsService
from app.services.notification import NotificationService

logger = structlog.get_logger()


async def generate_daily_report():
    """Generate and send daily usage report."""
    logger.info("Starting generate_daily_report job")

    yesterday = datetime.utcnow().date() - timedelta(days=1)

    try:
        async with async_session_maker() as session:
            analytics = AnalyticsService(session)
            notification = NotificationService(session)

            # Generate report
            report = await analytics.generate_daily_report(yesterday)

            # Send to admins
            await notification.send_admin_report(report)

            logger.info(
                "generate_daily_report completed",
                date=str(yesterday),
                total_requests=report.total_requests,
            )
    except Exception as e:
        logger.error(
            "generate_daily_report failed",
            date=str(yesterday),
            error=str(e),
            exc_info=True,
        )
        raise
```

## Schedule Types

### Interval Schedule

```python
# Every N seconds/minutes/hours/days/weeks
scheduler.add_job(
    my_job,
    "interval",
    seconds=30,      # Every 30 seconds
    # minutes=5,     # Every 5 minutes
    # hours=1,       # Every hour
    # days=1,        # Every day
    # weeks=1,       # Every week
    id="my_job",
)
```

### Cron Schedule

```python
# Cron-like scheduling
scheduler.add_job(
    my_job,
    "cron",
    # Time-based
    hour=2,          # At 2 AM
    minute=30,       # At minute 30
    second=0,        # At second 0

    # Day-based
    day_of_week="mon-fri",  # Weekdays only
    # day="1,15",           # 1st and 15th of month
    # month="1-6",          # January to June

    id="my_job",
)

# Examples:
# Every day at midnight:     hour=0, minute=0
# Every Monday at 9 AM:      day_of_week="mon", hour=9
# First of month at noon:    day=1, hour=12
# Every hour at :30:         minute=30
```

### Date Schedule (One-time)

```python
from datetime import datetime, timedelta

# Run once at specific time
scheduler.add_job(
    my_job,
    "date",
    run_date=datetime.utcnow() + timedelta(hours=1),
    id="my_one_time_job",
)
```

## Job Management API

```python
# app/api/v1/endpoints/jobs.py
"""
Job management endpoints (admin only).
"""
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_current_admin_user
from app.core.scheduler import scheduler
from app.schemas.job import JobResponse, JobListResponse

router = APIRouter(prefix="/jobs", tags=["Jobs"])


@router.get("", response_model=JobListResponse)
async def list_jobs(
    _: Annotated[None, Depends(get_current_admin_user)],
):
    """List all scheduled jobs."""
    jobs = scheduler.get_jobs()
    return JobListResponse(
        jobs=[
            JobResponse(
                id=job.id,
                name=job.name,
                next_run=job.next_run_time,
                trigger=str(job.trigger),
            )
            for job in jobs
        ]
    )


@router.post("/{job_id}/run")
async def run_job_now(
    job_id: str,
    _: Annotated[None, Depends(get_current_admin_user)],
):
    """Trigger a job to run immediately."""
    job = scheduler.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    job.modify(next_run_time=datetime.utcnow())
    return {"message": f"Job {job_id} triggered"}


@router.post("/{job_id}/pause")
async def pause_job(
    job_id: str,
    _: Annotated[None, Depends(get_current_admin_user)],
):
    """Pause a scheduled job."""
    job = scheduler.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    job.pause()
    return {"message": f"Job {job_id} paused"}


@router.post("/{job_id}/resume")
async def resume_job(
    job_id: str,
    _: Annotated[None, Depends(get_current_admin_user)],
):
    """Resume a paused job."""
    job = scheduler.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    job.resume()
    return {"message": f"Job {job_id} resumed"}
```

## Error Handling & Monitoring

### Job with Retry Logic

```python
# app/jobs/sync.py
"""
External sync with retry logic.
"""
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from app.db.session import async_session_maker
from app.services.external import ExternalService

logger = structlog.get_logger()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
)
async def sync_external_data():
    """Sync data from external API with retry."""
    logger.info("Starting sync_external_data job")

    async with async_session_maker() as session:
        service = ExternalService(session)
        result = await service.sync()

        logger.info(
            "sync_external_data completed",
            synced_count=result.synced_count,
        )
```

### Job Event Listeners

```python
# app/core/scheduler.py (continued)
"""
Job event listeners for monitoring.
"""
from apscheduler.events import (
    EVENT_JOB_ERROR,
    EVENT_JOB_EXECUTED,
    EVENT_JOB_MISSED,
)


def job_executed_listener(event):
    """Log successful job execution."""
    logger.info(
        "Job executed successfully",
        job_id=event.job_id,
        scheduled_time=event.scheduled_run_time,
    )


def job_error_listener(event):
    """Log job errors and alert."""
    logger.error(
        "Job failed",
        job_id=event.job_id,
        exception=str(event.exception),
        traceback=event.traceback,
    )
    # TODO: Send alert to monitoring system


def job_missed_listener(event):
    """Log missed job runs."""
    logger.warning(
        "Job missed",
        job_id=event.job_id,
        scheduled_time=event.scheduled_run_time,
    )


# Register listeners
scheduler.add_listener(job_executed_listener, EVENT_JOB_EXECUTED)
scheduler.add_listener(job_error_listener, EVENT_JOB_ERROR)
scheduler.add_listener(job_missed_listener, EVENT_JOB_MISSED)
```

## Testing Cron Jobs

```python
# tests/unit/test_jobs.py
"""
Unit tests for cron jobs.
"""
import pytest
from unittest.mock import AsyncMock, patch

from app.jobs.cleanup import cleanup_expired_tokens


class TestCleanupJob:
    """Tests for cleanup job."""

    @pytest.mark.asyncio
    async def test_cleanup_expired_tokens(self):
        """Test expired token cleanup."""
        with patch("app.jobs.cleanup.async_session_maker") as mock_session:
            mock_service = AsyncMock()
            mock_service.delete_expired.return_value = 5

            with patch("app.jobs.cleanup.TokenService", return_value=mock_service):
                await cleanup_expired_tokens()

            mock_service.delete_expired.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_handles_error(self):
        """Test cleanup handles database errors."""
        with patch("app.jobs.cleanup.async_session_maker") as mock_session:
            mock_session.side_effect = Exception("DB Error")

            with pytest.raises(Exception):
                await cleanup_expired_tokens()
```

## Cron Jobs Checklist

- [ ] Jobs call Services, not duplicate logic
- [ ] Idempotent (safe to retry)
- [ ] Proper error handling with logging
- [ ] Job persistence configured (SQLAlchemyJobStore)
- [ ] `replace_existing=True` to avoid duplicates
- [ ] `coalesce=True` for missed runs
- [ ] `max_instances=1` to prevent overlap
- [ ] Event listeners for monitoring
- [ ] Admin API for job management
- [ ] Tests for job logic
