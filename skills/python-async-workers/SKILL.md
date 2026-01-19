---
name: python-async-workers
description: Generate Python background workers, cron jobs, message queue consumers. Use when creating scheduled tasks (APScheduler), Celery workers, RabbitMQ/Redis queue consumers, or async job processing. Follows project async patterns and integrates with existing services.
allowed-tools:
  - Read
  - Write
  - Edit
  - Glob
  - Grep
  - Bash
---

# Python Async Workers

Background processing patterns for FastAPI applications.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Application                     │
├─────────────────────────────────────────────────────────────┤
│  API Request  │  Cron Scheduler  │  Queue Consumer          │
│      ↓        │        ↓         │        ↓                 │
│  Enqueue Job  │  Trigger Task    │  Process Message         │
│      ↓        │        ↓         │        ↓                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   Service Layer                      │   │
│  │              (Shared Business Logic)                 │   │
│  └─────────────────────────────────────────────────────┘   │
│      ↓                   ↓                  ↓               │
│  Message Queue      Database            External APIs       │
│  (RabbitMQ/Redis)   (PostgreSQL)                            │
└─────────────────────────────────────────────────────────────┘
```

## Components

| Component | Purpose | Location |
|-----------|---------|----------|
| **Cron Jobs** | Scheduled recurring tasks | `app/jobs/` |
| **Celery Tasks** | Distributed task queue | `app/tasks/` |
| **Queue Consumers** | Message processing | `app/consumers/` |
| **Workers** | Background processors | `app/workers/` |

## When to Use What

| Use Case | Solution |
|----------|----------|
| Run every X minutes/hours | Cron (APScheduler) |
| Async processing after API call | Celery Task |
| Process messages from external system | Queue Consumer |
| Long-running background process | Worker with asyncio |
| Distributed across multiple servers | Celery + RabbitMQ |
| Simple in-process background | FastAPI BackgroundTasks |

## Key Principles

### 1. Reuse Service Layer

Workers should call existing services, not duplicate logic:

```python
# GOOD: Reuse service
async def process_order(order_id: int):
    async with get_session() as session:
        service = OrderService(session)
        await service.process(order_id)

# BAD: Duplicate logic in worker
async def process_order(order_id: int):
    # Don't copy-paste service code here!
    pass
```

### 2. Idempotency

All background tasks must be idempotent (safe to retry):

```python
# GOOD: Check before processing
async def send_email(user_id: int, email_type: str):
    if await was_email_sent(user_id, email_type):
        return  # Already sent, skip
    await do_send_email(user_id, email_type)
    await mark_email_sent(user_id, email_type)
```

### 3. Error Handling

Always handle errors gracefully with retries:

```python
@celery.task(bind=True, max_retries=3)
def process_payment(self, payment_id: int):
    try:
        # Process
        pass
    except TransientError as e:
        raise self.retry(exc=e, countdown=60)
    except PermanentError as e:
        # Log and don't retry
        logger.error(f"Payment {payment_id} failed permanently: {e}")
```

## Reference Navigation

**Scheduling:**

- [python-cron-jobs.md](references/python-cron-jobs.md) - APScheduler, periodic tasks

**Task Queues:**

- [python-celery.md](references/python-celery.md) - Celery tasks, workers, beat
- [python-llm-async.md](references/python-llm-async.md) - Long-running LLM requests, webhooks, polling

**Message Queues:**

- [python-message-queue.md](references/python-message-queue.md) - RabbitMQ, Redis streams
- [python-consumers.md](references/python-consumers.md) - Consumer patterns, error handling

## Integration with python-backend-development

This skill extends `python-backend-development`:

- Workers call **Services** from `app/services/`
- Use same **Repository** pattern for database access
- Follow same **async patterns** (async def, await)
- Use same **error handling** (custom exceptions)
- Share **configuration** from `app/core/config.py`
