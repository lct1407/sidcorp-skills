# Python Consumers

Message queue consumer patterns and error handling for FastAPI applications.

## Consumer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Consumer Worker                         │
├─────────────────────────────────────────────────────────────┤
│  1. Connect to Queue                                         │
│  2. Consume Message                                          │
│  3. Process (call Service)                                   │
│  4. Acknowledge/Reject                                       │
│  5. Handle Errors (retry/DLQ)                               │
└─────────────────────────────────────────────────────────────┘
```

## RabbitMQ Consumer

### Basic Consumer

```python
# app/consumers/email.py
"""
Email queue consumer.
"""
import asyncio
import json
import structlog

import aio_pika
from aio_pika import IncomingMessage

from app.core.rabbitmq import get_rabbitmq_channel, QUEUE_EMAIL
from app.services.email import EmailService

logger = structlog.get_logger()


class EmailConsumer:
    """Consumer for email queue."""

    def __init__(self):
        self.email_service = EmailService()
        self.running = False

    async def start(self):
        """Start consuming messages."""
        self.running = True
        channel = await get_rabbitmq_channel()
        queue = await channel.get_queue(QUEUE_EMAIL)

        logger.info("Email consumer started", queue=QUEUE_EMAIL)

        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                if not self.running:
                    break
                await self.process_message(message)

    async def stop(self):
        """Stop consuming."""
        self.running = False
        logger.info("Email consumer stopping")

    async def process_message(self, message: IncomingMessage):
        """Process single message."""
        try:
            async with message.process(requeue=False):
                payload = json.loads(message.body)

                logger.info(
                    "Processing email",
                    message_id=message.message_id,
                    to=payload.get("to"),
                )

                await self.email_service.send(
                    to=payload["to"],
                    subject=payload["subject"],
                    body=payload["body"],
                    template=payload.get("template"),
                )

                logger.info(
                    "Email sent",
                    message_id=message.message_id,
                )

        except Exception as e:
            logger.error(
                "Email processing failed",
                message_id=message.message_id,
                error=str(e),
                exc_info=True,
            )
            # Message will be sent to DLQ due to requeue=False
```

### Consumer with Retry

```python
# app/consumers/ai.py
"""
AI completion consumer with retry logic.
"""
import asyncio
import json
import structlog

import aio_pika
from aio_pika import IncomingMessage

from app.core.rabbitmq import get_rabbitmq_channel, QUEUE_AI_COMPLETION
from app.db.session import async_session_maker
from app.services.completion import CompletionService

logger = structlog.get_logger()

MAX_RETRIES = 3
RETRY_DELAYS = [60, 300, 900]  # 1min, 5min, 15min


class AIConsumer:
    """Consumer for AI completion queue."""

    async def process_message(self, message: IncomingMessage):
        """Process with retry logic."""
        retry_count = message.headers.get("x-retry-count", 0)

        try:
            async with message.process(requeue=False):
                payload = json.loads(message.body)

                logger.info(
                    "Processing AI completion",
                    task_id=payload["task_id"],
                    retry=retry_count,
                )

                async with async_session_maker() as session:
                    service = CompletionService(session)
                    await service.process_completion(
                        task_id=payload["task_id"],
                        prompt=payload["prompt"],
                        model=payload["model"],
                    )

                logger.info(
                    "AI completion processed",
                    task_id=payload["task_id"],
                )

        except Exception as e:
            logger.error(
                "AI completion failed",
                task_id=payload.get("task_id"),
                retry=retry_count,
                error=str(e),
            )

            # Retry if under limit
            if retry_count < MAX_RETRIES:
                await self.retry_message(message, retry_count)
            else:
                logger.error(
                    "AI completion max retries exceeded",
                    task_id=payload.get("task_id"),
                )
                # Message goes to DLQ

    async def retry_message(self, message: IncomingMessage, retry_count: int):
        """Republish message for retry."""
        channel = await get_rabbitmq_channel()
        exchange = await channel.get_exchange("tasks")

        delay = RETRY_DELAYS[retry_count]

        # Create new message with updated retry count
        new_message = aio_pika.Message(
            body=message.body,
            headers={"x-retry-count": retry_count + 1},
            expiration=str(delay * 1000),  # Delay in ms
        )

        # Publish to delay queue
        await exchange.publish(
            new_message,
            routing_key="ai.retry",
        )

        logger.info(
            "Message scheduled for retry",
            message_id=message.message_id,
            retry=retry_count + 1,
            delay_seconds=delay,
        )
```

## Redis Stream Consumer

```python
# app/consumers/stream_consumer.py
"""
Redis Stream consumer.
"""
import asyncio
import structlog

from app.core.redis_stream import RedisStreamService
from app.services.email import EmailService

logger = structlog.get_logger()


class StreamConsumer:
    """Generic Redis Stream consumer."""

    def __init__(
        self,
        stream: RedisStreamService,
        consumer_name: str,
        handler,
    ):
        self.stream = stream
        self.consumer_name = consumer_name
        self.handler = handler
        self.running = False

    async def start(self):
        """Start consuming from stream."""
        self.running = True
        logger.info(
            "Stream consumer started",
            stream=self.stream.stream_name,
            consumer=self.consumer_name,
        )

        while self.running:
            try:
                # Process new messages
                messages = await self.stream.consume(
                    consumer_name=self.consumer_name,
                    count=10,
                    block=5000,
                )

                for message_id, data in messages:
                    await self.process_message(message_id, data)

                # Claim pending messages from dead consumers
                pending = await self.stream.claim_pending(
                    consumer_name=self.consumer_name,
                    min_idle_time=60000,  # 1 minute
                )

                for message_id, data in pending:
                    logger.info(
                        "Processing claimed message",
                        message_id=message_id,
                    )
                    await self.process_message(message_id, data)

            except Exception as e:
                logger.error(
                    "Consumer error",
                    error=str(e),
                    exc_info=True,
                )
                await asyncio.sleep(5)  # Back off on error

    async def stop(self):
        """Stop consuming."""
        self.running = False

    async def process_message(self, message_id: str, data: dict):
        """Process single message."""
        try:
            await self.handler(data)
            await self.stream.ack(message_id)

            logger.info(
                "Message processed",
                message_id=message_id,
            )

        except Exception as e:
            logger.error(
                "Message processing failed",
                message_id=message_id,
                error=str(e),
            )
            # Don't ack - message stays pending for retry


# Usage
async def handle_email(data: dict):
    service = EmailService()
    await service.send(
        to=data["to"],
        subject=data["subject"],
        body=data["body"],
    )

email_consumer = StreamConsumer(
    stream=email_stream,
    consumer_name="email-worker-1",
    handler=handle_email,
)
```

## Worker Process

### Standalone Worker

```python
# app/workers/main.py
"""
Worker process entry point.
"""
import asyncio
import signal
import structlog

from app.consumers.email import EmailConsumer
from app.consumers.ai import AIConsumer
from app.core.rabbitmq import setup_rabbitmq, close_rabbitmq

logger = structlog.get_logger()

# Global consumers
consumers = []


async def main():
    """Main worker process."""
    # Setup
    await setup_rabbitmq()

    # Create consumers
    email_consumer = EmailConsumer()
    ai_consumer = AIConsumer()
    consumers.extend([email_consumer, ai_consumer])

    # Start consumers concurrently
    await asyncio.gather(
        email_consumer.start(),
        ai_consumer.start(),
    )


async def shutdown():
    """Graceful shutdown."""
    logger.info("Shutting down workers...")

    # Stop all consumers
    for consumer in consumers:
        await consumer.stop()

    # Close connections
    await close_rabbitmq()

    logger.info("Shutdown complete")


def handle_signal(sig):
    """Handle shutdown signals."""
    logger.info(f"Received signal {sig}")
    asyncio.create_task(shutdown())


if __name__ == "__main__":
    # Setup signal handlers
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda s=sig: handle_signal(s))

    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(shutdown())
        loop.close()
```

### Docker Worker

```dockerfile
# Dockerfile.worker
FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /app/app

# Run worker
CMD ["python", "-m", "app.workers.main"]
```

```yaml
# docker-compose.yml
services:
  worker:
    build:
      context: .
      dockerfile: Dockerfile.worker
    environment:
      - RABBITMQ_URL=amqp://guest:guest@rabbitmq:5672/
      - DATABASE_URL=postgresql+asyncpg://user:pass@db:5432/app
    depends_on:
      - rabbitmq
      - db
    deploy:
      replicas: 3
    restart: unless-stopped
```

## Error Handling Patterns

### Dead Letter Queue Handler

```python
# app/consumers/dlq.py
"""
Dead letter queue handler for failed messages.
"""
import json
import structlog

from app.core.rabbitmq import get_rabbitmq_channel
from app.services.notification import NotificationService

logger = structlog.get_logger()


class DLQHandler:
    """Handle messages in dead letter queue."""

    async def start(self, queue_name: str):
        """Process DLQ messages."""
        channel = await get_rabbitmq_channel()
        queue = await channel.get_queue(f"{queue_name}.dlq")

        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                await self.handle_dead_letter(message, queue_name)

    async def handle_dead_letter(self, message, original_queue: str):
        """Handle a dead letter message."""
        async with message.process():
            payload = json.loads(message.body)

            logger.error(
                "Dead letter received",
                queue=original_queue,
                message_id=message.message_id,
                payload=payload,
            )

            # Notify admins
            await NotificationService().send_alert(
                title=f"Dead Letter: {original_queue}",
                message=f"Message {message.message_id} failed permanently",
                data=payload,
            )

            # Store for manual review
            await self.store_for_review(message, original_queue)

    async def store_for_review(self, message, queue_name: str):
        """Store dead letter for manual review."""
        # Store in database or S3 for later analysis
        pass
```

### Circuit Breaker

```python
# app/consumers/circuit_breaker.py
"""
Circuit breaker for consumer resilience.
"""
import asyncio
from datetime import datetime, timedelta
from enum import Enum
import structlog

logger = structlog.get_logger()


class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker for external service calls."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_max_calls: int = 3,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0

    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            if self._should_try_recovery():
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
            else:
                raise CircuitOpenError("Circuit breaker is open")

        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls >= self.half_open_max_calls:
                raise CircuitOpenError("Half-open limit reached")
            self.half_open_calls += 1

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _should_try_recovery(self) -> bool:
        """Check if enough time passed to try recovery."""
        if self.last_failure_time is None:
            return True
        return datetime.utcnow() > self.last_failure_time + timedelta(
            seconds=self.recovery_timeout
        )

    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                "Circuit breaker opened",
                failures=self.failure_count,
            )


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


# Usage in consumer
circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)


async def call_external_service(data):
    return await circuit_breaker.call(
        external_api.process,
        data,
    )
```

## Health Checks

```python
# app/consumers/health.py
"""
Consumer health check endpoint.
"""
from fastapi import APIRouter

from app.core.rabbitmq import get_rabbitmq_connection
from app.core.redis_stream import get_redis

router = APIRouter()


@router.get("/health/consumers")
async def consumer_health():
    """Check consumer dependencies."""
    checks = {}

    # RabbitMQ
    try:
        conn = await get_rabbitmq_connection()
        checks["rabbitmq"] = "healthy" if not conn.is_closed else "unhealthy"
    except Exception as e:
        checks["rabbitmq"] = f"unhealthy: {str(e)}"

    # Redis
    try:
        redis = await get_redis()
        await redis.ping()
        checks["redis"] = "healthy"
    except Exception as e:
        checks["redis"] = f"unhealthy: {str(e)}"

    all_healthy = all(v == "healthy" for v in checks.values())

    return {
        "status": "healthy" if all_healthy else "unhealthy",
        "checks": checks,
    }
```

## Consumer Checklist

- [ ] Graceful shutdown handling (SIGTERM/SIGINT)
- [ ] Message acknowledgment after successful processing
- [ ] Retry logic with exponential backoff
- [ ] Dead letter queue for permanently failed messages
- [ ] Circuit breaker for external dependencies
- [ ] Structured logging with message IDs
- [ ] Health check endpoints
- [ ] Metrics/monitoring (message rate, processing time)
- [ ] Idempotent message processing
- [ ] Connection recovery on failure
