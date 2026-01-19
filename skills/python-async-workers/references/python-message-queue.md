# Python Message Queue

RabbitMQ and Redis message queue patterns for FastAPI applications.

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Producer   │────▶│   Queue     │────▶│  Consumer   │
│  (FastAPI)  │     │ (RabbitMQ/  │     │  (Worker)   │
│             │     │   Redis)    │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
```

## RabbitMQ

### Installation

```bash
pip install aio-pika
```

### Connection Manager

```python
# app/core/rabbitmq.py
"""
RabbitMQ connection management.
"""
import aio_pika
from aio_pika import Channel, Connection, ExchangeType
from aio_pika.abc import AbstractRobustConnection

from app.core.config import settings

_connection: AbstractRobustConnection | None = None
_channel: Channel | None = None


async def get_rabbitmq_connection() -> AbstractRobustConnection:
    """Get or create RabbitMQ connection."""
    global _connection
    if _connection is None or _connection.is_closed:
        _connection = await aio_pika.connect_robust(
            settings.RABBITMQ_URL,  # amqp://user:password@localhost:5672/
            client_properties={"connection_name": "ai-services"},
        )
    return _connection


async def get_rabbitmq_channel() -> Channel:
    """Get or create RabbitMQ channel."""
    global _channel
    connection = await get_rabbitmq_connection()
    if _channel is None or _channel.is_closed:
        _channel = await connection.channel()
        await _channel.set_qos(prefetch_count=10)
    return _channel


async def close_rabbitmq():
    """Close RabbitMQ connection."""
    global _connection, _channel
    if _channel:
        await _channel.close()
        _channel = None
    if _connection:
        await _connection.close()
        _connection = None
```

### Queue Setup

```python
# app/core/rabbitmq.py (continued)
"""
Queue and exchange declarations.
"""

# Exchange names
EXCHANGE_EVENTS = "events"
EXCHANGE_TASKS = "tasks"
EXCHANGE_DLX = "dlx"  # Dead letter exchange

# Queue names
QUEUE_EMAIL = "email.send"
QUEUE_AI_COMPLETION = "ai.completion"
QUEUE_WEBHOOK = "webhook.deliver"


async def setup_rabbitmq():
    """Declare exchanges and queues."""
    channel = await get_rabbitmq_channel()

    # Dead letter exchange
    dlx = await channel.declare_exchange(
        EXCHANGE_DLX,
        ExchangeType.DIRECT,
        durable=True,
    )

    # Events exchange (fanout - broadcast)
    events_exchange = await channel.declare_exchange(
        EXCHANGE_EVENTS,
        ExchangeType.FANOUT,
        durable=True,
    )

    # Tasks exchange (direct - routing)
    tasks_exchange = await channel.declare_exchange(
        EXCHANGE_TASKS,
        ExchangeType.DIRECT,
        durable=True,
    )

    # Email queue with dead letter
    email_queue = await channel.declare_queue(
        QUEUE_EMAIL,
        durable=True,
        arguments={
            "x-dead-letter-exchange": EXCHANGE_DLX,
            "x-dead-letter-routing-key": f"{QUEUE_EMAIL}.dlq",
            "x-message-ttl": 86400000,  # 24 hours
        },
    )
    await email_queue.bind(tasks_exchange, routing_key="email")

    # AI completion queue
    ai_queue = await channel.declare_queue(
        QUEUE_AI_COMPLETION,
        durable=True,
        arguments={
            "x-dead-letter-exchange": EXCHANGE_DLX,
            "x-message-ttl": 3600000,  # 1 hour
        },
    )
    await ai_queue.bind(tasks_exchange, routing_key="ai")

    # Dead letter queues
    for queue_name in [QUEUE_EMAIL, QUEUE_AI_COMPLETION]:
        dlq = await channel.declare_queue(f"{queue_name}.dlq", durable=True)
        await dlq.bind(dlx, routing_key=f"{queue_name}.dlq")
```

### Producer (Publishing Messages)

```python
# app/services/queue.py
"""
Message queue producer service.
"""
import json
import uuid
from datetime import datetime
from typing import Any

import aio_pika
from aio_pika import Message

from app.core.rabbitmq import get_rabbitmq_channel, EXCHANGE_TASKS


class QueueService:
    """Service for publishing messages to queues."""

    async def publish(
        self,
        routing_key: str,
        payload: dict[str, Any],
        correlation_id: str | None = None,
        priority: int = 0,
    ) -> str:
        """Publish message to queue."""
        channel = await get_rabbitmq_channel()
        exchange = await channel.get_exchange(EXCHANGE_TASKS)

        message_id = correlation_id or str(uuid.uuid4())

        message = Message(
            body=json.dumps(payload).encode(),
            message_id=message_id,
            correlation_id=message_id,
            content_type="application/json",
            priority=priority,
            timestamp=datetime.utcnow(),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        )

        await exchange.publish(message, routing_key=routing_key)

        return message_id

    async def publish_email(
        self,
        to: str,
        subject: str,
        body: str,
        template: str | None = None,
    ) -> str:
        """Publish email task."""
        return await self.publish(
            routing_key="email",
            payload={
                "to": to,
                "subject": subject,
                "body": body,
                "template": template,
            },
        )

    async def publish_ai_completion(
        self,
        task_id: str,
        prompt: str,
        model: str,
        user_id: int,
    ) -> str:
        """Publish AI completion task."""
        return await self.publish(
            routing_key="ai",
            payload={
                "task_id": task_id,
                "prompt": prompt,
                "model": model,
                "user_id": user_id,
            },
            correlation_id=task_id,
        )
```

### Usage in FastAPI

```python
# app/api/v1/endpoints/completions.py
"""
API endpoint using queue.
"""
from fastapi import APIRouter, Depends

from app.services.queue import QueueService
from app.schemas.completion import CompletionRequest

router = APIRouter()


@router.post("/completions")
async def create_completion(
    request: CompletionRequest,
    queue: QueueService = Depends(),
):
    """Create async completion via queue."""
    task_id = str(uuid.uuid4())

    # Publish to queue (returns immediately)
    await queue.publish_ai_completion(
        task_id=task_id,
        prompt=request.prompt,
        model=request.model,
        user_id=request.user_id,
    )

    return {"task_id": task_id, "status": "queued"}
```

## Redis Streams

### Installation

```bash
pip install redis
```

### Redis Stream Manager

```python
# app/core/redis_stream.py
"""
Redis Streams for message queuing.
"""
import json
from typing import Any

import redis.asyncio as redis

from app.core.config import settings

_redis: redis.Redis | None = None


async def get_redis() -> redis.Redis:
    """Get Redis connection."""
    global _redis
    if _redis is None:
        _redis = redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
        )
    return _redis


class RedisStreamService:
    """Redis Streams producer/consumer."""

    def __init__(self, stream_name: str, group_name: str = "workers"):
        self.stream_name = stream_name
        self.group_name = group_name

    async def setup(self):
        """Create consumer group if not exists."""
        client = await get_redis()
        try:
            await client.xgroup_create(
                self.stream_name,
                self.group_name,
                id="0",
                mkstream=True,
            )
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    async def publish(self, data: dict[str, Any]) -> str:
        """Add message to stream."""
        client = await get_redis()
        message_id = await client.xadd(
            self.stream_name,
            {"data": json.dumps(data)},
            maxlen=10000,  # Limit stream size
        )
        return message_id

    async def consume(
        self,
        consumer_name: str,
        count: int = 10,
        block: int = 5000,
    ) -> list[tuple[str, dict]]:
        """Read messages from stream."""
        client = await get_redis()

        # Read new messages
        messages = await client.xreadgroup(
            groupname=self.group_name,
            consumername=consumer_name,
            streams={self.stream_name: ">"},
            count=count,
            block=block,
        )

        result = []
        if messages:
            for stream, entries in messages:
                for message_id, data in entries:
                    result.append((message_id, json.loads(data["data"])))

        return result

    async def ack(self, message_id: str):
        """Acknowledge message processing."""
        client = await get_redis()
        await client.xack(self.stream_name, self.group_name, message_id)

    async def claim_pending(
        self,
        consumer_name: str,
        min_idle_time: int = 60000,
        count: int = 10,
    ) -> list[tuple[str, dict]]:
        """Claim pending messages from dead consumers."""
        client = await get_redis()

        # Get pending messages
        pending = await client.xpending_range(
            self.stream_name,
            self.group_name,
            min="-",
            max="+",
            count=count,
        )

        result = []
        for entry in pending:
            if entry["time_since_delivered"] >= min_idle_time:
                # Claim the message
                messages = await client.xclaim(
                    self.stream_name,
                    self.group_name,
                    consumer_name,
                    min_idle_time,
                    [entry["message_id"]],
                )
                for msg_id, data in messages:
                    result.append((msg_id, json.loads(data["data"])))

        return result
```

### Stream Producer

```python
# app/services/stream.py
"""
Redis Stream producer.
"""
from app.core.redis_stream import RedisStreamService

# Stream instances
email_stream = RedisStreamService("stream:email", "email-workers")
ai_stream = RedisStreamService("stream:ai", "ai-workers")


async def init_streams():
    """Initialize all streams."""
    await email_stream.setup()
    await ai_stream.setup()


async def publish_email_task(to: str, subject: str, body: str) -> str:
    """Publish email task to stream."""
    return await email_stream.publish({
        "to": to,
        "subject": subject,
        "body": body,
    })


async def publish_ai_task(task_id: str, prompt: str, model: str) -> str:
    """Publish AI task to stream."""
    return await ai_stream.publish({
        "task_id": task_id,
        "prompt": prompt,
        "model": model,
    })
```

## Message Patterns

### Request-Reply Pattern

```python
# app/services/rpc.py
"""
RPC pattern over RabbitMQ.
"""
import asyncio
import json
import uuid

import aio_pika
from aio_pika import Message

from app.core.rabbitmq import get_rabbitmq_channel


class RPCClient:
    """RPC client for request-reply pattern."""

    def __init__(self):
        self.futures: dict[str, asyncio.Future] = {}
        self.callback_queue = None

    async def setup(self):
        """Setup callback queue."""
        channel = await get_rabbitmq_channel()

        # Exclusive callback queue
        self.callback_queue = await channel.declare_queue(
            "",
            exclusive=True,
            auto_delete=True,
        )

        await self.callback_queue.consume(self._on_response)

    async def _on_response(self, message: aio_pika.IncomingMessage):
        """Handle RPC response."""
        async with message.process():
            correlation_id = message.correlation_id
            if correlation_id in self.futures:
                future = self.futures.pop(correlation_id)
                future.set_result(json.loads(message.body))

    async def call(
        self,
        routing_key: str,
        payload: dict,
        timeout: float = 30.0,
    ) -> dict:
        """Make RPC call and wait for response."""
        channel = await get_rabbitmq_channel()
        exchange = await channel.get_exchange("tasks")

        correlation_id = str(uuid.uuid4())
        future = asyncio.Future()
        self.futures[correlation_id] = future

        message = Message(
            body=json.dumps(payload).encode(),
            correlation_id=correlation_id,
            reply_to=self.callback_queue.name,
        )

        await exchange.publish(message, routing_key=routing_key)

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self.futures.pop(correlation_id, None)
            raise
```

### Pub/Sub Pattern

```python
# app/services/pubsub.py
"""
Pub/Sub pattern for events.
"""
import json
from typing import Callable, Awaitable

import aio_pika

from app.core.rabbitmq import get_rabbitmq_channel, EXCHANGE_EVENTS


async def publish_event(event_type: str, data: dict):
    """Publish event to all subscribers."""
    channel = await get_rabbitmq_channel()
    exchange = await channel.get_exchange(EXCHANGE_EVENTS)

    message = aio_pika.Message(
        body=json.dumps({
            "type": event_type,
            "data": data,
        }).encode(),
        content_type="application/json",
    )

    await exchange.publish(message, routing_key="")


async def subscribe_events(
    handler: Callable[[str, dict], Awaitable[None]],
):
    """Subscribe to all events."""
    channel = await get_rabbitmq_channel()
    exchange = await channel.get_exchange(EXCHANGE_EVENTS)

    # Exclusive queue for this subscriber
    queue = await channel.declare_queue("", exclusive=True)
    await queue.bind(exchange)

    async def process_message(message: aio_pika.IncomingMessage):
        async with message.process():
            payload = json.loads(message.body)
            await handler(payload["type"], payload["data"])

    await queue.consume(process_message)


# Usage
async def on_event(event_type: str, data: dict):
    if event_type == "user.created":
        print(f"New user: {data['email']}")

await subscribe_events(on_event)
await publish_event("user.created", {"email": "user@example.com"})
```

## FastAPI Integration

```python
# app/main.py
"""
FastAPI with message queue lifecycle.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.rabbitmq import setup_rabbitmq, close_rabbitmq
from app.services.stream import init_streams


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan."""
    # Startup
    await setup_rabbitmq()
    await init_streams()
    yield
    # Shutdown
    await close_rabbitmq()


app = FastAPI(lifespan=lifespan)
```

## Message Queue Checklist

- [ ] Connection pooling/robust connections
- [ ] Queue durability (durable=True)
- [ ] Message persistence (delivery_mode=PERSISTENT)
- [ ] Dead letter queues for failed messages
- [ ] Message TTL configured
- [ ] Consumer acknowledgment (manual ack)
- [ ] Prefetch limit set (QoS)
- [ ] Retry logic with backoff
- [ ] Graceful shutdown
- [ ] Health checks for queue connectivity
