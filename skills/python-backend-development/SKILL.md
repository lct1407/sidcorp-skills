---
name: python-backend-development
description: Generate Python FastAPI code following project design patterns. Use when creating models, schemas, repositories, services, controllers, database migrations, authentication, or tests. Enforces layered architecture, async patterns, OWASP security, and Alembic migration naming conventions (yyyymmdd_HHmm_feature).
allowed-tools:
  - Read
  - Write
  - Edit
  - Glob
  - Grep
  - Bash
---

# Python Backend Development Standards

## Architecture Overview

```
Router/Controller → Service → Repository → Database
      ↓                ↓            ↓
   Schemas         Business      SQLAlchemy
  (Pydantic)        Logic         Models
```

## Layer Responsibilities

| Layer | Location | Purpose |
|-------|----------|---------|
| **Models** | `app/models/` | SQLAlchemy ORM, database schema |
| **Schemas** | `app/schemas/` | Pydantic DTOs (request/response) |
| **Repositories** | `app/repositories/` | Database CRUD operations |
| **Services** | `app/services/` | Business logic orchestration |
| **Controllers** | `app/api/v1/` | FastAPI routes, thin handlers |
| **Migrations** | `alembic/versions/` | Database migrations |

## Naming Conventions

### Files

- All lowercase with underscores: `user_profile.py`
- One entity per file
- Match filename to class: `user.py` → `class User`

### Classes

- Models: `User`, `BlogPost` (PascalCase, singular)
- Schemas: `UserCreate`, `UserResponse`, `UserUpdate`
- Repositories: `UserRepository`
- Services: `UserService`

### Database

- Table names: plural snake_case (`users`, `blog_posts`)
- Column names: snake_case (`created_at`, `user_id`)

## Alembic Migrations

### File Naming Convention

```
yyyymmdd_HHmm_<feature>.py
```

**Examples:**

- `20260117_0930_create_users_table.py`
- `20260117_1045_add_email_to_users.py`
- `20260117_1400_create_api_keys_table.py`

### Create Migration Command

```bash
# Generate with autogenerate
alembic revision --autogenerate -m "description"

# Then rename the file to follow convention:
# FROM: xxxx_description.py
# TO:   yyyymmdd_HHmm_description.py
```

## Code Standards

### Async Everything

- All database operations must be async
- Use `async def` for all handlers, services, repositories
- Use `await` for all I/O operations

### Dependency Injection

- Use FastAPI `Depends()` for dependencies
- Inject database sessions into services
- Services inject repositories

### Error Handling

- Use custom exceptions in `app/core/exceptions.py`
- Let FastAPI exception handlers convert to HTTP responses
- Never catch and swallow exceptions silently

### Security

- Argon2id for password hashing
- Parameterized queries (SQLAlchemy ORM)
- Input validation (Pydantic)
- Rate limiting on auth endpoints

## Reference Navigation

**Core Patterns:**

- [python-models.md](references/python-models.md) - SQLAlchemy ORM (User, ApiKey)
- [python-schemas.md](references/python-schemas.md) - Pydantic v2 patterns
- [python-repositories.md](references/python-repositories.md) - Repository pattern
- [python-services.md](references/python-services.md) - Service layer patterns
- [python-api-design.md](references/python-api-design.md) - FastAPI routes and controllers
- [python-migrations.md](references/python-migrations.md) - Alembic migration patterns
- [python-utils.md](references/python-utils.md) - Utility functions (datetime, string, validation)

**Security & Auth:**

- [python-authentication.md](references/python-authentication.md) - JWT, API Keys, OAuth
- [python-middleware.md](references/python-middleware.md) - Dual auth, rate limiting, credits
- [python-security.md](references/python-security.md) - OWASP Top 10, input validation

**Quality & Operations:**

- [python-testing.md](references/python-testing.md) - pytest patterns, fixtures
- [python-performance.md](references/python-performance.md) - Caching, query optimization, async
- [python-debugging.md](references/python-debugging.md) - Logging, profiling, error tracking
- [python-devops.md](references/python-devops.md) - Docker, CI/CD, deployment
