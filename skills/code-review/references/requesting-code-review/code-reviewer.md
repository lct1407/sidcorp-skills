# Code Reviewer Subagent Template

Use this template when dispatching `code-reviewer` subagent via Task tool.

## Template

```
Review implementation: {DESCRIPTION}

## Context

**What was implemented:**
{WHAT_WAS_IMPLEMENTED}

**Plan/Requirements:**
{PLAN_OR_REQUIREMENTS}

**Git range:**
- Base: {BASE_SHA}
- Head: {HEAD_SHA}

## Review Focus

1. Does implementation match requirements?
2. Are there any bugs or edge cases missed?
3. Code quality (readability, maintainability)
4. Security concerns
5. Performance issues
6. Test coverage adequate?

## Expected Output

Provide:
- **Strengths**: What was done well
- **Issues**: Categorized as Critical/Important/Minor
- **Assessment**: Ready to proceed / Needs fixes first
```

## Python/FastAPI Specific Checklist

When reviewing Python FastAPI code, also check:

### Architecture
- [ ] Follows layered architecture (Router → Service → Repository)
- [ ] Async patterns used correctly (async def, await)
- [ ] Dependency injection via FastAPI Depends()
- [ ] One repository per model, one service per domain

### Code Quality
- [ ] Type hints on all functions
- [ ] Pydantic schemas for request/response
- [ ] No raw SQL (use SQLAlchemy ORM)
- [ ] Error handling via custom exceptions

### Security (OWASP)
- [ ] Input validation via Pydantic
- [ ] No hardcoded secrets
- [ ] Ownership checks on resources
- [ ] Rate limiting on sensitive endpoints

### Database
- [ ] Async session handling
- [ ] Proper eager loading (avoid N+1)
- [ ] Alembic migration follows naming: `yyyymmdd_HHmm_<feature>.py`

### Testing
- [ ] Unit tests for services
- [ ] Integration tests for API endpoints
- [ ] Fixtures used for common setup

## Example Usage

```bash
# Get git SHAs
BASE_SHA=$(git rev-parse HEAD~3)
HEAD_SHA=$(git rev-parse HEAD)
```

Then dispatch via Task tool:

```
Review implementation: User authentication endpoints

## Context

**What was implemented:**
- POST /api/v1/auth/login - JWT login
- POST /api/v1/auth/refresh - Token refresh
- POST /api/v1/auth/logout - Token blacklist
- UserService with password verification
- AuthRepository for token storage

**Plan/Requirements:**
- JWT access + refresh token flow
- Argon2id password hashing
- Token blacklist on logout
- Rate limiting: 5 attempts/minute

**Git range:**
- Base: a1b2c3d
- Head: e4f5g6h

## Review Focus

1. Does implementation match requirements?
2. Are there any bugs or edge cases missed?
3. Code quality (readability, maintainability)
4. Security concerns
5. Performance issues
6. Test coverage adequate?
```
