---
name: verification-before-completion
description: Use when about to claim work is complete, fixed, or passing, before committing or creating PRs - requires running verification commands and confirming output before making any success claims; evidence before assertions always
---

# Verification Before Completion

## Overview

Claiming work is complete without verification is dishonesty, not efficiency.

**Core principle:** Evidence before claims, always.

**Violating the letter of this rule is violating the spirit of this rule.**

## The Iron Law

```
NO COMPLETION CLAIMS WITHOUT FRESH VERIFICATION EVIDENCE
```

If you haven't run the verification command in this message, you cannot claim it passes.

## The Gate Function

```
BEFORE claiming any status or expressing satisfaction:

1. IDENTIFY: What command proves this claim?
2. RUN: Execute the FULL command (fresh, complete)
3. READ: Full output, check exit code, count failures
4. VERIFY: Does output confirm the claim?
   - If NO: State actual status with evidence
   - If YES: State claim WITH evidence
5. ONLY THEN: Make the claim

Skip any step = lying, not verifying
```

## Common Failures

| Claim | Requires | Not Sufficient |
|-------|----------|----------------|
| Tests pass | Test command output: 0 failures | Previous run, "should pass" |
| Linter clean | Linter output: 0 errors | Partial check, extrapolation |
| Build succeeds | Build command: exit 0 | Linter passing, logs look good |
| Bug fixed | Test original symptom: passes | Code changed, assumed fixed |
| Regression test works | Red-green cycle verified | Test passes once |
| Agent completed | VCS diff shows changes | Agent reports "success" |
| Requirements met | Line-by-line checklist | Tests passing |

## Red Flags - STOP

- Using "should", "probably", "seems to"
- Expressing satisfaction before verification ("Great!", "Perfect!", "Done!", etc.)
- About to commit/push/PR without verification
- Trusting agent success reports
- Relying on partial verification
- Thinking "just this once"
- Tired and wanting work over
- **ANY wording implying success without having run verification**

## Rationalization Prevention

| Excuse | Reality |
|--------|---------|
| "Should work now" | RUN the verification |
| "I'm confident" | Confidence ≠ evidence |
| "Just this once" | No exceptions |
| "Linter passed" | Linter ≠ compiler |
| "Agent said success" | Verify independently |
| "I'm tired" | Exhaustion ≠ excuse |
| "Partial check is enough" | Partial proves nothing |
| "Different words so rule doesn't apply" | Spirit over letter |

## Key Patterns

**Tests:**
```
✅ [Run test command] [See: 34/34 pass] "All tests pass"
❌ "Should pass now" / "Looks correct"
```

**Regression tests (TDD Red-Green):**
```
✅ Write → Run (pass) → Revert fix → Run (MUST FAIL) → Restore → Run (pass)
❌ "I've written a regression test" (without red-green verification)
```

**Build:**
```
✅ [Run build] [See: exit 0] "Build passes"
❌ "Linter passed" (linter doesn't check compilation)
```

**Requirements:**
```
✅ Re-read plan → Create checklist → Verify each → Report gaps or completion
❌ "Tests pass, phase complete"
```

**Agent delegation:**
```
✅ Agent reports success → Check VCS diff → Verify changes → Report actual state
❌ Trust agent report
```

## Python/FastAPI Verification Commands

Use these specific commands for this project:

**Tests (pytest):**
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=term-missing

# Run specific test file
pytest tests/integration/test_users_api.py -v

# Run single test
pytest tests/unit/test_user_service.py::TestUserService::test_create_user -v
```
```
✅ pytest output: "34 passed in 2.5s" → "All 34 tests pass"
❌ "Tests should pass" without running pytest
```

**Linter (ruff):**
```bash
# Check for errors
ruff check .

# Check and auto-fix
ruff check . --fix

# Format check
ruff format --check .

# Format and fix
ruff format .
```
```
✅ ruff check: "All checks passed!" → "Linter clean"
❌ "Code looks fine" without running ruff
```

**Database migrations (alembic):**
```bash
# Check current revision
alembic current

# Check pending migrations
alembic history --verbose

# Apply migrations
alembic upgrade head

# Verify migration applied
alembic current
```
```
✅ alembic upgrade head: "Running upgrade..." + alembic current shows latest → "Migration applied"
❌ "Migration should work" without running alembic
```

**Server startup:**
```bash
# Start dev server
uvicorn app.main:app --reload --port 8000

# Check health endpoint
curl http://localhost:8000/health
```
```
✅ curl /health: {"status": "healthy"} → "Server running"
❌ "Server should start" without testing endpoint
```

**Type checking (optional):**
```bash
# If using mypy
mypy app/
```

## Full Verification Sequence

Before claiming "implementation complete":

```bash
# 1. Lint check
ruff check . && ruff format --check .

# 2. Run tests
pytest --cov=app

# 3. Apply migrations (if any)
alembic upgrade head

# 4. Start server and test
uvicorn app.main:app --port 8000 &
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/users  # Test specific endpoint
```

Only after ALL commands succeed with expected output → claim completion.

## Why This Matters

From 24 failure memories:
- your human partner said "I don't believe you" - trust broken
- Undefined functions shipped - would crash
- Missing requirements shipped - incomplete features
- Time wasted on false completion → redirect → rework
- Violates: "Honesty is a core value. If you lie, you'll be replaced."

## When To Apply

**ALWAYS before:**
- ANY variation of success/completion claims
- ANY expression of satisfaction
- ANY positive statement about work state
- Committing, PR creation, task completion
- Moving to next task
- Delegating to agents

**Rule applies to:**
- Exact phrases
- Paraphrases and synonyms
- Implications of success
- ANY communication suggesting completion/correctness

## The Bottom Line

**No shortcuts for verification.**

Run the command. Read the output. THEN claim the result.

This is non-negotiable.