# Alembic Migrations

## File Naming Convention

```
yyyymmdd_HHmm_<feature>.py
```

**Examples:**

- `20260117_0930_create_users_table.py`
- `20260117_1045_add_email_index_to_users.py`
- `20260117_1400_create_api_keys_table.py`

## Commands

```bash
# Apply all pending migrations
alembic upgrade head

# Rollback one migration
alembic downgrade -1

# Rollback to specific revision
alembic downgrade <revision_id>

# Create new migration (autogenerate from models)
alembic revision --autogenerate -m "create_users_table"

# Create empty migration
alembic revision -m "add_custom_data"

# Show current revision
alembic current

# Show migration history
alembic history --verbose
```

## After Creating Migration

Rename the generated file to follow naming convention:

```bash
# FROM: abc123_create_users_table.py
# TO:   20260117_0930_create_users_table.py
```

## Migration Examples

### Create Users Table

```python
# alembic/versions/20260117_0930_create_users_table.py
"""Create users table

Revision ID: 20260117_0930
Revises:
Create Date: 2026-01-17 09:30:00
"""
from alembic import op
import sqlalchemy as sa

revision = '20260117_0930'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('email', sa.String(255), nullable=False),
        sa.Column('username', sa.String(50), nullable=False),
        sa.Column('hashed_password', sa.String(255), nullable=False),
        sa.Column('full_name', sa.String(255), nullable=True),
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.Column('is_superuser', sa.Boolean(), default=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_users_email', 'users', ['email'], unique=True)
    op.create_index('ix_users_username', 'users', ['username'], unique=True)


def downgrade() -> None:
    op.drop_index('ix_users_username', table_name='users')
    op.drop_index('ix_users_email', table_name='users')
    op.drop_table('users')
```

### Create API Keys Table

```python
# alembic/versions/20260117_1000_create_api_keys_table.py
"""Create api_keys table

Revision ID: 20260117_1000
Revises: 20260117_0930
Create Date: 2026-01-17 10:00:00
"""
from alembic import op
import sqlalchemy as sa

revision = '20260117_1000'
down_revision = '20260117_0930'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'api_keys',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('description', sa.String(500), nullable=True),
        sa.Column('key_hash', sa.String(64), nullable=False),
        sa.Column('key_prefix', sa.String(12), nullable=False),
        sa.Column('scopes', sa.JSON(), default=[]),
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.Column('last_used_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    )
    op.create_index('ix_api_keys_user_id', 'api_keys', ['user_id'])
    op.create_index('ix_api_keys_key_hash', 'api_keys', ['key_hash'], unique=True)


def downgrade() -> None:
    op.drop_index('ix_api_keys_key_hash', table_name='api_keys')
    op.drop_index('ix_api_keys_user_id', table_name='api_keys')
    op.drop_table('api_keys')
```

### Add Column

```python
# alembic/versions/20260117_1100_add_avatar_to_users.py
"""Add avatar column to users

Revision ID: 20260117_1100
Revises: 20260117_1000
Create Date: 2026-01-17 11:00:00
"""
from alembic import op
import sqlalchemy as sa

revision = '20260117_1100'
down_revision = '20260117_1000'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('users', sa.Column('avatar_url', sa.String(500), nullable=True))


def downgrade() -> None:
    op.drop_column('users', 'avatar_url')
```

### Add Index

```python
# alembic/versions/20260117_1130_add_index_users_active.py
"""Add index on users is_active

Revision ID: 20260117_1130
Revises: 20260117_1100
Create Date: 2026-01-17 11:30:00
"""
from alembic import op

revision = '20260117_1130'
down_revision = '20260117_1100'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index('ix_users_is_active', 'users', ['is_active'])


def downgrade() -> None:
    op.drop_index('ix_users_is_active', table_name='users')
```

### Data Migration

```python
# alembic/versions/20260117_1200_migrate_user_data.py
"""Migrate user data

Revision ID: 20260117_1200
Revises: 20260117_1130
Create Date: 2026-01-17 12:00:00
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.sql import table, column

revision = '20260117_1200'
down_revision = '20260117_1130'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Define table for data operations
    users = table(
        'users',
        column('id', sa.Integer),
        column('full_name', sa.String),
        column('first_name', sa.String),
        column('last_name', sa.String),
    )

    # Update data
    connection = op.get_bind()
    connection.execute(
        users.update().values(
            full_name=users.c.first_name + ' ' + users.c.last_name
        )
    )


def downgrade() -> None:
    pass  # Data migration cannot be reversed
```

## Alembic Configuration

### alembic.ini

```ini
[alembic]
script_location = alembic
prepend_sys_path = .
version_path_separator = os

sqlalchemy.url = postgresql+asyncpg://user:pass@localhost:5432/dbname

[post_write_hooks]
hooks = ruff
ruff.type = exec
ruff.executable = ruff
ruff.options = format REVISION_SCRIPT_FILENAME
```

### alembic/env.py

```python
import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

from app.db.base import Base
from app.models import *  # Import all models

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

## Migration Rules

1. **Always test** - Run upgrade and downgrade locally before committing
2. **Atomic migrations** - One logical change per migration
3. **Reversible** - Always implement downgrade (except data migrations)
4. **No model imports** - Use raw SQL or `sa.table()` for data migrations
5. **Naming** - Use `yyyymmdd_HHmm_<feature>.py` format
