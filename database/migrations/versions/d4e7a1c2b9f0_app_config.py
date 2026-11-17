"""app_config: Dashi accounts and dashboard settings stored in the database

Revision ID: d4e7a1c2b9f0
Revises: b1b42b2ec823
Create Date: 2026-11-17 12:00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'd4e7a1c2b9f0'
down_revision: Union[str, Sequence[str], None] = 'b1b42b2ec823'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('app_config',
    sa.Column('key', sa.String(length=64), nullable=False),
    sa.Column('value', sa.JSON(), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('key')
    )


def downgrade() -> None:
    op.drop_table('app_config')
