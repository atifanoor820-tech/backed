"""users and sign-in sessions; conversations get an owner

Revision ID: e2a9c4d7f1b3
Revises: d4e7a1c2b9f0
Create Date: 2026-11-17 18:00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'e2a9c4d7f1b3'
down_revision: Union[str, Sequence[str], None] = 'd4e7a1c2b9f0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('users',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('email', sa.String(length=254), nullable=False),
    sa.Column('name', sa.String(length=100), nullable=False),
    sa.Column('password_hash', sa.String(length=255), nullable=False),
    sa.Column('role', sa.String(length=16), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_users_email', 'users', ['email'], unique=True)
    op.create_table('auth_sessions',
    sa.Column('token_hash', sa.String(length=64), nullable=False),
    sa.Column('user_id', sa.String(length=36), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('token_hash')
    )
    op.create_index('ix_auth_sessions_user_id', 'auth_sessions', ['user_id'])
    with op.batch_alter_table('conversations') as b:
        b.add_column(sa.Column('user_id', sa.String(length=36), nullable=True))
        b.create_index('ix_conversations_user_id', ['user_id'])
        b.create_foreign_key('fk_conversations_user_id', 'users', ['user_id'], ['id'], ondelete='CASCADE')


def downgrade() -> None:
    with op.batch_alter_table('conversations') as b:
        b.drop_constraint('fk_conversations_user_id', type_='foreignkey')
        b.drop_index('ix_conversations_user_id')
        b.drop_column('user_id')
    op.drop_index('ix_auth_sessions_user_id', table_name='auth_sessions')
    op.drop_table('auth_sessions')
    op.drop_index('ix_users_email', table_name='users')
    op.drop_table('users')
