"""
RadAssist AI — User Model (Phase 6, Step 1)

Accounts for the pilot deployment.

════════════════════════════════════════════════════════════════════
⚠️  ACCOUNTS ARE CREATED BY AN OPERATOR, NOT BY VISITORS
════════════════════════════════════════════════════════════════════
There is no registration endpoint. A public `/auth/register` on a clinical
tool means anyone who finds the URL can create an account and read uploaded
patient reports — which is arguably worse than no authentication, because the
login screen then performs reassurance rather than access control.

    docker-compose exec backend python scripts/create_user.py \
        --email radiologist@hospital.org

The pilot has known users by name. A registration form solves a problem this
deployment does not have.

════════════════════════════════════════════════════════════════════
WHAT IS DELIBERATELY NOT STORED
════════════════════════════════════════════════════════════════════
No plaintext password, ever, in any column, log line or error message. The
hash is bcrypt with the library's default cost, so it rises with hardware
rather than being pinned to what was fast in 2026.

No password reset token, no security questions, no recovery email. Each is a
credential path that has to be secured as carefully as the password itself,
and for a handful of operator-created accounts the recovery procedure is
"the operator runs the script again".
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, String
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base


class User(Base):
    """A person who may sign in."""

    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)

    # ⚠️  STORED LOWERCASE, ALWAYS.
    # Email is case-insensitive in practice, and treating it otherwise means
    # Alice@x.org and alice@x.org become two accounts — a duplicate identity
    # that splits a person's reports across them. Normalised on write and on
    # lookup rather than relying on a citext extension the host may not have.
    email = Column(String(320), unique=True, nullable=False, index=True)

    # bcrypt output. Never the password.
    hashed_password = Column(String(255), nullable=False)

    full_name = Column(String(200), nullable=True)

    # ⚠️  DEACTIVATE RATHER THAN DELETE.
    # Deleting a user would orphan or cascade their reports, and the sign-off
    # trail exists precisely so an approval can be attributed later. A
    # departed clinician's approvals must remain attributable to them.
    is_active = Column(Boolean, nullable=False, default=True)

    # Operator accounts that may create other users. Not a permission system —
    # one boolean, because a pilot does not need roles and inventing them now
    # would mean guessing at a hierarchy nobody has described.
    is_admin = Column(Boolean, nullable=False, default=False)

    last_login_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    @staticmethod
    def normalise_email(email: str) -> str:
        """Lowercase and strip. Used on write AND on lookup, or the unique
        constraint protects nothing."""
        return (email or "").strip().lower()

    def __repr__(self) -> str:
        # No hash, no name — repr lands in logs and exception output.
        return f"<User {self.id} active={self.is_active}>"
