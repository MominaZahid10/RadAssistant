"""
RadAssist AI — Auth dependencies (Phase 6)

`get_current_user` is the gate. Routes that need a signed-in user declare it;
routes that do not are on a short, deliberate public list.

⚠️  DEFAULT DENY.
Protection is applied per router with an explicit public allowlist, rather
than per route by hand. Adding a route to the allowlist is a decision someone
makes and can be reviewed; forgetting to protect a new route is an accident —
and the second is the failure that actually happens, months later, when
somebody adds an endpoint in a hurry.

tests/test_authz.py enumerates every registered route and fails if one is
neither protected nor listed.
"""

from __future__ import annotations

import uuid

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import AuthError, decode_access_token
from app.models.user import User

# auto_error=False so a MISSING token and an INVALID token both come through
# here and get the same 401 with the same body. Letting FastAPI raise its own
# error for one and ours for the other lets a caller distinguish "no token"
# from "bad token", which is a small oracle but a free one.
_bearer = HTTPBearer(auto_error=False)


def _unauthorised() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated.",
        # Required by RFC 6750 so clients know how to authenticate.
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Resolve the signed-in user, or reject.

    ⚠️  THE DATABASE IS CHECKED ON EVERY REQUEST, NOT JUST THE SIGNATURE.
    A valid signature only proves the token was issued by us. It says nothing
    about whether the account still exists or is still active. Without this
    lookup, deactivating a departed clinician would leave their token working
    until it expired — up to twelve hours of access to patient reports after
    someone believed they had revoked it.
    """
    if credentials is None or not credentials.credentials:
        raise _unauthorised()

    try:
        claims = decode_access_token(credentials.credentials)
    except AuthError:
        raise _unauthorised()

    try:
        user_id = uuid.UUID(str(claims.get("sub")))
    except (ValueError, TypeError):
        raise _unauthorised()

    user = (await db.execute(
        select(User).where(User.id == user_id)
    )).scalar_one_or_none()

    if user is None or not user.is_active:
        # Same response as a bad token. A distinct "account disabled" message
        # would confirm the account exists to whoever holds a stale token.
        raise _unauthorised()

    return user


async def get_current_admin(user: User = Depends(get_current_user)) -> User:
    """For operations only an operator should perform."""
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This action requires an administrator account.",
        )
    return user
