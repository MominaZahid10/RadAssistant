"""
RadAssist AI — Auth endpoints (Phase 6)

    POST /api/v1/auth/register  create an account (gated)
    POST /api/v1/auth/login     email + password  →  bearer token
    GET  /api/v1/auth/me        the signed-in user

⚠️  REGISTRATION IS SAFE BECAUSE OF OWNERSHIP, NOT INSTEAD OF IT.

Open signup was rejected earlier on the grounds that anyone who found the URL
could read uploaded patient reports. That was true while every signed-in user
saw every report. Migration 0006 gave reports and images an owner, so a new
account now lands in an empty workspace — and the objection stopped applying.

The two are coupled: if ownership were ever reverted, ALLOW_REGISTRATION would
have to go false in the same change.

    ALLOW_REGISTRATION=true   demo, portfolio, reviewer — anyone may sign up
    ALLOW_REGISTRATION=false  clinical deployment — operator creates accounts
                              with scripts/create_user.py

Administrators are never created here, only by that script.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.database import get_db
from app.core.deps import get_current_user
from app.core.limits import LOGIN, REGISTER, per_ip
from app.core.security import (
    AuthError,
    create_access_token,
    dummy_verify,
    hash_password,
    verify_password,
)
from app.models.user import User
from app.schemas.auth import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)

settings = get_settings()
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an account",
    # ⚠️  PER-IP, AND THIS IS WHAT STOPS ACCOUNT ENUMERATION.
    # Registration has to reveal whether an email is taken — a signup form
    # cannot avoid it without leaving the user unable to sign in. 5/hour makes
    # checking a staff directory take weeks rather than minutes.
    dependencies=[Depends(per_ip(REGISTER, "register"))],
    description=(
        "Self-service signup, returning a token so the caller is signed in "
        "immediately.\n\n"
        "Disabled by setting `ALLOW_REGISTRATION=false`, which a clinical "
        "deployment should do — there, accounts are created by an operator "
        "running `scripts/create_user.py`.\n\n"
        "New accounts are never administrators."
    ),
)
async def register(payload: RegisterRequest, db: AsyncSession = Depends(get_db)):
    if not settings.ALLOW_REGISTRATION:
        # 403 rather than 404: the route genuinely exists and is switched off
        # deliberately. Pretending it is absent would send whoever deployed
        # this looking for a bug.
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Registration is disabled on this deployment. Ask an "
                "administrator to create your account."
            ),
        )

    email = User.normalise_email(payload.email)

    existing = (await db.execute(
        select(User).where(User.email == email)
    )).scalar_one_or_none()

    if existing:
        # ⚠️  THIS RESPONSE LEAKS THAT THE ACCOUNT EXISTS, AND THERE IS NO
        # WAY AROUND IT.
        # Login goes to some trouble not to reveal which addresses are
        # registered. A registration form cannot preserve that: it either
        # tells you the address is taken, or it silently does nothing and the
        # user is left unable to sign in with no explanation.
        #
        # The honest trade is to accept the disclosure here, where it is
        # unavoidable, rather than break the form to protect a property that
        # signup inherently gives away. Rate limiting (Step 4) is what stops
        # it becoming a bulk enumeration tool.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with that email already exists.",
        )

    try:
        hashed = hash_password(payload.password)
    except AuthError as e:
        raise HTTPException(status_code=422, detail=str(e))

    user = User(
        email=email,
        hashed_password=hashed,
        full_name=payload.full_name,
        is_active=True,
        # ⚠️  NEVER FROM THE PAYLOAD. There is no such field, and adding one
        # would make this endpoint a privilege escalation with a form on it.
        is_admin=False,
    )
    db.add(user)

    try:
        await db.commit()
    except IntegrityError:
        # Two simultaneous signups for the same address both passed the
        # existence check above. The unique constraint is what actually
        # decides, which is why it exists in the database and not only here.
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with that email already exists.",
        )

    await db.refresh(user)

    try:
        token = create_access_token(str(user.id))
    except AuthError as e:
        logger.error("Cannot issue token after registration: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured on this server.",
        )

    user.last_login_at = datetime.now(timezone.utc)
    await db.commit()

    logger.info("New account registered: %s", user.id)
    return TokenResponse(
        access_token=token,
        expires_in=settings.JWT_EXPIRE_MINUTES * 60,
    )


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Sign in and receive a bearer token",
    # ⚠️  BY IP, NOT BY ACCOUNT, DELIBERATELY.
    # Locking an account after N failures lets anyone lock a colleague out by
    # guessing wrong at their address — denial of service with a login form.
    # Limiting the source costs the attacker, not the victim.
    dependencies=[Depends(per_ip(LOGIN, "login"))],
    description=(
        "Exchange an email and password for a short-lived access token.\n\n"
        "Send it on every subsequent request as `Authorization: Bearer <token>`."
    ),
)
async def login(payload: LoginRequest, db: AsyncSession = Depends(get_db)):
    email = User.normalise_email(payload.email)

    user = (await db.execute(
        select(User).where(User.email == email)
    )).scalar_one_or_none()

    # ⚠️  ONE FAILURE PATH, ONE MESSAGE, ONE DURATION.
    #
    # Returning early when the email is unknown skips the bcrypt comparison
    # and answers in microseconds, while a real account takes ~100ms. That gap
    # is measurable over the network and turns this form into an account
    # enumerator: an attacker learns which of a hospital's addresses are
    # registered without ever guessing a password.
    #
    # So an unknown email still burns a hash comparison, and both cases return
    # exactly the same 401. "No such user" and "wrong password" are the same
    # answer as far as the caller is concerned.
    if user is None:
        dummy_verify(payload.password)
        raise _bad_credentials()

    if not verify_password(payload.password, user.hashed_password):
        logger.info("Failed login for existing account %s", user.id)
        raise _bad_credentials()

    if not user.is_active:
        # Also indistinguishable from wrong credentials. A distinct "account
        # disabled" response confirms the account exists.
        raise _bad_credentials()

    try:
        token = create_access_token(str(user.id))
    except AuthError as e:
        # JWT_SECRET missing or unusable. This is a deployment fault, not a
        # user fault, and it must not read as "wrong password".
        logger.error("Cannot issue token: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured on this server.",
        )

    user.last_login_at = datetime.now(timezone.utc)
    await db.commit()

    return TokenResponse(
        access_token=token,
        expires_in=settings.JWT_EXPIRE_MINUTES * 60,
    )


def _bad_credentials() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect email or password.",
        headers={"WWW-Authenticate": "Bearer"},
    )


@router.get(
    "/me",
    response_model=UserResponse,
    summary="The signed-in user",
    description="Confirms a token is valid and returns who it belongs to.",
)
async def me(user: User = Depends(get_current_user)):
    return UserResponse.model_validate(user)
