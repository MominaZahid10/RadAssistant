"""
RadAssist AI — Password hashing and JWT (Phase 6, Step 1)

════════════════════════════════════════════════════════════════════
⚠️  JWT_SECRET HAS NO DEFAULT, ON PURPOSE
════════════════════════════════════════════════════════════════════
A signing key with a fallback value is worse than no key at all. It works in
development, ships to production unnoticed, and every deployment that ever
copied this repository shares it — so anyone holding the default can mint a
valid token for any user on any instance.

So the app refuses to start without one, loudly, the same way it already
refuses to start without an embedding model. A boot failure with a clear
message costs minutes. A shared signing key costs everything the token
protects, silently, for as long as nobody notices.

    python -c "import secrets; print(secrets.token_urlsafe(48))"

════════════════════════════════════════════════════════════════════
WHY bcrypt AND NOT A HAND-ROLLED HASH
════════════════════════════════════════════════════════════════════
SHA-256 of a password is not password storage. It is fast, which is exactly
the property an attacker with the database wants — modern hardware tries
billions of SHA-256 candidates per second. bcrypt is deliberately slow and
salted per password, so a stolen table cannot be attacked in bulk and two
users with the same password do not share a hash.

The cost factor is left at the library default so it rises with hardware
rather than being pinned to whatever was comfortable when this was written.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


class AuthError(Exception):
    """A token could not be issued or verified."""


# ══════════════════════════════════════════════════════════════
# SECRET
# ══════════════════════════════════════════════════════════════


def _require_secret() -> str:
    """
    The signing key, or a refusal to continue.

    Checked on every use rather than only at import, so a misconfigured
    deployment cannot get past startup with the key missing and then fail at
    the first login — by which time the failure looks like a login bug.
    """
    secret = (settings.JWT_SECRET or "").strip()

    if not secret:
        raise AuthError(
            "JWT_SECRET is not set. Authentication cannot run without a "
            "signing key, and this deliberately has no default — a fallback "
            "value would mean every deployment shares one.\n"
            "Generate one:\n"
            '    python -c "import secrets; print(secrets.token_urlsafe(48))"\n'
            "Then add it to backend/.env as JWT_SECRET=..."
        )

    # A short key can be brute-forced offline from a single captured token,
    # after which anyone can mint credentials for any user.
    if len(secret) < 32:
        raise AuthError(
            f"JWT_SECRET is only {len(secret)} characters. Use at least 32 — "
            f"a short key can be recovered offline from one captured token."
        )

    # Catches the most likely copy-paste: the example value from a README.
    if secret.lower().startswith(("changeme", "your_", "secret", "example")):
        raise AuthError(
            "JWT_SECRET still looks like a placeholder. Generate a real one."
        )

    return secret


# ══════════════════════════════════════════════════════════════
# PASSWORDS
# ══════════════════════════════════════════════════════════════


def hash_password(password: str) -> str:
    """
    bcrypt hash. The plaintext is never returned, stored or logged.

    bcrypt truncates at 72 bytes, and silently — a 100-character passphrase
    would have its tail ignored, so two different long passwords could share a
    hash. Rejected rather than truncated: silently weakening a credential the
    user believed was strong is worse than telling them.
    """
    import bcrypt

    if not password:
        raise AuthError("Password must not be empty.")

    encoded = password.encode("utf-8")
    if len(encoded) > 72:
        raise AuthError(
            "Password is too long for bcrypt (72 bytes maximum). Beyond that "
            "the tail is ignored, which would make it weaker than it looks."
        )

    return bcrypt.hashpw(encoded, bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    """
    Check a password against a stored hash. Never raises on a bad input —
    a malformed hash in the database must read as "wrong password", not as a
    500 that tells the caller something about the stored value.
    """
    import bcrypt

    if not password or not hashed:
        return False
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


# A bcrypt hash of a value nobody uses. Compared against when the email does
# not exist, so a login attempt for an unknown account costs the same time as
# one for a known account.
#
# ⚠️  WITHOUT THIS, RESPONSE TIME LEAKS THE USER LIST.
# Returning early on "no such user" skips the hash comparison and answers in
# microseconds, while a real account takes ~100ms. That difference is
# measurable over the network and turns the login form into an account
# enumerator.
_DUMMY_HASH = "$2b$12$C6UzMDM.H6dfI/f/IKcEe.PjbGRnsUKlKu/aSU8lS4NLlwXzHIhWy"


def dummy_verify(password: str) -> None:
    """Burn the same time a real verification would. See _DUMMY_HASH."""
    verify_password(password or "x", _DUMMY_HASH)


# ══════════════════════════════════════════════════════════════
# TOKENS
# ══════════════════════════════════════════════════════════════

_ALGORITHM = "HS256"


def create_access_token(subject: str, extra: dict[str, Any] | None = None) -> str:
    """
    Sign a short-lived access token for `subject` (the user's UUID).

    ⚠️  THE SUBJECT IS AN ID, NOT AN EMAIL.
    Emails change. A token carrying an email would keep working against an
    address the user no longer has, and would stop working the moment they
    updated it — for the same account.

    No refresh token in this phase. A pilot with named users can sign in
    again; a refresh token is a second long-lived credential to store, rotate
    and revoke, and adding one without a revocation story is worse than not
    having it.
    """
    import jwt

    secret = _require_secret()
    now = datetime.now(timezone.utc)

    payload: dict[str, Any] = {
        "sub": str(subject),
        "iat": now,
        "exp": now + timedelta(minutes=settings.JWT_EXPIRE_MINUTES),
        # Marks this as an access token. If a refresh token is ever added,
        # this is what stops one being accepted where the other is expected.
        "typ": "access",
    }
    if extra:
        # ⚠️  NEVER PUT A SECRET IN HERE.
        # A JWT is signed, not encrypted. Anyone holding the token can read
        # every claim by base64-decoding the middle segment.
        payload.update(extra)

    return jwt.encode(payload, secret, algorithm=_ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any]:
    """
    Verify signature and expiry, and return the claims.

    Raises AuthError for every failure mode with the SAME message. Telling a
    caller whether a token was expired, malformed or wrongly signed hands an
    attacker a debugging oracle for free.
    """
    import jwt

    secret = _require_secret()

    try:
        claims = jwt.decode(
            token,
            secret,
            # ⚠️  ALGORITHM PINNED, AND IT MUST STAY PINNED.
            # Accepting whatever the token's header declares is the classic
            # JWT vulnerability: an attacker sets alg to "none", or swaps
            # HS256 for RS256 so the public key is used as an HMAC key, and
            # forges tokens at will.
            algorithms=[_ALGORITHM],
            options={"require": ["exp", "sub"]},
        )
    except Exception as e:  # noqa: BLE001
        # Logged with detail, returned without it.
        logger.info("Token rejected: %s", type(e).__name__)
        raise AuthError("Invalid or expired token.") from e

    if claims.get("typ") != "access":
        raise AuthError("Invalid or expired token.")

    return claims
