"""
RadAssist AI — Rate limiting (Phase 6, Step 4)

Caps the endpoints that cost money or enable enumeration.

════════════════════════════════════════════════════════════════════
WHY THIS MATTERS MORE NOW THAN IT DID YESTERDAY
════════════════════════════════════════════════════════════════════
Until registration opened, the only account was one the operator created.
Now anyone can sign up, and every chat message is a paid LLM call. The
project's own risk table lists *"API cost overruns at scale"* with the
mitigation *"per-user usage caps"* — which needed users to exist first.

There is a second reason. Registration returns 409 when an email is taken,
which reveals that the account exists. Login goes to real trouble to avoid
that disclosure; a signup form cannot. Rate limiting is what stops an
unavoidable single-address disclosure becoming a bulk enumeration tool.

════════════════════════════════════════════════════════════════════
⚠️  IN-MEMORY, THEREFORE PER-PROCESS. STATED PLAINLY.
════════════════════════════════════════════════════════════════════
Counters live in this process. Run two instances behind a load balancer and
the effective limit doubles, because neither knows about the other's requests.

That is acceptable for a single-container pilot and it is NOT acceptable at
scale — the fix is a shared counter in Redis, which is a change of storage
and not of logic. It is written down here rather than discovered later by
someone wondering why a "20 per minute" cap allowed sixty.

A restart also clears every counter. In practice that is fine: the window is
a minute, and a deploy is not a rate-limit bypass anyone can trigger on demand.

WHY NOT slowapi:
It is the usual choice and it is fine, but it keys on IP address by default.
Here the meaningful subject is the USER — an authenticated request should be
counted against the account, not the network. Getting that from slowapi means
a custom key function anyway, at which point this is forty lines with no
dependency and behaviour that is obvious from reading it.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request, status

from app.core.deps import get_current_user
from app.models.user import User


@dataclass(frozen=True)
class Limit:
    """`times` requests per `seconds`."""
    times: int
    seconds: int

    @property
    def description(self) -> str:
        unit = {60: "minute", 3600: "hour", 86400: "day"}.get(
            self.seconds, f"{self.seconds}s"
        )
        return f"{self.times} per {unit}"


class SlidingWindow:
    """
    A sliding-window counter keyed by an arbitrary string.

    ⚠️  SLIDING, NOT FIXED.
    A fixed window resets on the minute, so 20 requests at 10:00:59 and 20
    more at 10:01:00 pass a "20 per minute" limit — 40 requests in one second.
    A sliding window counts the last 60 seconds from now, which is what the
    limit was meant to say.
    """

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str, limit: Limit) -> float | None:
        """
        Record a hit. Returns None if allowed, or seconds to wait if not.

        The rejected request is NOT recorded. Counting it would extend the
        penalty every time a blocked client retried, so an impatient caller
        could lock themselves out indefinitely while a patient one recovered.
        """
        now = time.monotonic()
        window = self._hits[key]

        cutoff = now - limit.seconds
        while window and window[0] <= cutoff:
            window.popleft()

        if len(window) >= limit.times:
            # The oldest hit is what has to age out.
            return max(0.0, window[0] + limit.seconds - now)

        window.append(now)
        return None

    def reset(self, key: str | None = None) -> None:
        """Clear one key, or everything. Used by tests."""
        if key is None:
            self._hits.clear()
        else:
            self._hits.pop(key, None)


_window = SlidingWindow()


# ══════════════════════════════════════════════════════════════
# LIMITS
# ══════════════════════════════════════════════════════════════
# Set by COST and by ABUSE POTENTIAL, not uniformly. Listing reports is a
# database read; a chat message is a paid model call. A uniform limit would
# either be too tight for browsing or too loose for generation.

# Every message is an LLM call. 20/min is far above human typing speed and
# far below what a script could spend.
CHAT = Limit(20, 60)

# Each upload runs OCR and a vision model, and writes to disk.
UPLOAD = Limit(10, 60)

# ⚠️  THE ONE THAT STOPS ACCOUNT ENUMERATION.
# Registration must reveal whether an email is taken. At 5 per hour per IP,
# checking a hospital's staff directory would take weeks.
REGISTER = Limit(5, 3600)

# Slows password guessing without locking out a genuine typo.
# ⚠️  NOT AN ACCOUNT LOCKOUT, DELIBERATELY.
# Locking an account after N failures lets anyone lock a colleague out by
# guessing wrong at their address — a denial-of-service with a login form.
# Limiting by IP costs the attacker and not the victim.
LOGIN = Limit(10, 60)

# NCBI rate-limits on their side, and a ban would affect every user of this
# deployment rather than the one who triggered it.
EXTERNAL_FETCH = Limit(3, 3600)


# ══════════════════════════════════════════════════════════════
# KEYS
# ══════════════════════════════════════════════════════════════


def _client_ip(request: Request) -> str:
    """
    Best-effort client address.

    ⚠️  X-Forwarded-For IS CLIENT-CONTROLLED UNLESS A PROXY OVERWRITES IT.
    Behind Railway/Render/nginx the first entry is the real client and the
    header is set by infrastructure, so trusting it is correct there. Running
    without a proxy, anyone can forge it and evade an IP limit entirely.
    That is a real weakness of IP-based limiting and not one this code can
    solve — it is why the *authenticated* limits key on the user instead,
    where the subject cannot be spoofed without a valid token.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _too_many(limit: Limit, retry_after: float) -> HTTPException:
    seconds = max(1, int(retry_after + 0.999))
    return HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail=f"Rate limit exceeded ({limit.description}). Try again in {seconds}s.",
        # Required by RFC 6585 so a client can back off correctly rather than
        # hammering and making it worse.
        headers={"Retry-After": str(seconds)},
    )


# ══════════════════════════════════════════════════════════════
# DEPENDENCIES
# ══════════════════════════════════════════════════════════════


def per_user(limit: Limit, bucket: str):
    """
    Limit by account.

    The bucket name separates counters, so uploading does not consume the
    chat allowance — one user hitting their upload cap should still be able
    to ask a question.
    """
    async def dependency(user: User = Depends(get_current_user)) -> None:
        retry_after = _window.check(f"{bucket}:user:{user.id}", limit)
        if retry_after is not None:
            raise _too_many(limit, retry_after)

    return dependency


def per_ip(limit: Limit, bucket: str):
    """
    Limit by address. For routes reached before there is a user to blame.
    """
    async def dependency(request: Request) -> None:
        retry_after = _window.check(f"{bucket}:ip:{_client_ip(request)}", limit)
        if retry_after is not None:
            raise _too_many(limit, retry_after)

    return dependency


def reset_all() -> None:
    """Clear every counter. Tests only."""
    _window.reset()
