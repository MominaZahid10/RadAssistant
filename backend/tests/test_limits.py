"""
Rate limiting (Phase 6, Step 4).

Two reasons this exists, both created by opening registration:

  COST      — every chat message is a paid model call, and anyone can now sign
              up. The project's risk table names "API cost overruns at scale"
              with the mitigation "per-user usage caps", which needed users.

  ENUMERATION — registration must reveal whether an email is taken; a signup
              form cannot avoid it without leaving people unable to sign in.
              Limiting the attempts is what stops one unavoidable disclosure
              becoming a directory scan.
"""

import time

import pytest

from app.core.limits import (
    CHAT,
    LOGIN,
    REGISTER,
    UPLOAD,
    Limit,
    SlidingWindow,
    reset_all,
)


@pytest.fixture(autouse=True)
def _clean():
    reset_all()
    yield
    reset_all()


# ══════════════════════════════════════════════════════════════
# THE WINDOW
# ══════════════════════════════════════════════════════════════


def test_requests_under_the_limit_pass():
    w = SlidingWindow()
    limit = Limit(3, 60)
    assert [w.check("k", limit) for _ in range(3)] == [None, None, None]


def test_the_next_request_is_refused_with_a_wait_time():
    w = SlidingWindow()
    limit = Limit(2, 60)
    w.check("k", limit)
    w.check("k", limit)

    retry_after = w.check("k", limit)
    assert retry_after is not None
    assert 0 < retry_after <= 60


def test_the_window_slides_rather_than_resetting_on_a_boundary():
    """
    ⚠️  A FIXED WINDOW ALLOWS DOUBLE THE LIMIT AT THE SEAM.
    Twenty requests at 10:00:59 and twenty more at 10:01:00 both pass a
    "20 per minute" fixed window — forty requests in one second. Counting the
    last 60 seconds from now is what the limit was meant to say.
    """
    w = SlidingWindow()
    limit = Limit(2, 1)          # 2 per second, so the test is fast

    assert w.check("k", limit) is None
    assert w.check("k", limit) is None
    assert w.check("k", limit) is not None      # third is refused

    time.sleep(1.05)                            # the first two age out
    assert w.check("k", limit) is None


def test_a_refused_request_is_not_counted():
    """
    ⚠️  OTHERWISE RETRYING EXTENDS THE PENALTY INDEFINITELY.
    If a blocked attempt were recorded, an impatient client hammering the
    endpoint would keep pushing its own unlock further away while a patient
    one recovered. The punishment should not scale with frustration.
    """
    w = SlidingWindow()
    limit = Limit(1, 2)

    w.check("k", limit)
    first_wait = w.check("k", limit)
    time.sleep(0.3)
    second_wait = w.check("k", limit)

    assert second_wait is not None
    assert second_wait < first_wait, "the wait should shrink, not reset"


# ══════════════════════════════════════════════════════════════
# ISOLATION — the property that makes this fair
# ══════════════════════════════════════════════════════════════


def test_one_key_cannot_exhaust_anothers_quota():
    """One user hitting their cap must not affect anybody else."""
    w = SlidingWindow()
    limit = Limit(1, 60)

    assert w.check("user:alice", limit) is None
    assert w.check("user:alice", limit) is not None   # alice is capped
    assert w.check("user:bob", limit) is None         # bob is unaffected


def test_buckets_are_separate():
    """
    Uploading must not consume the chat allowance. Someone who has hit their
    upload cap should still be able to ask a question.
    """
    w = SlidingWindow()
    limit = Limit(1, 60)

    assert w.check("upload:user:x", limit) is None
    assert w.check("upload:user:x", limit) is not None
    assert w.check("chat:user:x", limit) is None


# ══════════════════════════════════════════════════════════════
# THE LIMITS THEMSELVES
# ══════════════════════════════════════════════════════════════


def test_registration_is_slow_enough_to_stop_a_directory_scan():
    """
    5 per hour per IP. Checking a hospital's staff list would take weeks
    rather than minutes — which is the point, because registration cannot
    hide whether an address is already taken.
    """
    assert REGISTER.times <= 5
    assert REGISTER.seconds >= 3600


def test_login_is_limited_by_ip_and_not_by_account():
    """
    ⚠️  AN ACCOUNT LOCKOUT IS A DENIAL OF SERVICE WITH A LOGIN FORM.
    Locking after N failures lets anyone lock a colleague out by guessing
    wrong at their address. Limiting the source costs the attacker; limiting
    the account costs the victim.
    """
    import inspect
    from app.api.v1.endpoints import auth as auth_module

    source = inspect.getsource(auth_module)
    assert "per_ip(LOGIN" in source
    assert "per_user(LOGIN" not in source


def test_chat_is_limited_per_user_not_per_ip():
    """
    Two clinicians behind one hospital NAT share an address. Counting by IP
    would have one of them exhaust the other's allowance.
    """
    import inspect
    from app.api.v1.endpoints import chat as chat_module

    source = inspect.getsource(chat_module)
    assert "per_user(CHAT" in source
    assert "per_ip(CHAT" not in source


def test_chat_allows_more_than_a_human_types_and_less_than_a_script_spends():
    assert 10 <= CHAT.times <= 60
    assert CHAT.seconds == 60


def test_upload_is_tighter_than_chat():
    """Each upload runs OCR and a vision model, and writes to disk."""
    assert UPLOAD.times < CHAT.times


# ══════════════════════════════════════════════════════════════
# THE RESPONSE
# ══════════════════════════════════════════════════════════════


def test_a_refusal_carries_retry_after():
    """
    RFC 6585. Without it a client cannot back off correctly, and the usual
    behaviour is to retry immediately and make the problem worse.
    """
    from app.core.limits import _too_many

    exc = _too_many(Limit(5, 60), 12.3)
    assert exc.status_code == 429
    assert exc.headers["Retry-After"] == "13"      # rounded UP, never to 0


def test_retry_after_is_never_zero():
    """A Retry-After of 0 invites an immediate retry that will also fail."""
    from app.core.limits import _too_many

    assert int(_too_many(Limit(5, 60), 0.01).headers["Retry-After"]) >= 1


def test_the_message_says_what_the_limit_is():
    """
    A bare "too many requests" leaves the caller guessing whether to wait a
    second or an hour.
    """
    from app.core.limits import _too_many

    assert "5 per minute" in _too_many(Limit(5, 60), 1).detail
    assert "5 per hour" in _too_many(Limit(5, 3600), 1).detail


# ══════════════════════════════════════════════════════════════
# HONEST LIMITATIONS
# ══════════════════════════════════════════════════════════════


def test_the_per_process_limitation_is_documented():
    """
    ⚠️  COUNTERS LIVE IN ONE PROCESS.
    Two instances behind a load balancer means the effective limit doubles,
    because neither knows about the other. Fine for a single-container pilot,
    wrong at scale — and it must be written down rather than discovered by
    someone wondering why "20 per minute" allowed sixty.
    """
    from app.core import limits

    doc = limits.__doc__ or ""
    assert "PER-PROCESS" in doc.upper()
    assert "redis" in doc.lower()


def test_the_forwarded_header_caveat_is_documented():
    """
    X-Forwarded-For is client-controlled unless a proxy overwrites it. Behind
    Railway or nginx that is fine; without one, an IP limit can be evaded by
    forging the header. That is why the authenticated limits key on the user,
    where the subject cannot be spoofed without a valid token.
    """
    import inspect
    from app.core import limits

    source = inspect.getsource(limits._client_ip)
    assert "client-controlled" in source.lower()
