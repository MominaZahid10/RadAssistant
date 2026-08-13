"""
Tests for authentication (Phase 6, Step 1).

Until this phase every endpoint was open: anyone who could reach port 8000
could list every report and download every uploaded image — photographs of
patient reports. What made that survivable was a rule ("synthetic data only"),
not a control.

Most of this file tests things that must NOT happen: a default signing key, a
password in a log, a login form that reveals which accounts exist.
"""

import secrets
import time

import pytest

from app.core import security as sec
from app.core.security import (
    AuthError,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


@pytest.fixture
def signing_key(monkeypatch):
    key = secrets.token_urlsafe(48)
    monkeypatch.setattr(sec.settings, "JWT_SECRET", key)
    return key


# ══════════════════════════════════════════════════════════════
# THE SIGNING KEY
# ══════════════════════════════════════════════════════════════


def test_no_secret_refuses_to_issue_a_token(monkeypatch):
    """
    ⚠️  THE MOST IMPORTANT TEST IN THIS FILE.
    A signing key with a fallback works in development, ships to production
    unnoticed, and is shared by every deployment that cloned the repository —
    so anyone holding the default can mint a valid token for any user on any
    instance. A loud boot failure costs minutes; a shared key costs everything
    the token protects, silently.
    """
    monkeypatch.setattr(sec.settings, "JWT_SECRET", "")
    with pytest.raises(AuthError, match="JWT_SECRET is not set"):
        create_access_token("some-user-id")


def test_the_config_default_is_empty():
    """
    Not "changeme", not a sample value. Empty, so it cannot be shipped.

    ⚠️  READS THE DECLARED DEFAULT, NOT AN INSTANCE.
    `Settings().JWT_SECRET` loads from .env, so once a developer sets a real
    key locally this test would pass for the wrong reason — and would keep
    passing even if someone later added a hard-coded fallback to the field.
    The field definition is what has to stay empty.
    """
    from app.config import Settings

    assert Settings.model_fields["JWT_SECRET"].default == ""


def test_a_short_secret_is_refused(monkeypatch):
    """A short key can be recovered offline from a single captured token."""
    monkeypatch.setattr(sec.settings, "JWT_SECRET", "tooshort")
    with pytest.raises(AuthError, match="at least 32"):
        create_access_token("some-user-id")


@pytest.mark.parametrize("placeholder", [
    "changeme-changeme-changeme-changeme",
    "your_secret_key_here_your_secret_key",
    "secret-secret-secret-secret-secret-x",
])
def test_placeholder_secrets_are_refused(monkeypatch, placeholder):
    """Catches the likeliest copy-paste: the example value from a README."""
    monkeypatch.setattr(sec.settings, "JWT_SECRET", placeholder)
    with pytest.raises(AuthError, match="placeholder"):
        create_access_token("some-user-id")


# ══════════════════════════════════════════════════════════════
# PASSWORDS
# ══════════════════════════════════════════════════════════════


def test_the_hash_does_not_contain_the_password():
    password = "correct horse battery staple"
    hashed = hash_password(password)
    assert password not in hashed
    assert hashed.startswith("$2b$")     # bcrypt, not a fast digest


def test_the_same_password_hashes_differently_each_time():
    """
    Per-password salting. Without it, two users with the same password share
    a hash — so cracking one cracks both, and a stolen table can be attacked
    in bulk with a precomputed dictionary.
    """
    assert hash_password("same password") != hash_password("same password")


def test_verification_round_trips():
    hashed = hash_password("s3cure-passphrase")
    assert verify_password("s3cure-passphrase", hashed) is True
    assert verify_password("s3cure-passphras", hashed) is False
    assert verify_password("", hashed) is False


def test_a_corrupt_hash_reads_as_a_wrong_password():
    """
    Never a 500. A malformed value in the column must not turn a login attempt
    into an error that tells the caller something about what is stored.
    """
    assert verify_password("anything", "not-a-bcrypt-hash") is False
    assert verify_password("anything", "") is False


def test_over_long_passwords_are_refused_not_truncated():
    """
    ⚠️  bcrypt SILENTLY IGNORES ANYTHING PAST 72 BYTES.
    Truncating would mean a 100-character passphrase is weaker than the user
    believes, and two different long passwords could share a hash. Refusing
    tells them; truncating does not.
    """
    with pytest.raises(AuthError, match="72 bytes"):
        hash_password("x" * 100)


def test_empty_passwords_are_refused():
    with pytest.raises(AuthError):
        hash_password("")


# ══════════════════════════════════════════════════════════════
# TOKENS
# ══════════════════════════════════════════════════════════════


def test_token_round_trips(signing_key):
    token = create_access_token("11111111-2222-3333-4444-555555555555")
    assert decode_access_token(token)["sub"] == "11111111-2222-3333-4444-555555555555"


def test_a_tampered_token_is_rejected(signing_key):
    token = create_access_token("user-1")
    tampered = token[:-4] + ("abcd" if not token.endswith("abcd") else "efgh")
    with pytest.raises(AuthError):
        decode_access_token(tampered)


def test_a_token_signed_with_another_key_is_rejected(monkeypatch, signing_key):
    """The whole point of signing."""
    token = create_access_token("user-1")
    monkeypatch.setattr(sec.settings, "JWT_SECRET", secrets.token_urlsafe(48))
    with pytest.raises(AuthError):
        decode_access_token(token)


def test_an_expired_token_is_rejected(monkeypatch, signing_key):
    monkeypatch.setattr(sec.settings, "JWT_EXPIRE_MINUTES", 0)
    token = create_access_token("user-1")
    time.sleep(1.1)
    with pytest.raises(AuthError):
        decode_access_token(token)


def test_the_algorithm_is_pinned():
    """
    ⚠️  THE CLASSIC JWT VULNERABILITY.
    Accepting whatever the token header declares lets an attacker set alg to
    "none", or swap HS256 for RS256 so a public key is used as an HMAC key —
    and forge tokens at will.
    """
    import inspect

    source = inspect.getsource(sec.decode_access_token)
    assert "algorithms=[_ALGORITHM]" in source


def test_an_unsigned_token_is_rejected(signing_key):
    """alg=none, spelled out."""
    import base64
    import json

    def b64(d: dict) -> str:
        raw = json.dumps(d).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    forged = f"{b64({'alg': 'none', 'typ': 'JWT'})}.{b64({'sub': 'user-1'})}."
    with pytest.raises(AuthError):
        decode_access_token(forged)


def test_a_token_without_an_expiry_is_rejected(signing_key):
    """A token that never expires cannot be revoked by waiting."""
    import jwt

    forever = jwt.encode({"sub": "user-1", "typ": "access"}, signing_key,
                         algorithm="HS256")
    with pytest.raises(AuthError):
        decode_access_token(forever)


def test_every_rejection_gives_the_same_message(signing_key):
    """
    Expired, malformed and wrongly-signed all return one message. Telling the
    caller which one hands them a debugging oracle for free.
    """
    messages = set()
    for bad in ("not.a.token", "", "a.b.c"):
        try:
            decode_access_token(bad)
        except AuthError as e:
            messages.add(str(e))
    assert messages == {"Invalid or expired token."}


def test_the_subject_is_an_id_not_an_email(signing_key):
    """
    Emails change. A token carrying one would keep working against an address
    the user no longer has, and stop working the moment they updated it — for
    the same account.
    """
    import inspect

    source = inspect.getsource(sec.create_access_token)
    assert '"sub": str(subject)' in source
    assert "email" not in source.split('"""')[2]      # not in the code body


# ══════════════════════════════════════════════════════════════
# NO REGISTRATION
# ══════════════════════════════════════════════════════════════


def test_registration_is_gated_by_a_setting():
    """
    ⚠️  THIS TEST USED TO ASSERT NO REGISTRATION ROUTE EXISTED.
    That was correct while every signed-in user could see every report — open
    signup then meant anyone with the URL could read uploaded patient
    material. Migration 0006 gave reports and images an owner, so a new
    account lands in an empty workspace and the objection lapsed.

    The two remain coupled: registration is only safe while ownership holds.
    If ownership is ever reverted, ALLOW_REGISTRATION must go false in the
    same change.
    """
    import inspect
    from app.api.v1.endpoints import auth as auth_module

    source = inspect.getsource(auth_module.register)
    assert "settings.ALLOW_REGISTRATION" in source
    assert "HTTP_403_FORBIDDEN" in source


def test_registration_can_never_create_an_administrator():
    """
    A client-settable admin flag is a privilege escalation with a form on it.
    Administrators come only from scripts/create_user.py --admin, which needs
    shell access to the container.
    """
    import inspect
    from app.api.v1.endpoints import auth as auth_module
    from app.schemas.auth import RegisterRequest

    assert "is_admin" not in RegisterRequest.model_fields
    assert "is_admin=False" in inspect.getsource(auth_module.register)


def test_password_length_is_enforced_where_it_is_set_not_where_it_is_checked():
    """
    A minimum length on the LOGIN form only tells an attacker the policy
    before they have an account. On registration it protects the account.
    """
    from app.schemas.auth import LoginRequest, RegisterRequest

    assert RegisterRequest.model_fields["password"].metadata  # has constraints
    assert not LoginRequest.model_fields["password"].metadata


def test_no_user_administration_endpoints_are_exposed():
    """Listing or editing other people's accounts is not a public surface."""
    from app.api.v1.router import api_v1_router

    paths = {getattr(r, "path", "") for r in api_v1_router.routes}
    for forbidden in ("/api/v1/users", "/api/v1/auth/users"):
        assert forbidden not in paths


def test_the_create_user_script_takes_no_password_argument():
    """
    `--password hunter2` lands in shell history, in `ps` output for every
    other user on the machine, and in Docker's logs.
    """
    from pathlib import Path

    source = Path("scripts/create_user.py").read_text(encoding="utf-8")
    assert '"--password"' not in source
    assert "getpass" in source


# ══════════════════════════════════════════════════════════════
# LOGIN — timing and disclosure
# ══════════════════════════════════════════════════════════════


def test_unknown_email_still_burns_a_hash_comparison():
    """
    ⚠️  OTHERWISE RESPONSE TIME LEAKS THE USER LIST.
    Returning early on "no such user" skips bcrypt and answers in
    microseconds, while a real account takes ~100ms. That gap is measurable
    over the network and turns the login form into an account enumerator —
    an attacker learns which of a hospital's addresses are registered without
    guessing a single password.
    """
    import inspect
    from app.api.v1.endpoints import auth as auth_module

    source = inspect.getsource(auth_module.login)
    assert "dummy_verify(payload.password)" in source


def test_wrong_password_and_unknown_email_return_the_same_thing():
    import inspect
    from app.api.v1.endpoints import auth as auth_module

    source = inspect.getsource(auth_module.login)
    # One helper, used for every failure — no branch-specific message.
    assert source.count("_bad_credentials()") >= 3
    assert "Incorrect email or password." in inspect.getsource(
        auth_module._bad_credentials
    )


def test_a_disabled_account_is_indistinguishable_from_bad_credentials():
    """A distinct "account disabled" reply confirms the account exists."""
    import inspect
    from app.api.v1.endpoints import auth as auth_module

    source = inspect.getsource(auth_module.login)
    assert "if not user.is_active:" in source
    assert "disabled" not in source.lower().split("raise _bad_credentials()")[-1]


def test_the_password_field_is_marked_repr_false():
    """Keeps it out of validation errors and any log line formatting the model."""
    from app.schemas.auth import LoginRequest

    assert LoginRequest.model_fields["password"].repr is False


def test_the_user_repr_leaks_nothing():
    """repr lands in logs and exception output."""
    from app.models.user import User

    user = User(email="doctor@hospital.org", hashed_password="$2b$12$abc",
                full_name="Dr Jones")
    text = repr(user)
    assert "doctor@hospital.org" not in text
    assert "$2b$" not in text
    assert "Jones" not in text


# ══════════════════════════════════════════════════════════════
# EMAIL NORMALISATION
# ══════════════════════════════════════════════════════════════


@pytest.mark.parametrize("raw", [
    "Doctor@Hospital.ORG", "  doctor@hospital.org  ", "DOCTOR@HOSPITAL.ORG",
])
def test_emails_normalise_to_one_form(raw):
    """
    Treating email as case-sensitive means Alice@x.org and alice@x.org become
    two accounts, splitting one person's reports across both.
    """
    from app.models.user import User

    assert User.normalise_email(raw) == "doctor@hospital.org"


def test_login_normalises_before_lookup():
    """Normalising on write but not on read makes the unique constraint useless."""
    import inspect
    from app.api.v1.endpoints import auth as auth_module

    assert "User.normalise_email(payload.email)" in inspect.getsource(
        auth_module.login
    )


# ══════════════════════════════════════════════════════════════
# SESSION VALIDITY
# ══════════════════════════════════════════════════════════════


def test_the_user_is_reloaded_on_every_request():
    """
    ⚠️  A SIGNATURE ONLY PROVES WE ISSUED THE TOKEN.
    It says nothing about whether the account still exists or is still active.
    Without a lookup, deactivating a departed clinician leaves their token
    working until it expires — up to twelve hours of access to patient reports
    after somebody believed they had revoked it.
    """
    import inspect
    from app.core import deps

    source = inspect.getsource(deps.get_current_user)
    assert "select(User)" in source
    assert "not user.is_active" in source
