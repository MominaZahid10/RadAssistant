"""
Authorisation coverage (Phase 6, Step 2).

⚠️  THIS FILE IS THE ONLY THING THAT KEEPS THE SYSTEM PROTECTED AFTER THE
PHASE ENDS.

Every other test here checks that some code does what it says. This one checks
that code which does NOT yet exist will be safe: it enumerates every route
registered on the API and fails if any of them is neither authenticated nor on
a short, explicit public list.

An endpoint added in a hurry six months from now — by someone who never read
the auth code — fails this test by default. That is the whole point. Security
that depends on each future author remembering is not security; it is a
hope with a good track record until the day it runs out.
"""

import pytest

from app.api.v1.router import api_v1_router
from app.core.deps import get_current_user


# ⚠️  CHANGING THIS LIST IS A SECURITY DECISION.
# Four routes, each for a reason that cannot be worked around:
#
#   /health         — liveness. Checked by the container and by anyone
#                     diagnosing an outage, both of which happen when nobody
#                     can log in. Exposes component names, not data.
#   /auth/login     — you cannot present a token in order to obtain one.
#   /auth/register  — nor to create the account that would issue one.
#                     ⚠️  This entry was ADDED in Phase 6 Step 3, and only
#                     became defensible once migration 0006 gave reports and
#                     images an owner. Before that, an open signup meant
#                     anyone with the URL could read uploaded patient
#                     material. It is also gated by ALLOW_REGISTRATION, which
#                     a clinical deployment sets false.
#   /auth/me        — guarded inside the route via get_current_user, so it is
#                     authenticated despite not carrying a router-level
#                     dependency. Verified separately below.
PUBLIC_PATHS = {
    "/api/v1/health",
    "/api/v1/auth/login",
    "/api/v1/auth/register",
    "/api/v1/auth/me",
}


def _routes():
    for route in api_v1_router.routes:
        if hasattr(route, "path") and hasattr(route, "dependant"):
            yield route


def _is_protected(route) -> bool:
    """True if get_current_user runs for this route, at any level."""
    def walk(dependant):
        if dependant.call is get_current_user:
            return True
        return any(walk(sub) for sub in dependant.dependencies)

    return walk(route.dependant)


# ══════════════════════════════════════════════════════════════
# THE LOAD-BEARING TEST
# ══════════════════════════════════════════════════════════════


def test_every_route_is_authenticated_or_explicitly_public():
    """
    ⚠️  IF THIS FAILS, SOMEBODY ADDED AN UNPROTECTED ENDPOINT.
    Either add the dependency, or add the path to PUBLIC_PATHS and explain
    why in the comment above it. Do not silence the test.
    """
    unprotected = [
        route.path for route in _routes()
        if not _is_protected(route) and route.path not in PUBLIC_PATHS
    ]
    assert unprotected == [], (
        "These routes are reachable without authentication:\n  "
        + "\n  ".join(sorted(unprotected))
    )


def test_the_public_list_has_not_quietly_grown():
    """
    Catches the lazy fix. Making the previous test pass by appending to
    PUBLIC_PATHS is exactly the mistake this pair exists to prevent, so the
    size is pinned and has to be changed deliberately.
    """
    assert len(PUBLIC_PATHS) == 4, (
        "PUBLIC_PATHS changed. Every entry is a route anyone on the internet "
        "can call. Justify the addition before updating this count."
    )


def test_every_public_path_actually_exists():
    """A stale entry silently widens the allowlist for a route that may return."""
    registered = {route.path for route in _routes()}
    assert PUBLIC_PATHS <= registered


# ══════════════════════════════════════════════════════════════
# THE ROUTES THAT MATTER MOST
# ══════════════════════════════════════════════════════════════


@pytest.mark.parametrize("path", [
    "/api/v1/images/{image_id}/file",        # the actual uploaded report photo
    "/api/v1/images/{image_id}/thumbnail",
    "/api/v1/images",
    "/api/v1/reports",
    "/api/v1/reports/{report_id}",
    "/api/v1/chat",
    "/api/v1/knowledge/documents",
])
def test_clinical_routes_require_authentication(path):
    """
    Named individually so a failure says WHICH door was left open, rather
    than only that one was.

    /images/{id}/file is the one to care about: it returns the photograph of
    a patient's report. A UUID is not an access control — it leaks through
    logs, browser history, referrer headers and screenshots.
    """
    matching = [r for r in _routes() if r.path == path]
    assert matching, f"{path} is not registered — did it move?"
    for route in matching:
        assert _is_protected(route), f"{path} is reachable without a token"


def test_auth_me_is_protected_even_though_it_is_in_the_public_list():
    """
    It appears in PUBLIC_PATHS because it carries no ROUTER-level dependency,
    but it declares get_current_user itself. Asserted so the allowlist entry
    cannot be mistaken for it being open.
    """
    route = next(r for r in _routes() if r.path == "/api/v1/auth/me")
    assert _is_protected(route)


def test_login_is_genuinely_open():
    """The one route that must not require a token, by definition."""
    route = next(r for r in _routes() if r.path == "/api/v1/auth/login")
    assert not _is_protected(route)


def test_health_is_open():
    """Checked when nobody can log in, which is when it is needed most."""
    route = next(r for r in _routes() if r.path == "/api/v1/health")
    assert not _is_protected(route)


# ══════════════════════════════════════════════════════════════
# HOW PROTECTION IS APPLIED
# ══════════════════════════════════════════════════════════════


def test_protection_is_applied_per_router_not_per_route():
    """
    ⚠️  THE DIFFERENCE BETWEEN FAIL-CLOSED AND FAIL-OPEN.
    Router-level means a new endpoint inside an existing router is protected
    the moment it exists. Per-route means protection is opt-in, and forgetting
    it produces no error at all.
    """
    import inspect
    from app.api.v1 import router as router_module

    source = inspect.getsource(router_module)
    assert "dependencies=_PROTECTED" in source
    # Every clinical router carries it.
    for name in ("knowledge", "chat", "images", "reports"):
        assert f"{name}.router, dependencies=_PROTECTED" in source


def test_a_missing_token_and_a_bad_token_are_indistinguishable():
    """
    auto_error=False routes both through our handler, so both produce the same
    401 with the same body. Letting FastAPI answer one and us the other lets a
    caller tell "no token" from "bad token" — a small oracle, but a free one.
    """
    import inspect
    from app.core import deps

    source = inspect.getsource(deps)
    assert "auto_error=False" in source


def test_registration_is_public_but_gated():
    """
    It has to be reachable without a token — you cannot present one to create
    the account that would issue it. The protection is ALLOW_REGISTRATION plus
    per-user ownership, not authentication.
    """
    route = next(r for r in _routes() if r.path == "/api/v1/auth/register")
    assert not _is_protected(route)

    import inspect
    from app.api.v1.endpoints import auth as auth_module

    assert "settings.ALLOW_REGISTRATION" in inspect.getsource(auth_module.register)
