"""
RadAssist AI — API v1 Router

WHY A SEPARATE ROUTER FILE?
As the project grows, we'll have MANY endpoint files:
- health.py (system checks)
- reports.py (report generation)
- cases.py (case management)
- knowledge.py (knowledge base search)
- chat.py (AI assistant)

Instead of importing all of them in main.py (messy), we collect
them all here with a URL prefix. This keeps main.py clean and
makes it easy to add new feature endpoints.

URL STRUCTURE:
All v1 endpoints live under /api/v1/...
    /api/v1/health     → health check
    /api/v1/reports    → report operations (Phase 3)
    /api/v1/knowledge  → knowledge base (Phase 2)
    /api/v1/chat       → AI assistant (Phase 3+)
"""

from fastapi import APIRouter, Depends
from app.api.v1.endpoints import health
from app.api.v1.endpoints import knowledge  # Phase 2: Knowledge base endpoints
from app.api.v1.endpoints import chat       # Phase 3: AI chat with RAG
from app.api.v1.endpoints import images     # Phase 4: DICOM & image storage
from app.api.v1.endpoints import reports    # Phase 5: report drafts & sign-off
from app.api.v1.endpoints import auth       # Phase 6: login (no registration)
from app.core.deps import get_current_user

# Create the v1 router — all v1 endpoints are collected here
api_v1_router = APIRouter(prefix="/api/v1")


# ══════════════════════════════════════════════════════════════
# AUTHENTICATION — DEFAULT DENY (Phase 6, Step 2)
# ══════════════════════════════════════════════════════════════
#
# ⚠️  APPLIED PER ROUTER, NOT PER ROUTE.
#
# Every clinical router carries the dependency as a whole, so a new endpoint
# added inside one is protected the moment it exists. The alternative —
# decorating each route by hand — means protection is opt-in, and the failure
# mode is silent: somebody adds an endpoint in a hurry six months from now,
# forgets the dependency, and nothing anywhere complains.
#
# Adding a router to the public list below is a visible decision that shows up
# in review. Forgetting to protect one is an accident. Only the second
# actually happens, so the design makes it impossible.
#
# PUBLIC, and this list is exactly three routes:
#   GET  /api/v1/health       — liveness, needed before anyone can log in
#   POST /api/v1/auth/login   — you cannot present a token to get a token
#   GET  /api/v1/auth/me      — self-guards via get_current_user in the route
#
# tests/test_authz.py enumerates every registered route and fails if one is
# neither protected nor on that list.
_PROTECTED = [Depends(get_current_user)]

# ── Public ───────────────────────────────────────────────────
api_v1_router.include_router(health.router)
api_v1_router.include_router(auth.router)        # /login public, /me self-guards

# ── Authenticated ────────────────────────────────────────────
# ⚠️  images MATTERS MOST HERE. /images/{id}/file serves the actual uploaded
# picture of a patient's report. A UUID is not an access control: it leaks
# through logs, browser history, referrer headers and screenshots.
api_v1_router.include_router(knowledge.router, dependencies=_PROTECTED)
api_v1_router.include_router(chat.router, dependencies=_PROTECTED)
api_v1_router.include_router(images.router, dependencies=_PROTECTED)
api_v1_router.include_router(reports.router, dependencies=_PROTECTED)

# Future phases will add:
# api_v1_router.include_router(cases.router, dependencies=_PROTECTED)

