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

from fastapi import APIRouter
from app.api.v1.endpoints import health
from app.api.v1.endpoints import knowledge  # Phase 2: Knowledge base endpoints

# Create the v1 router — all v1 endpoints are collected here
api_v1_router = APIRouter(prefix="/api/v1")

# ── Register endpoint modules ────────────────────────────────
# Each "include_router" adds all endpoints from that module.
# As we build new features, we just add one line here.
api_v1_router.include_router(health.router)
api_v1_router.include_router(knowledge.router)  # Phase 2: /api/v1/knowledge/*

# Future phases will add:
# api_v1_router.include_router(reports.router)   # Phase 3
# api_v1_router.include_router(cases.router)     # Phase 3
# api_v1_router.include_router(chat.router)      # Phase 3+

