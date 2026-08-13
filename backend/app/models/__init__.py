# ══════════════════════════════════════════════════════════════
# Database Models Package
# ══════════════════════════════════════════════════════════════
# Every model imported here will be auto-discovered by SQLAlchemy
# and its table will be created when the app starts up.
#
# HOW IT WORKS:
# In main.py, we call Base.metadata.create_all() which scans all
# classes that inherit from Base. By importing them HERE, Python
# loads them into memory so SQLAlchemy can find them.
# If you create a new model file but forget to import it here,
# its table WON'T be created. This is the #1 gotcha.

from app.models.document import Document  # noqa: F401
from app.models.image import MedicalImage  # noqa: F401 — Phase 4
from app.models.report import Report, ReportStatus  # noqa: F401 — Phase 5
from app.models.user import User  # noqa: F401 — Phase 6

# ⚠️  EVERY model must be imported here. Alembic's autogenerate
# compares Base.metadata against the live database — a model that
# isn't imported is invisible to it, and autogenerate will propose
# DROPPING its table.

# Future phases will add:
# from app.models.user import User          # Phase 6 (Authentication)
# from app.models.case import Case          # Phase 5+ (Cases)
# from app.models.audit_log import AuditLog # Phase 6 (Audit trail)
