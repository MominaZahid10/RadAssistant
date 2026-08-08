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

# Future phases will add:
# from app.models.user import User          # Phase 6 (Authentication)
# from app.models.case import Case          # Phase 3 (Cases)
# from app.models.report import Report      # Phase 3 (Reports)
# from app.models.audit_log import AuditLog # Phase 6 (Audit trail)
