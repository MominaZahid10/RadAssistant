# ══════════════════════════════════════════════════════════════
# Business Logic / Services Package
# ══════════════════════════════════════════════════════════════
# Services contain the actual logic (not just routing).
# Each service handles one responsibility:
#
# Phase 2:
#   embedding.py     → Load & run the embedding model
#   qdrant_service.py → Manage vector storage in Qdrant
#   ingestion.py     → Full document processing pipeline
#
# Future Phases:
#   report_service.py → Report generation (Phase 3)
#   llm_service.py    → LLM integration (Phase 3)
#   chat_service.py   → AI assistant (Phase 3+)
