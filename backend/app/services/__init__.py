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
# Phase 3:
#   llm_service.py   → Multi-provider LLM integration (Groq/Mistral/OpenAI)
#   rag_service.py   → RAG orchestrator (retrieval + prompt + generation)

