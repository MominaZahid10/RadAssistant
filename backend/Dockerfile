# ══════════════════════════════════════════════════════════════
# RadAssist AI — Backend Dockerfile
# ══════════════════════════════════════════════════════════════
#
# WHAT IS A DOCKERFILE?
# A recipe that tells Docker how to build a container image.
# Think of it as: "Start with Python, copy my code in, install
# dependencies, and run the server."
#
# WHY CONTAINERS?
# They guarantee "it works on my machine" actually works everywhere.
# Your PostgreSQL, Qdrant, backend, and frontend each get their
# own isolated container with their own dependencies.
# ══════════════════════════════════════════════════════════════

# Start from an official Python 3.12 slim image (smaller = faster)
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# ── Install system dependencies ──────────────────────────────
# build-essential: some Python packages need C libraries to compile
# tesseract-ocr:   the actual OCR ENGINE. `pytesseract` in requirements.txt
#                  is only a thin Python wrapper — it shells out to this
#                  binary. Without it, every image upload fails at runtime.
# tesseract-ocr-eng: the English language training data
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# ── Install Python dependencies ──────────────────────────────
# TRICK: Copy requirements.txt FIRST, then install.
# Docker caches each step. If requirements.txt hasn't changed,
# Docker skips reinstalling (saves minutes on rebuilds).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ────────────────────────────────────
COPY . .

# ── Expose the port FastAPI runs on ──────────────────────────
EXPOSE 8000

# ── Start the server ─────────────────────────────────────────
# --host 0.0.0.0: Listen on all interfaces (required in Docker)
# --port 8000: The port number
#
# NOTE: --reload is deliberately NOT here. It's a development-only flag,
# and baking it into the image makes the image undeployable. docker-compose
# adds it back via a `command:` override for local dev.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
