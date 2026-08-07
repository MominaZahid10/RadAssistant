# 🩻 RadAssist AI

**An Explainable, Multimodal Retrieval-Augmented Radiology Reporting & Clinical Decision Support System**

RadAssist AI assists radiologists in drafting accurate, standardized, and evidence-grounded reports. Every AI suggestion comes with traceable evidence — the radiologist remains the final decision-maker.

## 🏗️ Architecture

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Next.js 16 + Tailwind CSS | Dashboard, report editor, evidence viewer |
| **Backend** | FastAPI (Python) | API, RAG orchestration, LLM integration |
| **Database** | PostgreSQL | Users, cases, reports, audit logs |
| **Vector DB** | Qdrant | Semantic search for RAG retrieval |
| **LLM** | Mistral / OpenAI (configurable) | Report generation, analysis |

## 🚀 Quick Start

### Prerequisites
- Docker Desktop
- Git (optional)

### Run Locally
```bash
# 1. Clone/navigate to the project
cd RAG

# 2. Start all services (first time takes a few minutes)
docker-compose up --build

# 3. Open in browser
#    Frontend:  http://localhost:3000
#    API Docs:  http://localhost:8000/docs
#    Qdrant:    http://localhost:6333/dashboard
```

### Stop
```bash
docker-compose down          # Stop containers (data preserved)
docker-compose down -v       # Stop + delete all data
```

## 📁 Project Structure
```
RAG/
├── backend/              # FastAPI Python backend
│   ├── app/
│   │   ├── main.py       # App entry point
│   │   ├── config.py     # Environment settings
│   │   ├── api/v1/       # API endpoints
│   │   ├── core/         # DB connection, auth
│   │   ├── models/       # Database table models
│   │   ├── schemas/      # API request/response shapes
│   │   └── services/     # Business logic
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/             # Next.js frontend
│   ├── src/
│   │   ├── app/          # Pages (App Router)
│   │   ├── components/   # Reusable UI components
│   │   └── lib/          # Utilities & API client
│   └── Dockerfile
├── docker-compose.yml    # Orchestrates all services
└── README.md
```

## 📋 Development Phases

| Phase | Name | Status |
|-------|------|--------|
| 1 | Architecture & Environment Setup | ✅ Current |
| 2 | Knowledge Base & Ingestion Pipeline | ⏳ Next |
| 3 | Core RAG + Report Generation (MVP) | 🔮 Planned |
| 4 | Multimodal Ingestion | 🔮 Planned |
| 5 | Decision Support Features | 🔮 Planned |
| 6 | Explainability, Auth & Hardening | 🔮 Planned |
| 7 | Deployment & Pilot | 🔮 Planned |

## 📝 License

This project is developed as an internship deliverable.
=======

