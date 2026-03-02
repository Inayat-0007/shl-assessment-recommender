---
title: SHL Assessment Recommender
emoji: 🎯
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
---

# SHL SmartMatch AI — Assessment Recommendation System

An intelligent recommendation engine that matches job descriptions to SHL's catalog of 389 Individual Test Solutions. Built with a hybrid approach combining semantic search (sentence-transformers), keyword overlap, assessment name matching, and LLM-powered query expansion (Google Gemini) to maximize Recall@10.

## Architecture

```
User Query / JD URL
        |
   [Input Sanitization & Security]
        |
   [Gemini LLM Query Expansion]
        |
   [Sentence-Transformer Embedding]
        |
   [Hybrid Search: Semantic + Keyword + Name Match]
        |
   [Constraint Filters: Type, Time, Job Level]
        |
   [Multi-Domain Balancing (K + P)]
        |
   Top 5-10 Recommendations (JSON)
```

## Quick Start

```bash
# Clone
git clone https://github.com/Inayat-0007/shl-assessment-recommender.git
cd shl-assessment-recommender

# Setup
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# Run Backend API
python main.py
# API: http://localhost:8000
# Docs: http://localhost:8000/docs

# Run Frontend (separate terminal)
streamlit run frontend/app.py
# UI: http://localhost:8501
```

## Live Deployment (Hugging Face Spaces + Streamlit Cloud)

The backend API is deployed on **Hugging Face Spaces** (Docker SDK, exposing port `7860`).

```bash
# Push to Hugging Face
git remote add hf https://huggingface.co/spaces/Inayat05/shl-assessment-recommender
git push hf main
```

The frontend runs on **Streamlit Cloud**, connecting to the backend via the `BACKEND_URL` secret.

## API Endpoints

### Health Check
```
GET /health
Response: {"status": "healthy", "assessments": 389}
```

### Recommend Assessments
```
POST /recommend
Body: {"query": "Java developer with collaboration skills"}

GET /recommend?query=Python+SQL+developer
```

### Response Format
```json
{
  "recommended_assessments": [
    {
      "url": "https://www.shl.com/products/product-catalog/view/java-8-new/",
      "adaptive_support": "No",
      "description": "...",
      "duration": 20,
      "remote_support": "Yes",
      "test_type": ["K"]
    }
  ]
}
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| LLM | Google Gemini 2.0 Flash (query expansion) |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Search | Hybrid: Cosine Similarity + Keyword Overlap + Name Match |
| Security | slowapi (rate limit), bleach (XSS), SSRF validation |
| Data | 389 Individual Test Solutions scraped from SHL catalog |

## Security Features

- **Rate Limiting**: 100 requests/min/IP via slowapi
- **Input Sanitization**: HTML/XSS stripping via bleach
- **SSRF Protection**: Private IP blocking on URL inputs
- **Security Headers**: X-Content-Type-Options, X-Frame-Options, CSP
- **Error Masking**: No stack traces or file paths exposed
- **Request Size Limiting**: 1MB max payload
- **CORS**: Open for evaluation (production would whitelist)

## Project Structure

```
SHL_SmartMatch_AI/
├── main.py                  # Entry point (port 8000)
├── Dockerfile               # Hugging Face deployment config (port 7860)
├── requirements.txt         # Dependencies
├── .env.example             # Environment template
├── generate_results.py      # Test set prediction generator
├── test_engine.py           # Engine unit tests
├── data/
│   ├── shl_catalog.csv      # Raw scraped catalog
│   ├── shl_catalog_clean.csv # Cleaned catalog (389 assessments)
│   ├── train_set.csv        # 10 labeled queries
│   ├── test_set.csv         # 9 unlabeled test queries
│   └── embeddings.npy       # Pre-computed embeddings for fast boot
├── scraper/
│   ├── scrape_catalog.py    # SHL catalog scraper
│   └── clean_catalog.py     # Data cleaning pipeline
├── src/
│   ├── engine.py            # Core recommendation engine
│   ├── api.py               # FastAPI application + security
│   ├── evaluate.py          # Mean Recall@K evaluation
│   └── utils.py             # Security utilities
├── frontend/
│   └── app.py               # Streamlit web interface
├── results/
│   └── results.csv          # Test set predictions
└── docs/
    └── approach_document.pdf # Approach documentation
```

## Evaluation

Mean Recall@10 computed against labeled train set (10 queries).
Results CSV generated on unlabeled test set (9 queries).

## Author

Built by **Mohammad Inayat Hussain** for the SHL GenAI Take-Home Assessment.
