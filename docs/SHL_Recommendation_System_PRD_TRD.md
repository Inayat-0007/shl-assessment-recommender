# SHL SmartMatch AI — Comprehensive PRD, TRD, and System Architecture

**Document Version:** 1.0  
**Project:** SHL Assessment Recommendation System  
**Author:** Mohammad Inayat Hussain
**Prepared For:** SHL Evaluation Process  

---

## 1. Executive Summary & Alignment with SHL Requirements

The **SHL SmartMatch AI** system is a complete, end-to-end intelligent recommendation engine designed to match natural language queries or Job Description (JD) URLs with the most relevant SHL Individual Test Solutions.

**Company Alignment Summary:**
- **Problem Solved:** Time-consuming manual keyword searches for assessments are replaced by an AI-powered semantic hybrid search engine.
- **Scraping Integrity:** Built a dynamic scraper (`scraper/scrape_catalog.py`) retrieving **389 Individual Test Solutions** directly from the official SHL catalog, excluding pre-packaged solutions as strictly required.
- **AI/LLM Integration:** Utilizes Google Gemini 2.0 Flash for semantic query expansion (understanding implied skills and domain context) and Sentence-Transformers (`all-MiniLM-L6-v2`) for rich vector embeddings, fulfilling the Modern LLM & Retrieval-Augmented Generation (RAG) requirement.
- **Data & Evaluation:** Leverages the provided `train_set.csv` (10 queries) to iterate on prompt and retrieval logic and `test_set.csv` (9 queries) to generate final predictions (`results/results.csv`) optimized for **Mean Recall@10**.
- **Multi-domain Balancing:** Implements custom balancing logic ensuring queries requiring both technical ("K") and behavioral ("P") testing return a balanced mix of assessments.
- **Security & API Structure:** Exposes exact endpoints (`GET /health`, `POST /recommend`) required by the SHL auto-evaluation pipeline, layered with robust security (Rate Limiting, XSS protection, SSRF URL protection) designed *not* to block the automated SHL evaluators.

---

## 2. Product Requirements Document (PRD)

### 2.1 Problem Statement
Recruiters and hiring managers spend excessive time navigating large catalogs to find the right assessments for specific roles. A keyword-only approach is insufficient for complex hybrid roles (e.g., "Java developer who is good at collaboration").

### 2.2 Product Goals
Build a sophisticated web application and API that accepts a job description text or a URL and intelligently returns 5 to 10 highly relevant SHL assessments.

### 2.3 Key Features
1. **Intelligent Text Recommendation:** Accepts natural language queries and returns prioritized SHL catalog links.
2. **Context-Aware URL Parsing (RAG Edge-case):** Accepts external JD URLs (e.g., LinkedIn, Glassdoor). Features a smart fallback that extracts meta tags (`og:title`, `<title>`) or URL slugs to bypass enterprise bot-protection walls.
3. **Multi-Domain Intelligence:** Automatically detects when a user is asking for mixed skillsets (e.g., Cognitive + Tech) and balances the final response.
4. **Interactive Dashboard:** A Streamlit frontend (`frontend/app.py`) allowing manual testing, URL submission, and visual data table exploration.

### 2.4 User Flow
1. User enters a query (text or URL) into the UI or via API.
2. If URL, system securely fetches the remote content (handling bot-walls via meta-tag extraction).
3. System sanitizes input and uses Gemini LLM to expand the query with industry-standard synonyms.
4. System semantically matches the expanded query against the embedded 389 SHL assessments.
5. Filter constraints (time limits, test types) are applied.
6. System outputs top 5-10 matched SHL Assessment URLs.

---

## 3. Technical Requirements Document (TRD)

### 3.1 Technology Stack & Libraries
- **Backend Framework:** FastAPI, Uvicorn (Fast, async, built-in validation)
- **Frontend Framework:** Streamlit (Rapid UI prototyping)
- **AI & ML (The "RAG" Stack):**
  - `sentence-transformers`: Local vector embedding (`all-MiniLM-L6-v2`) for fast cosine similarity.
  - `google-generativeai` (Gemini): LLM used to expand user queries contextually.
  - `scikit-learn`: For rapid cosine similarity computation.
- **Data Engineering:** `pandas`, `numpy` (In-memory database for catalog filtering)
- **Scraping & Parsing:** `BeautifulSoup4`, `requests`, `lxml`
- **Security:** `slowapi` (Rate limiting), `bleach` (XSS sanitization)

### 3.2 System Architecture (Data Pipeline & RAG)
The system operates on an automated RAG-like pipeline without a heavy external vector database, utilizing Pandas and fast numpy matrices for extreme low-latency inference:

1. **Ingestion (Scraping):** The catalog is crawled, cleaned, resulting in `data/shl_catalog_clean.csv`.
2. **Embedding (Index):** On API startup, `SentenceTransformer` embeds all 389 descriptions into a (389, 384) matrix.
3. **Retrieval (Engine Logic):**
   - **Query Expansion (Gemini):** Takes "Java" and expands to "Java, Object-oriented, Spring, Backend".
   - **Semantic Search:** Cosine similarity between the expanded query embedding and catalog embeddings.
   - **Keyword Match:** Direct token overlap boosting for exact terminology (e.g., precise technology names).
   - **Constraint Filtering:** Hard filters the dataframe based on parsed constraints.

### 3.3 Security Implementation (Enterprise Grade)
The API is hardened against attacks while maintaining availability for SHL's automated evaluators:
1. **Rate Limiting:** `slowapi` restricts to 100 requests/minute per IP. Headers (`X-RateLimit-Limit`) expose this safely.
2. **Input Sanitization (XSS):** `bleach` strips all HTML/JS from the input payloads.
3. **Payload Size Restriction:** Middleware cuts off requests > 1MB to prevent RAM exhaustion.
4. **SSRF (Server-Side Request Forgery) Protection:** When user provides a URL to scrape, the engine blocks internal/private IPs (e.g., `127.0.0.1`, `169.254.169.254` AWS metadata).
5. **Security Headers:** Added `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, and strict CSP.
6. **Error Masking:** Global exception handler returns generic `500 Internal server error` with a unique UUID trace ID. NO stack traces or local Python paths (`C:\Users\...`) are leaked to the client.

### 3.4 APIs & Endpoints
Strict compliance with SHL documentation:
- **`GET /health`** -> Returns `{"status": "healthy", "engine": "loaded", "assessments": 389}`
- **`POST /recommend`** -> Accepts `{"query": "string"}` -> Returns `{"recommended_assessments": [...]}`
- **`GET /recommend?query=...`** -> Browser friendly version of the POST endpoint.

---

## 4. Codebase Structure & File Explanations

```text
SHL_SmartMatch_AI/
├── main.py                     # Entry point (Uvicorn server)
├── requirements.txt            # Locked dependencies (bleach, fastapi, sentence-transformers, etc.)
├── generate_results.py         # SHL requirement: Generates predictions for the 9 test queries
├── Procfile                    # Deployment configuration (Render.com)
├── data/
│   ├── shl_catalog_clean.csv   # The 389 scraped independent test solutions
│   ├── train_set.csv           # 10 labeled queries used for tuning
│   └── test_set.csv            # 9 unlabeled queries for final prediction
├── docs/
│   └── SHL_Recommendation_System_PRD_TRD.md  (This file)
├── frontend/
│   └── app.py                  # Streamlit dashboard interface
├── results/
│   └── results.csv             # FINAL Output: 9 queries x 10 Assessment_URLs
├── scraper/
│   ├── scrape_catalog.py       # Beautifulsoup logic targeting shl.com/products/product-catalog
│   └── clean_catalog.py        # Deduplication and duration normalization
└── src/
    ├── api.py                  # FastAPI routers, Middleware, CORS, Error Handlers
    ├── engine.py               # The Core RAG Engine (Embeddings, Gemini, Hybrid Search)
    ├── evaluate.py             # Computes Mean Recall@K against the train_set.csv
    └── utils.py                # Pure mathematical and security helpers (URL verification, Santization)
```

---

## 5. Deployment Guide (GitHub & Cloud)

### 5.1 Local Execution
```bash
# Activation & Installs
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Run Backend
python main.py

# Run Frontend
streamlit run frontend/app.py
```

### 5.2 GitHub Publishing
1. Ensure `.env` is listed in `.gitignore` (which includes `GEMINI_API_KEY`).
2. Only commit `.env.example`.
3. Repo should be set to Public (or Private and shared with SHL evaluators).

### 5.3 Cloud Deployment
**Backend (Render.com):**
- Connect GitHub repo to Render Web Service.
- Build Command: `pip install -r requirements.txt`
- Start Command: Uses the `Procfile` (`web: uvicorn src.api:app --host 0.0.0.0 --port $PORT`)
- Add Environment Variables (`GEMINI_API_KEY`).

**Frontend (Streamlit Cloud):**
- Connect Repo to Streamlit.
- Target `frontend/app.py`.
- Add Environment variable: `BACKEND_URL` pointing to the Render API URL.

---

## 6. Postman Testing & SHL Verification Summary

An automated audit (`audit_shl.py`) was executed simulating Postman API hits against the environment matching SHL's evaluation pipeline:

| Category | Status | Details |
| :--- | :---: | :--- |
| **Data Ingestion** | PASSED | 389 items scraped; 0 "Pre-Packaged" solutions. |
| **API Endpoints** | PASSED | `/health` acts successfully. `POST /recommend` matches exact JSON layout requested. |
| **Security Handling** | PASSED | XSS injected payloads sanitized. Rate limits properly throw 429. Internal stack traces successfully masked. |
| **RAG/LLM Efficacy** | PASSED | Multi-domain balancing correctly pairs K (Knowledge) and P (Personality) assessments based on Gemini's logical inferences. |
| **Test Set CSV** | PASSED | Generated `results.csv` contains exactly 'Query' and 'Assessment_url' columns; 90 distinct recommendations formatted correctly. |

**Final Conclusion:** The architecture directly solves the SHL take-home challenge accurately, defensively, and efficiently utilizing minimal overhead while maximizing evaluation rubric scores.
