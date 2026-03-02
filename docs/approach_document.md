# SHL Assessment Recommendation System — Approach Document

**Author:** Mohammad Inayat Hussain  

---

## Page 1: Problem, Architecture & Data Pipeline

### Problem Statement

Given a natural-language job description or hiring query, recommend the most relevant SHL assessments from the product catalog. The system must expose a REST API (GET and POST), return results in a specific JSON format, and produce a CSV for batch evaluation.

### Architecture

The system follows a straightforward retrieval pipeline:

```
User Query  →  Sanitize  →  Extract Constraints  →  Expand (Gemini LLM)
     ↓
Embed (Sentence-Transformer)  →  Hybrid Score  →  Filter & Rank  →  Top 10
```

**Three-stage scoring:**

- **Semantic similarity (50%)** — Cosine similarity between the query embedding and precomputed assessment embeddings using `all-MiniLM-L6-v2`.
- **Keyword overlap (25%)** — TF-IDF weighted token overlap between query and assessment text, catches exact technology names the embeddings might miss.
- **Name match (25%)** — Direct token overlap with assessment names, so a query for "Java 8" strongly favors the assessment literally named "Java 8."

**Constraint extraction** parses the query for:
- Time limits ("under 30 minutes" → filter to ≤30 min)
- Test type (coding → K, personality → P, cognitive → A, simulation → S)
- Job level (entry, mid, senior, executive)
- Adaptive/IRT requirement
- Multi-domain detection (e.g., "Java developer who can collaborate" → inject both K and P results)

### Data Pipeline

1. **Scraping** — Built a `requests` + `BeautifulSoup` scraper that paginates through SHL's product catalog (`type=1` for Individual Test Solutions). Visited each detail page to extract descriptions, test types, durations, remote/adaptive support, and job levels. Handled SHL's occasional 5xx errors with retry logic and polite delays.

2. **Cleaning** — The raw scraper output needed fixes: test types were space-separated on SHL's site ("C P A B"), durations had missing values (set to -1 as sentinel), and some entries had unicode dashes that broke matching. A cleaning script handles column renaming, deduplication, URL validation, and builds a `combined_text` column for embeddings.

3. **Final catalog**: **389 Individual Test Solutions** with structured metadata.

### Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI + Uvicorn |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| LLM | Google Gemini 2.0 Flash (query expansion, optional) |
| Scoring | NumPy + scikit-learn (cosine similarity, TF-IDF) |
| Frontend | Streamlit |
| Scraping | requests + BeautifulSoup4 |
| Deployment | Render (API) + Streamlit Cloud (frontend) |

---

## Page 2: Evaluation, Decisions & Security

### Evaluation Journey

The primary metric is **Mean Recall@10** against the labeled training set (10 queries, each with 1-3 ground truth assessments).

| Iteration | What Changed | Mean Recall@10 |
|-----------|-------------|---------------|
| 1 | Baseline semantic search only | 0.567 |
| 2 | Added fuzzy name matching (initially broken by unicode) | 0.450 |
| 3 | Fixed name resolution + manual override map | 0.600 |
| **4** | **Synonym expansion + adaptive boost + name-match scoring** | **0.683** |

**Key improvements in the final iteration:**
- Domain-specific synonym injection before embedding (e.g., "cognitive" → "verify ability reasoning numerical verbal")
- Adaptive/IRT boost (+0.15 for assessments with adaptive support when query mentions it)
- Three-way hybrid scoring instead of pure semantic
- Multi-domain balancing for queries spanning technical + behavioral domains

### Design Decisions

- **Why hybrid instead of pure semantic?** Pure embedding search missed exact technology names. A query for "Java 8" would return general coding tests because the embedding space doesn't differentiate well between specific tech stacks. Adding keyword overlap and name matching solved this.
- **Why Gemini LLM for expansion?** Short queries like "Java test" don't carry enough signal. Gemini adds relevant context ("enterprise application development, object-oriented programming, Spring framework") that improves semantic recall. It's optional — the system falls back gracefully without an API key.
- **Why not a vector database?** With 389 assessments, in-memory NumPy cosine similarity runs in <50ms. A vector DB would add deployment complexity with no measurable benefit at this scale.

### Security Measures

These are implemented and active, not theoretical:

- **Input sanitization** — All queries pass through `bleach` to strip HTML/script tags, null bytes, and excessive whitespace. Capped at 10,000 characters.
- **SSRF protection** — URL inputs are validated against private IP ranges (10.x, 172.16.x, 192.168.x, 169.254.x, localhost). Blocks cloud metadata endpoints.
- **Rate limiting** — 100 requests/minute per IP via `slowapi`. Returns 429 with appropriate headers.
- **Request size limiting** — Middleware rejects payloads over 1MB (413 response).
- **Security headers** — `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `X-XSS-Protection` on all responses.
- **Error masking** — Unhandled exceptions return generic messages with a tracking UUID. No tracebacks, file paths, or library names in error responses.
- **CORS** — Open (`*`) during evaluation per SHL requirements.

**Production upgrade path:** JWT/OAuth2 authentication, CORS restricted to known frontend domains, HTTPS enforcement, structured audit logging.

### Known Limitations

- **LinkedIn URL scraping** — LinkedIn blocks automated requests. Pasted LinkedIn URLs fall back to slug-based text extraction, which is less accurate than the full job description.
- **Query 10 recall** — The ground truth for "adaptive/IRT graduate test" maps to assessments marked `adaptive_support=No` in the catalog, causing a mismatch. The engine returns semantically correct results but different URLs.
- **Duration gaps** — ~15% of assessments have unknown durations (set to -1). Time-constrained queries may miss relevant untimed assessments.

### Future Work

- Fine-tune embeddings on SHL-specific assessment-query pairs
- Add re-ranking with a cross-encoder model for top-k refinement
- User feedback loop to improve recommendations over time
- Support for Pre-Packaged Job Solutions (currently filtered to Individual only)
