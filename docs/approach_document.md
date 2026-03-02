# SHL Assessment Recommendation System
# Approach Document

**Author:** Mohammad Inayat Hussain

---

## Page 1: Problem, Architecture & Data Pipeline

### Problem Statement

Given a natural-language job description or hiring query, recommend the most relevant SHL assessments from the product catalog. The system must expose a REST API (GET and POST), return results in a specific JSON format, and produce a CSV of predictions for batch evaluation.

### Architecture

The system follows a straightforward retrieval pipeline:

```
User Query -> Sanitize -> Extract Constraints -> Expand (Gemini LLM)
  -> Embed (Sentence-Transformer) -> Hybrid Score -> Filter & Rank -> Top 10
```

Three-stage scoring:

- **Semantic similarity (50%)** — Cosine similarity between the query embedding and precomputed assessment embeddings using all-MiniLM-L6-v2.
- **Keyword overlap (25%)** — Token intersection ratio between query words and assessment text, after stop-word removal. This catches exact technology names that embeddings miss.
- **Assessment name match (25%)** — Direct token overlap with assessment titles. If a user searches "Java 8", the assessment literally named "Java 8 (New)" gets a strong boost.

Constraint extraction parses the query for:
- Time limits ("under 30 minutes" -> 30)
- Test types ("cognitive" -> A, "personality" -> P, "coding" -> K)
- Job levels ("entry-level", "senior", "executive")
- Adaptive/IRT requirements

### Data Pipeline

1. **Scraping** — Built a requests + BeautifulSoup scraper that paginates through SHL's product catalog (type=1 for Individual Test Solutions). Visited each detail page to extract descriptions, test types, durations, remote/adaptive support, and job levels. Handled SHL's occasional 5xx errors with retry logic and polite delays.

2. **Cleaning** — The raw scraper output needed fixes: test types were space-separated on SHL's site ("C P A B"), durations had missing values (set to -1 as sentinel), and some entries had unicode dashes that broke matching. A cleaning script handles column renaming, deduplication, URL validation, and builds a combined_text column for embeddings.

3. **Final catalog:** 389 Individual Test Solutions with structured metadata.

### Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI + Uvicorn |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (384-dim) |
| LLM | Google Gemini 2.0 Flash (query expansion, optional) |
| Scoring | NumPy + scikit-learn (cosine similarity) |
| Frontend | Streamlit |
| Scraping | requests + BeautifulSoup4 |
| Deployment | Hugging Face Spaces (API) + Streamlit Cloud (frontend) |

---

## Page 2: Evaluation, Decisions & Security

### Evaluation Journey

The primary metric is Mean Recall@10 against the labeled training set (10 queries, each with 1-3 ground truth assessments).

| Iteration | What Changed | Mean Recall@10 |
|-----------|-------------|---------------|
| 1 | Baseline semantic search only | 0.567 |
| 2 | Added fuzzy name matching (initially broken by unicode) | 0.450 |
| 3 | Fixed name resolution + manual override map | 0.600 |
| 4 | Synonym expansion + adaptive boost + name-match scoring | 0.683 |

Key improvements in the final iteration:
- Domain-specific synonym injection before embedding ("behavioral" -> "OPQ personality questionnaire") bridges the vocabulary gap between how users ask and how SHL names its products.
- Adaptive/IRT boost rewards assessments that match explicit adaptive requirements.
- Assessment name matching gives strong weight to exact technology name hits.

### Design Decisions

- **Why hybrid instead of pure semantic?** Pure embedding search missed exact technology names. A query for "Java 8" would return general coding tests because the embedding space does not differentiate well between specific tech stacks. Adding keyword overlap and name matching solved this.
- **Why Gemini is optional?** The system works without an API key by falling back to raw queries. This ensures the evaluation pipeline never fails due to LLM quota limits or network issues.
- **Why no external vector database?** With only 389 assessments, a NumPy matrix fits in memory and cosine similarity runs in under 50ms. Adding Pinecone or ChromaDB would be unnecessary complexity.

### Security Measures

These are implemented and active, not theoretical:
- **Input sanitization** — All queries pass through bleach to strip HTML/script tags, null bytes, and excessive whitespace. Capped at 10,000 characters.
- **SSRF protection** — URL inputs are validated against private/internal IP ranges (127.0.0.1, 169.254.169.254, 10.x, 172.16.x, 192.168.x) to block cloud metadata attacks.
- **Rate limiting** — slowapi enforces 100 requests/minute per IP.
- **Error masking** — Global exception handler returns generic 500 responses with a trace ID. No stack traces or file paths leak to the client.
- **Security headers** — X-Content-Type-Options, X-Frame-Options, Content-Security-Policy, and HSTS on every response.

Production upgrade path: JWT/OAuth2 authentication, CORS restricted to known frontend domains, structured audit logging.

### Known Limitations

- **LinkedIn URL scraping** — LinkedIn blocks automated requests. Pasted LinkedIn URLs fall back to slug-based text extraction, which is less accurate than the full job description.
- **Query ambiguity** — Very short queries like "test" lack enough signal for meaningful ranking. The system still returns results but relevance drops.

### Future Work

- Fine-tune embeddings on SHL-specific assessment-query pairs
- Add re-ranking with cross-encoder models for higher precision
- Implement user feedback loop to improve recommendations over time
