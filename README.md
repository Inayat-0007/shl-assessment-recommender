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

An intelligent, hybrid-search recommendation engine built for the SHL GenAI Take-Home Assessment. It accurately matches open-ended job descriptions or URLs to SHL's catalog of 389 Individual Test Solutions without relying on heavy external vector databases.

## 🔗 Important Assessment Links

- 🐙 **GitHub Repository:** [Inayat-0007/shl-assessment-recommender](https://github.com/Inayat-0007/shl-assessment-recommender)
- 🚀 **Live API Endpoint (Hugging Face):** [inayat05-shl-assessment-recommender.hf.space](https://inayat05-shl-assessment-recommender.hf.space)
- 💻 **Live Frontend UI (Streamlit):** [https://shl-assessment-recommender-4dmfefreyamknkkhsytfhc.streamlit.app/](https://shl-assessment-recommender-4dmfefreyamknkkhsytfhc.streamlit.app/)
- 📄 **Approach Document:** [`docs/approach_document.pdf`](https://github.com/Inayat-0007/shl-assessment-recommender/blob/main/docs/approach_document.pdf)
- 📊 **Predictions CSV:** [`results/results.csv`](https://github.com/Inayat-0007/shl-assessment-recommender/blob/main/results/results.csv)

---

## 🎯 About the Assessment

The goal of this assessment was to build an intelligent recommendation system capable of navigating SHL's extensive product catalog. The system needed to take a natural language query (e.g., "I need a Java coding test under 30 minutes") or a job board URL and output the top 5 to 10 most relevant assessments.

The solution had to strictly adhere to SHL's 16 elimination rules, meaning no heavy orchestrators (like LangChain), no external vector databases (like Pinecone), careful consideration of cloud execution constraints, and robust security measures that still permit automated evaluation bots.

---

## 🧠 The Problem I Faced & How I Implemented It

**The Challenge:**
The biggest hurdle was the classic "Semantic Search vs. Exact Match" dilemma. Given the strict rule against external vector databases, relying purely on Local/In-memory Vector Embeddings (Sentence Transformers) often resulted in the AI misunderstanding exact technological requirements. For example, a search for "Java" would semantically map to "Coffee" or broad "Software Engineering" concepts, dropping the actual "Java" coding tests in rank.

Furthermore, cloud environments (like Hugging Face free tiers) mandate strict memory limits and fast boot requirements, meaning loading giant language models on startup would cause health-check timeouts and crashes.

**The Implementation:**
To solve this, I designed a **Lightweight Hybrid Search Engine** powered entirely by fast, in-memory structures (Numpy/Pandas). 

1. **Lazy Loading:** Model weights (`all-MiniLM-L6-v2`) and pre-computed embeddings are loaded asynchronously in a background thread. This allows the API to pass cloud health checks instantly.
2. **Hybrid Scoring:** Instead of pure semantic search, the engine utilizes a custom mathematical blend: 
   - *50% Semantic Intent* (vector embeddings for broad understanding)
   - *25% Keyword Overlap* (for technical precision)
   - *25% Assessment Name Match* (to strongly boost exact skill matches like "HTML" or "C#")
3. **Smart LLM Expansion:** I layered Google Gemini 2.0 Flash at the very beginning of the pipeline to infer hidden skills from sparse job descriptions before the hybrid search occurs, dramatically increasing my Recall@10 metrics.

---

## ⚙️ Architecture & Tech Stack

| Component | Technology |
|-----------|-----------|
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` |
| **LLM Inference** | Google Gemini 2.0 Flash (Query Expansion) |
| **API Framework** | FastAPI + Uvicorn |
| **Search Engine** | Custom Numpy/Pandas Hybrid Matcher |
| **Security Layer**| `slowapi` (Rate limit), `bleach` (XSS sanitization) |

### Endpoints (Fully compliant with SHL Rubric)
- `GET /health` -> Liveness check
- `POST /recommend` -> JSON body evaluation endpoint
- `GET /recommend` -> Browser-friendly URL parameter endpoint

---

## 🚀 Quick Start (Local Development)

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
```

---

## 👋 A Little About Me

Hi, I'm **Mohammad Inayat Hussain**! I am a passionate and results-driven software engineer who loves solving complex architectural challenges. I thrive at the intersection of traditional software engineering and modern AI capabilities. For this assessment, I focused heavily on ensuring the system wasn't just "functional AI," but also a clean, scalable, mathematically sound, and defensive production-grade application.

Feel free to explore the code, test the API, and see how the SmartMatch Engine works under the hood!
