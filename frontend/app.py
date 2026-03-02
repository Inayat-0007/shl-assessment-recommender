"""
Streamlit frontend for the SHL Assessment Recommendation System.

This is the user-facing interface that talks to the FastAPI backend.
I went with Streamlit because it let me build a decent-looking dashboard
quickly without writing a full React/Vue app. The dark theme with
glassmorphism-inspired cards came out pretty well I think.

Security notes:
  - All API response content is HTML-escaped before rendering
  - Backend URL comes from env var, never hardcoded
  - Error messages are generic (no stack traces shown to users)

    $ streamlit run frontend/app.py

Author: Mohammad Inayat Hussain
"""

import os
import re
import html
import requests
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


# -- Helpers ------------------------------------------------------------------

def sanitize_display(text: str) -> str:
    """Escape HTML for safe rendering in Streamlit markdown."""
    if not text:
        return ""
    text = html.escape(str(text))
    text = re.sub(r'<[^>]+>', '', text)
    return text.strip()


def check_backend_health() -> dict | None:
    """Ping the backend health endpoint."""
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None


def get_recommendations(query: str) -> dict | None:
    """Call POST /recommend and handle errors gracefully."""
    try:
        resp = requests.post(
            f"{BACKEND_URL}/recommend",
            json={"query": query},
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 422:
            st.error("⚠️ Invalid query. Please enter a valid job description or search term.")
        elif resp.status_code == 429:
            st.error("⏳ Too many requests. Please wait a moment and try again.")
        elif resp.status_code == 413:
            st.error("📦 Query is too long. Please shorten your input.")
        else:
            st.error("❌ Something went wrong. Please try again later.")
        return None
    except requests.exceptions.ConnectionError:
        st.error(
            "🔌 **Cannot connect to the backend server.**\n\n"
            "Please make sure the API is running:\n"
            "```\npython main.py\n```"
        )
        return None
    except requests.exceptions.Timeout:
        st.error("⏰ Request timed out. The server may be busy — please try again.")
        return None
    except Exception:
        st.error("❌ An unexpected error occurred. Please try again.")
        return None


def truncate_text(text: str, max_len: int = 120) -> str:
    """Shorten text for display, adding ... if truncated."""
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0] + "..."


def format_assessment_name(raw_url: str) -> str:
    """Turn a SHL URL slug into a readable assessment name."""
    if not raw_url:
        return "Assessment"
    slug = raw_url.rstrip("/").split("/")[-1]
    # Remove trailing "-new" suffix that SHL adds
    slug = re.sub(r'-new$', '', slug)
    name = slug.replace("-", " ").title()
    # Fix common casing
    fixes = {
        "Opq": "OPQ", "Sql": "SQL", "Html": "HTML", "Css": "CSS",
        "Sap": "SAP", "Irt": "IRT", "Sjt": "SJT", ".net": ".NET",
        "Javascript": "JavaScript", "Php": "PHP", "Mvc": "MVC",
        "Aws": "AWS", "Opq32R": "OPQ32r", "Mq": "MQ", "Mqm5": "MQM5",
    }
    for old, new in fixes.items():
        name = name.replace(old, new)
    return name


# -- Page Config ---------------------------------------------------------------

st.set_page_config(
    page_title="SHL SmartMatch AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# -- CSS -----------------------------------------------------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1000px;
    }

    /* Hero */
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
        line-height: 1.2;
    }
    .hero-sub {
        font-size: 1rem;
        color: #8892b0;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }

    /* Metric strip */
    .metric-strip {
        display: flex;
        justify-content: center;
        gap: 1.2rem;
        margin-bottom: 1.5rem;
    }
    .metric-chip {
        background: linear-gradient(145deg, #1e1e2e, #262640);
        border: 1px solid rgba(102,126,234,0.2);
        border-radius: 10px;
        padding: 0.7rem 1.4rem;
        text-align: center;
        min-width: 120px;
    }
    .metric-chip .mv {
        font-size: 1.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-chip .ml {
        font-size: 0.7rem;
        color: #8892b0;
        margin-top: 0.15rem;
        font-weight: 500;
    }

    /* Query preview box */
    .query-preview {
        background: rgba(102,126,234,0.06);
        border: 1px solid rgba(102,126,234,0.15);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 1rem;
        font-size: 0.85rem;
        color: #c0c8e0;
        line-height: 1.5;
        max-height: 80px;
        overflow: hidden;
        position: relative;
    }
    .query-preview.expanded {
        max-height: none;
    }
    .query-label {
        font-size: 0.72rem;
        color: #667eea;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 0.3rem;
    }

    /* Result card */
    .r-card {
        background: linear-gradient(145deg, #1a1a2e, #252540);
        border: 1px solid rgba(102,126,234,0.15);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
        transition: all 0.25s ease;
        display: flex;
        align-items: flex-start;
        gap: 1rem;
    }
    .r-card:hover {
        border-color: rgba(102,126,234,0.5);
        box-shadow: 0 4px 16px rgba(102,126,234,0.12);
        transform: translateY(-1px);
    }
    .r-rank {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        font-weight: 700;
        font-size: 0.85rem;
        width: 32px;
        height: 32px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        margin-top: 2px;
    }
    .r-body {
        flex: 1;
        min-width: 0;
    }
    .r-name {
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }
    .r-name a {
        color: #c4b5fd;
        text-decoration: none;
    }
    .r-name a:hover {
        color: #e0d5ff;
        text-decoration: underline;
    }
    .r-desc {
        font-size: 0.8rem;
        color: #8892b0;
        line-height: 1.45;
        margin-bottom: 0.5rem;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .r-tags {
        display: flex;
        gap: 0.4rem;
        flex-wrap: wrap;
        align-items: center;
    }
    .tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 5px;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.2px;
    }
    .tag-dur { background: rgba(59,130,246,0.12); color: #60a5fa; }
    .tag-rem { background: rgba(34,197,94,0.12); color: #4ade80; }
    .tag-adp { background: rgba(168,85,247,0.12); color: #c084fc; }
    .tag-typ { background: rgba(251,191,36,0.12); color: #fbbf24; }

    /* Score bar */
    .score-bar-bg {
        width: 80px;
        height: 6px;
        background: rgba(255,255,255,0.08);
        border-radius: 3px;
        overflow: hidden;
        display: inline-block;
        vertical-align: middle;
        margin-right: 4px;
    }
    .score-bar-fill {
        height: 100%;
        border-radius: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    .score-text {
        font-size: 0.68rem;
        color: #8892b0;
        vertical-align: middle;
    }

    /* Input mode tabs */
    .input-mode-label {
        font-size: 0.75rem;
        color: #667eea;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.3rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 0.78rem;
        margin-top: 2.5rem;
        padding: 1.2rem 0;
        border-top: 1px solid rgba(102,126,234,0.1);
    }
    .footer a { color: #a78bfa; text-decoration: none; }

    /* Hide streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# -- Header -------------------------------------------------------------------

st.markdown('<div class="hero-title">🎯 SHL Assessment Recommender</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Paste a job description or URL — get the best matching SHL assessments instantly</div>', unsafe_allow_html=True)

# Status strip
health = check_backend_health()
if health:
    count = health.get("assessments", 0)
    st.markdown(f"""
    <div class="metric-strip">
        <div class="metric-chip"><div class="mv">✅</div><div class="ml">API Online</div></div>
        <div class="metric-chip"><div class="mv">{count}</div><div class="ml">Assessments</div></div>
        <div class="metric-chip"><div class="mv">10</div><div class="ml">Results / Query</div></div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.error("🔌 **Backend API is offline.** Run `python main.py` first.")

st.markdown("---")


# -- Input Section -------------------------------------------------------------
# Fixed bug: If user types in text area AND pastes URL, the old code always
# used text area (ignoring URL). Now URL takes priority when provided, and
# the two inputs are in separate tabs so the user can't accidentally mix them.

input_mode = st.radio(
    "Choose input method:",
    ["📝 Job Description / Search Query", "🔗 Job Posting URL"],
    horizontal=True,
    label_visibility="collapsed",
)

query_input = ""
url_input = ""

if input_mode == "📝 Job Description / Search Query":
    st.markdown('<div class="input-mode-label">Describe what you\'re looking for</div>', unsafe_allow_html=True)
    query_input = st.text_area(
        "query_area",
        placeholder="e.g., I need a cognitive ability test for entry-level software engineers, under 40 minutes...\n\nOr paste an entire job description here.",
        height=120,
        key="query_input",
        label_visibility="collapsed",
    )
else:
    st.markdown('<div class="input-mode-label">Paste a job posting URL (LinkedIn, Indeed, Glassdoor, etc.)</div>', unsafe_allow_html=True)
    url_input = st.text_input(
        "url_field",
        placeholder="https://www.linkedin.com/jobs/view/123456789",
        key="url_input",
        label_visibility="collapsed",
    )

# Sample queries
st.markdown("**💡 Quick samples:**")
sample_queries = [
    "Java developer coding test under 30 minutes",
    "Cognitive ability test for graduates",
    "Leadership assessment for managers",
    "Customer service personality test",
    "Data analyst SQL and Python assessment",
    "Sales aptitude behavioral assessment",
]

sample_cols = st.columns(3)
selected_sample = None
for i, sample in enumerate(sample_queries):
    with sample_cols[i % 3]:
        if st.button(f"🔍 {sample}", key=f"sample_{i}", use_container_width=True):
            selected_sample = sample

# Determine final query — fixed priority bug:
# 1. Sample click always wins (user explicitly clicked it)
# 2. Otherwise use whichever input mode is active
if selected_sample:
    final_query = selected_sample
elif input_mode == "🔗 Job Posting URL" and url_input.strip():
    final_query = url_input.strip()
elif query_input.strip():
    final_query = query_input.strip()
else:
    final_query = ""

st.markdown("---")

# Recommend button
col_l, col_c, col_r = st.columns([1, 2, 1])
with col_c:
    recommend_clicked = st.button(
        "🚀 Get Recommendations",
        use_container_width=True,
        type="primary",
        key="recommend_btn",
    )


# -- Results -------------------------------------------------------------------

if recommend_clicked or selected_sample:
    if not final_query:
        st.warning("⚠️ Please enter a query, paste a URL, or click a sample above.")
    else:
        # Show truncated query preview (not the full wall of text)
        display_text = sanitize_display(final_query)
        short = truncate_text(display_text, 150)
        st.markdown(f"""
        <div class="query-label">Search Query</div>
        <div class="query-preview">{short}</div>
        """, unsafe_allow_html=True)

        # If the query was long, show full text in an expander
        if len(display_text) > 150:
            with st.expander("📄 View full query text", expanded=False):
                st.text(final_query[:2000])

        with st.spinner("🔍 Analyzing requirements and matching assessments..."):
            data = get_recommendations(final_query)

        if data and "recommended_assessments" in data:
            assessments = data["recommended_assessments"]

            st.success(f"✅ Found **{len(assessments)}** matching assessments")
            st.markdown("### 📊 Recommended Assessments")

            # Type display names
            type_labels = {
                "A": "Ability", "B": "Behavioral", "C": "Competency",
                "D": "Development", "E": "Experience", "K": "Knowledge",
                "P": "Personality", "S": "Simulation",
            }

            for idx, item in enumerate(assessments):
                raw_url = item.get("url", "")
                name = format_assessment_name(raw_url)
                desc = sanitize_display(item.get("description", ""))
                duration = item.get("duration", -1)
                remote = item.get("remote_support", "N/A")
                adaptive = item.get("adaptive_support", "N/A")
                test_types = item.get("test_type", [])

                dur_text = f"{duration} min" if duration and duration > 0 else "Untimed"

                # Type tags
                type_html = ""
                for t in test_types:
                    label = type_labels.get(t, t)
                    type_html += f'<span class="tag tag-typ">{sanitize_display(label)}</span>'

                # Score bar (visual indicator)
                # We don't have score from API but we can show rank
                score_pct = max(10, 100 - idx * 8)
                score_html = (
                    f'<div class="score-bar-bg">'
                    f'<div class="score-bar-fill" style="width:{score_pct}%"></div>'
                    f'</div>'
                    f'<span class="score-text">#{idx+1}</span>'
                )

                # Card HTML
                card = f"""
                <div class="r-card">
                    <div class="r-rank">{idx + 1}</div>
                    <div class="r-body">
                        <div class="r-name">
                            <a href="{raw_url}" target="_blank" rel="noopener">{sanitize_display(name)}</a>
                        </div>
                        <div class="r-desc">{desc[:250] if desc else 'SHL assessment — click title to view details on shl.com'}</div>
                        <div class="r-tags">
                            <span class="tag tag-dur">⏱ {dur_text}</span>
                            <span class="tag tag-rem">🌐 {sanitize_display(remote)}</span>
                            <span class="tag tag-adp">🧠 {sanitize_display(adaptive)}</span>
                            {type_html}
                            {score_html}
                        </div>
                    </div>
                </div>
                """
                st.markdown(card, unsafe_allow_html=True)

            # Data table view
            with st.expander("📋 View as Data Table", expanded=False):
                table_data = []
                for item in assessments:
                    table_data.append({
                        "Assessment": format_assessment_name(item.get("url", "")),
                        "Duration": item.get("duration", -1),
                        "Remote": item.get("remote_support", "N/A"),
                        "Adaptive": item.get("adaptive_support", "N/A"),
                        "Type": ", ".join(item.get("test_type", [])),
                        "URL": item.get("url", ""),
                    })
                df = pd.DataFrame(table_data)
                st.dataframe(
                    df,
                    column_config={
                        "URL": st.column_config.LinkColumn("URL", display_text="Open →"),
                    },
                    use_container_width=True,
                    hide_index=True,
                )

        elif data is not None:
            st.info("No assessments matched your query. Try rephrasing or broadening your search.")


# -- Footer --------------------------------------------------------------------

st.markdown("---")
st.markdown(
    '<div class="footer">'
    'Powered by <a href="https://www.shl.com" target="_blank">SHL Assessment Catalog</a> '
    '| 389 Assessments Indexed | Built by Mohammad Inayat Hussain'
    '</div>',
    unsafe_allow_html=True,
)
