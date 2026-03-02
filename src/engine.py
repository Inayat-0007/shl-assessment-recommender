"""
SHL Assessment Recommendation Engine

This is the core of the whole system. It takes a natural language query
(e.g. "I need a Java test under 30 minutes for mid-level devs") or a
URL pointing to a job description, and returns the top 5-10 most
relevant SHL assessments.

I went with a hybrid approach because pure semantic search misses exact
technology name matches (user says "Java" but embedding thinks "coffee"),
and pure keyword search can't understand intent. The combination works well.

Pipeline:
  1. Gemini LLM expands the query with implied skills
  2. Sentence-transformers encodes it to a 384-dim vector
  3. Cosine similarity gives the semantic score
  4. Token overlap gives the keyword score
  5. Assessment name matching gives a tech-precision boost
  6. Weighted combination -> ranked list
  7. Constraint filters narrow it down (time, type, level)
  8. Multi-domain balancing ensures mixed K+P results when needed

Author: Mohammad Inayat Hussain
"""

import os
import re
import html
import ipaddress
import logging
import warnings
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils import sanitize_input, validate_url as utils_validate_url, mask_sensitive, is_url as utils_is_url

warnings.filterwarnings("ignore", category=FutureWarning)  # transformers spams FutureWarnings
load_dotenv()

# Gemini is optional - the system works fine without it (just less smart expansion)
GEMINI_AVAILABLE = False
try:
    import google.generativeai as genai
    _gemini_key = os.getenv("GEMINI_API_KEY", "")
    if _gemini_key and _gemini_key != "your-actual-key-here":
        genai.configure(api_key=_gemini_key)
        GEMINI_AVAILABLE = True
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("shl_engine")

# Paths
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CATALOG_FILE = DATA_DIR / "shl_catalog_clean.csv"

# I picked all-MiniLM-L6-v2 because it's fast, small (~80MB), and gives
# surprisingly good results for short text similarity. Larger models
# like mpnet were marginally better but 3x slower on my machine.
MODEL_NAME = "all-MiniLM-L6-v2"

# After a lot of experimentation, 50/25/25 gave the best Recall@10.
# Pure semantic (70/30/0) missed exact tech name matches too often.
SEMANTIC_WEIGHT = 0.50
KEYWORD_WEIGHT = 0.25
NAME_WEIGHT = 0.25

MAX_QUERY_LEN = 10_000       # chars
MAX_JD_SIZE = 5 * 1024 * 1024  # 5 MB
JD_FETCH_TIMEOUT = 10         # seconds
MIN_RESULTS = 5
MAX_RESULTS = 10

# Maps common query keywords to SHL test type codes.
# This drives both the type detection and the constraint filtering.
KEYWORD_TYPE_MAP: dict[str, list[str]] = {
    # Cognitive / Ability
    "cognitive": ["A"],
    "aptitude": ["A"],
    "ability": ["A"],
    "reasoning": ["A"],
    "numerical": ["A", "K"],
    "verbal": ["A", "K"],
    "inductive": ["A"],
    "deductive": ["A"],
    "logical": ["A"],
    # Knowledge / Technical
    "knowledge": ["K"],
    "technical": ["K"],
    "coding": ["K"],
    "programming": ["K"],
    "software": ["K"],
    "developer": ["K"],
    "java": ["K"],
    "python": ["K"],
    "html": ["K"],
    "css": ["K"],
    "javascript": ["K"],
    "frontend": ["K"],
    "web": ["K"],
    ".net": ["K"],
    "sql": ["K"],
    "excel": ["K"],
    "microsoft": ["K"],
    "sap": ["K"],
    "oracle": ["K"],
    "it skills": ["K"],
    # Personality / Behavioral
    "personality": ["P"],
    "behavioral": ["P", "B"],
    "behaviour": ["P", "B"],
    "behavior": ["P", "B"],
    "motivation": ["P"],
    "opq": ["P"],
    "leadership": ["P", "C"],
    "interpersonal": ["P"],
    "teamwork": ["P"],
    "team skills": ["P"],
    "collaborate": ["P"],
    "collaboration": ["P"],
    "soft skills": ["P"],
    "people skills": ["P"],
    "communication": ["P"],
    "work with": ["P"],
    "cultural fit": ["P"],
    # Competency
    "competency": ["C"],
    "competencies": ["C"],
    "managerial": ["C"],
    "management": ["C"],
    # Simulation
    "simulation": ["S"],
    "customer service": ["S"],
    "call center": ["S"],
    "contact center": ["S"],
    # Biodata / SJT
    "situational": ["B"],
    "judgement": ["B"],
    "judgment": ["B"],
    "biodata": ["B"],
    "sjt": ["B"],
    # Development / 360
    "360": ["D"],
    "development": ["D"],
    "feedback": ["D"],
    # Exercises
    "exercise": ["E"],
    "assessment center": ["E"],
    "role play": ["E"],
}

# Job level keywords
JOB_LEVEL_KEYWORDS: dict[str, list[str]] = {
    "entry": ["entry", "entry-level", "junior", "graduate", "intern", "trainee", "apprentice"],
    "mid": ["mid", "mid-level", "mid-professional", "professional", "associate", "experienced"],
    "senior": ["senior", "lead", "principal", "supervisor", "manager"],
    "executive": ["executive", "director", "vp", "c-level", "chief", "president"],
}

# Test type full-name mapping (for logging/debug)
TEST_TYPE_NAMES = {
    "A": "Ability & Aptitude",
    "B": "Biodata & SJT",
    "C": "Competency",
    "D": "Development & 360",
    "E": "Exercise",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavioral",
    "S": "Simulation",
}

# Gemini prompt for query expansion
GEMINI_EXPANSION_PROMPT = (
    "You are an HR assessment expert. Extract key skills, competencies, "
    "and job requirements from this query. Infer implied skills that are "
    "not explicitly stated. Return a comma-separated list of relevant "
    "skills and competencies. Be concise. Do not include explanations.\n\n"
    "Query: {query}\n\nSkills and competencies:"
)


class AssessmentEngine:
    """
    Core recommendation engine for SHL assessments.

    Usage:
        engine = AssessmentEngine()
        results = engine.recommend("I need a Java test under 30 minutes")
    """

    def __init__(self) -> None:
        """Load catalog data, initialize model, and pre-compute embeddings."""
        log.info("Initializing AssessmentEngine...")

        # ── Load catalog ──
        if not CATALOG_FILE.exists():
            raise FileNotFoundError(
                f"Catalog not found at {CATALOG_FILE}. "
                "Run scraper/clean_catalog.py first."
            )

        self.df = pd.read_csv(CATALOG_FILE)
        log.info(f"  Loaded {len(self.df)} assessments from catalog")

        # Ensure combined_text exists and has no NaN
        if "combined_text" not in self.df.columns:
            raise ValueError("Catalog missing 'combined_text' column. Re-run clean_catalog.py.")
        self.df["combined_text"] = self.df["combined_text"].fillna("").astype(str)

        # Ensure test_type is string
        self.df["test_type"] = self.df["test_type"].fillna("").astype(str)

        # Pre-split test types for fast filtering
        self.df["test_type_list"] = self.df["test_type"].apply(
            lambda x: [t.strip() for t in x.split(",") if t.strip()]
        )

        # Ensure duration is integer
        self.df["duration"] = pd.to_numeric(self.df["duration"], errors="coerce").fillna(-1).astype(int)

        # ── Load sentence-transformers model ──
        log.info(f"  Loading model: {MODEL_NAME}...")
        self.model = SentenceTransformer(MODEL_NAME)
        log.info(f"  Model loaded successfully")

        # ── Pre-compute embeddings ──
        log.info(f"  Computing embeddings for {len(self.df)} assessments...")
        texts = self.df["combined_text"].tolist()
        self.embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        log.info(f"  Embeddings shape: {self.embeddings.shape}")

        # ── Pre-compute keyword sets for each assessment ──
        self.keyword_sets = [
            set(text.lower().split()) for text in texts
        ]

        # ── Gemini LLM setup ──
        self._gemini_model = None
        self._query_cache: dict[str, str] = {}  # Simple in-memory cache
        if GEMINI_AVAILABLE:
            try:
                self._gemini_model = genai.GenerativeModel("gemini-2.0-flash")
                log.info(f"  Gemini LLM: ENABLED (key: {mask_sensitive(os.getenv('GEMINI_API_KEY', ''))}")
            except Exception as e:
                log.warning(f"  Gemini LLM: DISABLED (init error: {e})")
        else:
            log.info("  Gemini LLM: DISABLED (no valid API key or google-generativeai not installed)")

        log.info(f"  AssessmentEngine ready — {len(self.df)} assessments indexed\n")


    def sanitize_query(self, query: str) -> str:
        """
        Sanitize user input using bleach-based sanitization from utils.

        Args:
            query: Raw user input string.

        Returns:
            Cleaned query string.
        """
        return sanitize_input(query)


    def extract_time_limit(self, query: str) -> Optional[int]:
        """
        Extract time constraint from query.

        Handles:
            "under 30 minutes", "max 40 min", "less than 1 hour",
            "within 20 mins", "no more than 15 minutes"

        Args:
            query: User query string.

        Returns:
            Time limit in minutes, or None if not found.
        """
        query_lower = query.lower()

        # Pattern: "under/less than/max/within/no more than X minutes/min/mins"
        patterns = [
            r"(?:under|less\s+than|max(?:imum)?|within|no\s+more\s+than|at\s+most|up\s+to)\s+(\d+)\s*(?:minutes?|mins?|min)\b",
            r"(\d+)\s*(?:minutes?|mins?|min)\s+(?:or\s+less|maximum|max|limit)",
            r"(?:under|less\s+than|max(?:imum)?|within)\s+(\d+)\s*(?:hours?|hrs?)\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                value = int(match.group(1))
                # Check if this was hours pattern
                if "hour" in pattern or "hrs" in pattern:
                    value *= 60
                return value

        return None

    def extract_test_types(self, query: str) -> Optional[list[str]]:
        """
        Detect test type preferences from query keywords.

        Args:
            query: User query string.

        Returns:
            List of test type letters (e.g., ["K", "P"]) or None.
        """
        query_lower = query.lower()
        detected_types: set[str] = set()

        for keyword, type_letters in KEYWORD_TYPE_MAP.items():
            # Use word boundary matching for short keywords
            if len(keyword) <= 3:
                if re.search(rf"\b{re.escape(keyword)}\b", query_lower):
                    detected_types.update(type_letters)
            else:
                if keyword in query_lower:
                    detected_types.update(type_letters)

        return sorted(detected_types) if detected_types else None

    def extract_job_level(self, query: str) -> Optional[str]:
        """
        Detect job level from query keywords.

        Args:
            query: User query string.

        Returns:
            Job level string (entry/mid/senior/executive) or None.
        """
        query_lower = query.lower()

        for level, keywords in JOB_LEVEL_KEYWORDS.items():
            for keyword in keywords:
                if re.search(rf"\b{re.escape(keyword)}\b", query_lower):
                    return level

        return None

    def extract_adaptive_requirement(self, query: str) -> bool:
        """
        Detect if query requires adaptive/IRT-enabled assessments.

        Triggers on: adaptive, IRT, interactive, computer adaptive testing.

        Returns:
            True if query mentions adaptive/IRT.
        """
        q = query.lower()
        adaptive_keywords = [
            "adaptive", "irt", "item response theory",
            "computer adaptive", "cat test", "interactive test",
            "verify interactive",
        ]
        return any(kw in q for kw in adaptive_keywords)

    def expand_query_synonyms(self, query: str) -> str:
        """
        Inject domain-specific synonyms into the query to bridge
        vocabulary gap between user language and catalog language.

        Maps common query terms to SHL-specific product terminology.
        This runs BEFORE embedding to improve semantic matching.

        Args:
            query: Cleaned user query.

        Returns:
            Query with appended synonym terms.
        """
        synonyms_added = []
        q_lower = query.lower()

        SYNONYM_MAP = {
            "behavioral": "personality questionnaire OPQ behavior competency",
            "behaviour": "personality questionnaire OPQ behavior competency",
            "personality": "OPQ32r personality questionnaire behavioral",
            "remote testing": "online unsupervised proctored",
            "adaptive": "verify interactive IRT computer adaptive",
            "irt": "verify interactive adaptive item response",
            "cognitive": "verify ability reasoning numerical verbal inductive",
            "aptitude": "verify ability numerical verbal reasoning",
            "data entry": "data entry typing clerical speed accuracy",
            "coding": "programming technical knowledge developer",
            "simulation": "simulation interactive scenario role-play",
            "leadership": "leadership enterprise management executive OPQ",
            "graduate": "graduate entry-level apprentice verify",
            "situational": "situational judgement SJT scenario",
        }

        for trigger, expansion in SYNONYM_MAP.items():
            if trigger in q_lower:
                synonyms_added.append(expansion)

        if synonyms_added:
            return query + " " + " ".join(synonyms_added)
        return query

    def extract_constraints(self, query: str) -> dict:
        """
        Extract all constraints from the query.

        Args:
            query: User query string.

        Returns:
            Dict with keys: time_limit, test_types, job_level, adaptive_required
        """
        return {
            "time_limit": self.extract_time_limit(query),
            "test_types": self.extract_test_types(query),
            "job_level": self.extract_job_level(query),
            "adaptive_required": self.extract_adaptive_requirement(query),
        }

    def clean_query_for_embedding(self, query: str) -> str:
        """
        Remove constraint phrases from query before embedding,
        so the semantic search focuses on the MEANING not the filters.

        Args:
            query: User query string.

        Returns:
            Cleaned query suitable for semantic embedding.
        """
        cleaned = query

        # Remove time constraint phrases
        time_patterns = [
            r"(?:under|less\s+than|max(?:imum)?|within|no\s+more\s+than|at\s+most|up\s+to)\s+\d+\s*(?:minutes?|mins?|min|hours?|hrs?)\b",
            r"\d+\s*(?:minutes?|mins?|min)\s+(?:or\s+less|maximum|max|limit)",
        ]
        for pattern in time_patterns:
            cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)

        # Normalize whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # If cleaning removed everything, use original
        if not cleaned or len(cleaned) < 5:
            return query

        return cleaned


    def expand_query(self, query: str) -> str:
        """
        Use Google Gemini to expand the query with inferred skills
        and competencies.

        Features:
            - Uses gemini-1.5-flash with temperature=0.3
            - Caches results to avoid redundant API calls
            - Falls back to original query on ANY failure
            - Input is sanitized before sending to LLM

        Args:
            query: Sanitized user query.

        Returns:
            Expanded query string, or original query on failure.
        """
        if not self._gemini_model:
            return query

        # Check cache first
        cache_key = query.strip().lower()
        if cache_key in self._query_cache:
            log.info("  Gemini: using cached expansion")
            return self._query_cache[cache_key]

        try:
            # Build the prompt with sanitized query
            prompt = GEMINI_EXPANSION_PROMPT.format(query=query)

            # Call Gemini with safety settings
            response = self._gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=200,
                ),
            )

            if response and response.text:
                expanded_skills = response.text.strip()
                # Combine original query with expanded skills
                expanded = f"{query} . Related skills: {expanded_skills}"
                log.info(f"  Gemini expanded: +{len(expanded_skills)} chars of skills")

                # Cache the result
                self._query_cache[cache_key] = expanded
                return expanded

        except Exception as e:
            log.warning(f"  Gemini expansion failed (falling back to raw query): {e}")

        return query


    def is_url(self, text: str) -> bool:
        """Check if input text appears to be a URL (delegates to utils)."""
        return utils_is_url(text)

    def validate_url(self, url: str) -> bool:
        """Validate URL for safety (delegates to utils)."""
        return utils_validate_url(url)

    def _detect_garbage_page(self, text: str) -> bool:
        """
        Detect if fetched page content is a login/bot-blocked page
        rather than actual job description content.

        Many sites (LinkedIn, Indeed, etc.) serve login walls to bots.
        This checks for telltale signs of garbage content.
        """
        text_lower = text.lower()
        # Markers that indicate a login/auth page, not a real JD
        garbage_markers = [
            "sign in", "log in", "create account", "join now",
            "enter your email", "forgot password", "privacy policy",
            "agree & join", "by clicking agree",
            "authwall", "captcha", "verify you are human",
            "cookie consent", "accept cookies", "cookie policy",
        ]
        marker_count = sum(1 for m in garbage_markers if m in text_lower)

        # If 3+ garbage markers found, it's likely a login page
        if marker_count >= 3:
            return True

        # If text is very short compared to a real JD
        word_count = len(text.split())
        if word_count < 30:
            return True

        return False

    def _is_job_board_url(self, url: str) -> bool:
        """Check if the URL is from a known job board that blocks bots."""
        job_board_domains = [
            "linkedin.com", "glassdoor.com", "glassdoor.co",
            "indeed.com", "monster.com", "naukri.com",
            "ziprecruiter.com", "lever.co", "greenhouse.io",
            "workday.com", "careers.google.com", "jobs.apple.com",
        ]
        url_lower = url.lower()
        return any(domain in url_lower for domain in job_board_domains)

    def _extract_meta_from_html(self, soup: BeautifulSoup, url: str) -> str:
        """
        Extract the most useful title/description from a page's metadata.
        This is the MOST RELIABLE method for job boards because they
        always put the job title in og:title for social sharing.

        Priority:
            1. og:title meta tag (most descriptive for job posts)
            2. <title> tag
            3. og:description meta tag
            4. meta description
        """
        parts = []

        # Try og:title first (LinkedIn/Glassdoor puts the job title here)
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            parts.append(og_title["content"])

        # Try page title
        if not parts and soup.title and soup.title.string:
            parts.append(soup.title.string.strip())

        # Try og:description (contains actual JD summary on most job boards)
        og_desc = soup.find("meta", property="og:description")
        if og_desc and og_desc.get("content"):
            parts.append(og_desc["content"])

        # Try meta description
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            if not og_desc:  # Don't double-add if og:description was already used
                parts.append(meta_desc["content"])

        if parts:
            combined = " . ".join(parts)
            # Clean common suffixes from job boards
            for suffix in [
                " | LinkedIn", " - LinkedIn", " | Indeed", " - Indeed",
                " | Glassdoor", " - Glassdoor", " | Monster", " - Monster",
                " | Naukri", " - Naukri.com",
            ]:
                combined = combined.replace(suffix, "")
            return combined.strip()

        return ""

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """
        Extract main content area text, ignoring navigation, sidebars,
        cookie banners, footer, ads, and other page chrome.

        Tries semantic HTML elements first, then falls back to
        largest text block heuristic.
        """
        # Strategy 1: Try semantic HTML5 content elements
        main_selectors = [
            {"name": "article"},
            {"name": "main"},
            {"role": "main"},
            {"class_": re.compile(r"job.?desc|job.?detail|posting.?detail|jd.?content", re.I)},
            {"id": re.compile(r"job.?desc|job.?detail|posting.?detail|jd.?content", re.I)},
            {"class_": re.compile(r"description|content.?body|main.?content", re.I)},
        ]

        for selector in main_selectors:
            elements = soup.find_all(**selector)
            if elements:
                # Get the largest matching element
                best = max(elements, key=lambda e: len(e.get_text()))
                text = best.get_text(separator=" ", strip=True)
                text = re.sub(r"\s+", " ", text).strip()
                if len(text) > 50:  # Must have substantial content
                    return text

        # Strategy 2: Fall back to all paragraph text (ignoring single-line items)
        paragraphs = soup.find_all("p")
        if paragraphs:
            long_paras = [p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 30]
            if long_paras:
                return " ".join(long_paras)

        return ""

    def fetch_jd_from_url(self, url: str) -> str:
        """
        Fetch job description text from a URL.

        Security:
            - Validates URL (blocks private IPs, metadata endpoints)
            - 10-second timeout
            - 5 MB max response size

        Smart Extraction:
            - For known job boards: always uses meta tags (most reliable)
            - Detects login/bot-blocked pages and falls back to meta tags
            - Extracts main content area, ignoring navigation/sidebar/footer
            - Caps extraction at 2000 chars to prevent noise pollution

        Args:
            url: URL to fetch job description from.

        Returns:
            Extracted text content, or empty string on failure.
        """
        if not self.validate_url(url):
            log.warning(f"  URL blocked by security validation: {url}")
            return ""

        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            }

            resp = requests.get(
                url,
                headers=headers,
                timeout=JD_FETCH_TIMEOUT,
                stream=True,
            )
            resp.raise_for_status()

            # Check content length
            content_length = resp.headers.get("Content-Length", "0")
            if int(content_length) > MAX_JD_SIZE:
                log.warning(f"  URL content too large: {content_length} bytes")
                return ""

            # Read with size limit
            content = resp.content[:MAX_JD_SIZE]
            soup = BeautifulSoup(content, "lxml")

            # ── Strategy 1: For known job boards, ALWAYS prefer meta tags ──
            # Job boards have the best og:title/og:description for social sharing
            # but serve garbage body content to bots
            if self._is_job_board_url(url):
                log.info(f"  Known job board detected - using meta tags")
                meta_text = self._extract_meta_from_html(soup, url)
                if meta_text:
                    log.info(f"  Extracted from meta: \"{meta_text[:100]}\"")
                    return meta_text[:MAX_QUERY_LEN]
                log.warning(f"  No meta tags found on job board page")

            # Remove script/style/nav/footer elements
            for tag in soup(["script", "style", "nav", "footer", "header",
                             "aside", "form", "iframe", "noscript"]):
                tag.decompose()

            # ── Strategy 2: Extract main content area ──
            main_text = self._extract_main_content(soup)

            if main_text and len(main_text) > 50:
                # Cap at 2000 chars to prevent noise from overwhelming semantic search
                text = main_text[:2000]
                log.info(f"  Extracted main content: {len(text)} chars")
            else:
                # Strategy 3: Full page text as last resort
                text = soup.get_text(separator=" ", strip=True)
                text = re.sub(r"\s+", " ", text).strip()

            # ── Detect garbage pages (login walls, bot protection) ──
            if self._detect_garbage_page(text):
                log.warning(f"  Detected login/bot-blocked page - extracting from meta tags")
                meta_text = self._extract_meta_from_html(soup, url)
                if meta_text:
                    log.info(f"  Extracted from meta: \"{meta_text[:100]}\"")
                    return meta_text[:MAX_QUERY_LEN]
                else:
                    log.warning(f"  No useful meta tags found in blocked page")
                    return ""

            # Cap at 2000 chars for quality (long text = too much noise)
            text = text[:2000]
            log.info(f"  Fetched JD from URL: {len(text)} chars")
            return text

        except requests.RequestException as e:
            log.warning(f"  Failed to fetch URL {url}: {e}")
            return ""
        except Exception as e:
            log.warning(f"  Error parsing URL content: {e}")
            return ""


    def compute_semantic_scores(self, query: str) -> np.ndarray:
        """
        Compute semantic similarity between query and all assessments.

        Args:
            query: Query text to embed.

        Returns:
            1D numpy array of cosine similarity scores (0 to 1).
        """
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        scores = cosine_similarity(query_embedding, self.embeddings)[0]
        # Clip to 0-1 range
        return np.clip(scores, 0.0, 1.0)

    def compute_keyword_scores(self, query: str) -> np.ndarray:
        """
        Compute keyword overlap ratio between query and each assessment.

        Uses simple word intersection ratio: |Q ∩ A| / |Q|

        Args:
            query: Query text.

        Returns:
            1D numpy array of keyword overlap scores (0 to 1).
        """
        query_words = set(query.lower().split())

        # Remove very common stop words to focus on meaningful terms
        stop_words = {
            "i", "we", "you", "the", "a", "an", "is", "are", "was", "were",
            "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", "can", "shall",
            "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
            "into", "through", "during", "before", "after", "and", "but", "or",
            "not", "no", "if", "then", "than", "that", "this", "it", "its",
            "my", "our", "your", "his", "her", "their", "what", "which", "who",
            "need", "want", "looking", "find", "search", "get", "test", "tests",
            "assessment", "assessments",
        }
        query_words -= stop_words

        if not query_words:
            return np.zeros(len(self.keyword_sets))

        scores = np.array([
            len(query_words & assessment_words) / len(query_words)
            for assessment_words in self.keyword_sets
        ])

        return scores

    def compute_name_match_scores(self, query: str) -> np.ndarray:
        """
        Compute assessment name matching bonus.

        Gives a strong boost to assessments whose names directly contain
        query keywords. Enhanced to handle:
        - Slash-separated names (e.g., "HTML/CSS" → tokens {html, css})
        - Exact technology matches (e.g., "javascript" in query → "JavaScript (New)")
        - Multi-token overlap with higher precision

        Args:
            query: Query text.

        Returns:
            1D numpy array of name-match bonus scores (0 to 1).
        """
        query_lower = query.lower()
        # Extract meaningful tokens, splitting on non-alphanumeric chars
        query_tokens = set(re.findall(r'[a-z0-9]+', query_lower)) - {
            "i", "we", "need", "want", "looking", "for", "a", "an", "the",
            "test", "tests", "assessment", "that", "can", "be", "under", "in",
            "with", "and", "or", "to", "of", "short", "quick", "new",
            "minutes", "min", "mins", "am", "hiring", "hire",
            "who", "is", "are", "some", "me", "my", "help",
        }

        if not query_tokens:
            return np.zeros(len(self.df))

        # Technology-specific keywords that deserve extra boost
        tech_keywords = {
            "html", "css", "javascript", "java", "python", "sql", "net",
            "excel", "sap", "oracle", "c", "ruby", "php", "angular",
            "react", "node", "typescript", "kotlin", "swift", "rust",
            "css3", "html5", "aws", "azure", "docker", "linux",
            "salesforce", "servicenow", "power", "sharepoint", "agile",
            "data", "entry", "typing", "accounting", "bookkeeping",
        }

        scores = np.zeros(len(self.df))
        for i, name in enumerate(self.df["assessment_name"]):
            name_lower = str(name).lower()
            # Split on slashes, hyphens, spaces, parens — catches "HTML/CSS", "C# (New)"
            name_tokens = set(re.findall(r'[a-z0-9]+', name_lower)) - {"new"}

            # Exact token overlap
            overlap = query_tokens & name_tokens

            # Substring matching for tech keywords:
            # "html" should match "html5", "css" should match "css3", etc.
            tech_query = query_tokens & tech_keywords
            if tech_query:
                for qt in tech_query:
                    if qt not in overlap:  # Not already matched exactly
                        for nt in name_tokens:
                            if nt.startswith(qt) or qt.startswith(nt):
                                overlap = overlap | {qt}
                                break

            if not overlap:
                continue

            # Base score: overlap ratio
            base_score = len(overlap) / len(query_tokens)

            # Tech boost: if the overlap contains technology keywords, boost significantly
            tech_overlap = overlap & tech_keywords
            if tech_overlap:
                # Strong boost: each tech keyword match is very valuable
                tech_boost = len(tech_overlap) * 0.3
                scores[i] = min(1.0, base_score + tech_boost)
            else:
                scores[i] = base_score

        return np.clip(scores, 0.0, 1.0)

    def compute_hybrid_scores(self, query: str, name_query: str = None) -> np.ndarray:
        """
        Compute hybrid score: weighted combination of semantic,
        keyword, and name-match components.

        Weights:
            - 0.50 semantic similarity (embeddings)
            - 0.25 keyword overlap (combined_text)
            - 0.25 name match bonus (assessment_name)

        The name match weight is higher (0.25) to ensure that when
        a user searches for a specific technology (e.g., "HTML", "Java"),
        assessments with that exact name are strongly surfaced.

        Args:
            query: Query text (synonym-expanded for semantic/keyword).
            name_query: Original clean query for name matching (optional).
                        If not provided, uses `query` for name matching too.

        Returns:
            1D numpy array of hybrid scores (0 to 1).
        """
        semantic = self.compute_semantic_scores(query)
        keyword = self.compute_keyword_scores(query)
        # Use original query for name matching to avoid synonym dilution
        name_match = self.compute_name_match_scores(name_query or query)

        hybrid = 0.50 * semantic + 0.25 * keyword + 0.25 * name_match
        return hybrid


    def apply_time_filter(self, indices: np.ndarray, time_limit: int) -> np.ndarray:
        """
        Filter assessments by time constraint.

        Rules:
            - duration == -1 (unknown): INCLUDE (don't penalize)
            - duration <= time_limit: INCLUDE
            - duration > time_limit: EXCLUDE

        Args:
            indices: Array of assessment indices to filter.
            time_limit: Maximum duration in minutes.

        Returns:
            Filtered array of indices.
        """
        durations = self.df.iloc[indices]["duration"].values
        mask = (durations <= time_limit) | (durations == -1)
        return indices[mask]

    def apply_type_filter(self, indices: np.ndarray, test_types: list[str]) -> np.ndarray:
        """
        Filter assessments by test type.

        Rule: Assessment matches if ANY of its types match ANY requested type.
        e.g., assessment "C,P,A,B" matches request for ["P"]

        Args:
            indices: Array of assessment indices to filter.
            test_types: List of required test type letters.

        Returns:
            Filtered array of indices.
        """
        requested = set(test_types)
        mask = np.array([
            bool(set(self.df.iloc[idx]["test_type_list"]) & requested)
            for idx in indices
        ])
        return indices[mask]




    def detect_multi_domain_query(self, query: str, test_types: Optional[list[str]]) -> bool:
        """
        Detect if query spans multiple test type domains.

        e.g., "Java developer with collaboration skills" → K + P
              "cognitive and personality assessment" → A + P

        Args:
            query: User query.
            test_types: Extracted test types.

        Returns:
            True if query is multi-domain.
        """
        if not test_types or len(test_types) < 2:
            return False

        # Check if types span different "families"
        technical = {"K", "A"}    # Knowledge, Ability
        behavioral = {"P", "B", "C"}  # Personality, Biodata, Competency
        practical = {"S", "E"}    # Simulation, Exercise

        families_hit = set()
        for t in test_types:
            if t in technical:
                families_hit.add("technical")
            elif t in behavioral:
                families_hit.add("behavioral")
            elif t in practical:
                families_hit.add("practical")

        return len(families_hit) >= 2

    def balance_results(
        self,
        results: list[dict],
        test_types: list[str],
        top_k: int,
    ) -> list[dict]:
        """
        Balance results to ensure multi-domain queries get diverse types.

        Strategy: Reserve 30% of slots for minority type family.

        Args:
            results: Sorted list of result dicts.
            test_types: Detected test types from query.
            top_k: Target number of results.

        Returns:
            Balanced list of results.
        """
        if len(results) <= MIN_RESULTS:
            return results

        technical = {"K", "A"}
        behavioral = {"P", "B", "C"}

        # Classify results into families
        tech_results = []
        behav_results = []
        other_results = []

        for r in results:
            r_types = set(r["test_type"])
            if r_types & technical:
                tech_results.append(r)
            elif r_types & behavioral:
                behav_results.append(r)
            else:
                other_results.append(r)

        # Determine minority family
        if not tech_results or not behav_results:
            return results[:top_k]

        minority_slots = max(2, int(top_k * 0.3))  # At least 2, up to 30%
        majority_slots = top_k - minority_slots

        if len(tech_results) < len(behav_results):
            minority = tech_results
            majority = behav_results
        else:
            minority = behav_results
            majority = tech_results

        balanced = majority[:majority_slots] + minority[:minority_slots] + other_results
        # Re-sort by score
        balanced.sort(key=lambda x: x["score"], reverse=True)

        return balanced[:top_k]


    def recommend(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Main recommendation method.

        Takes a natural-language query or URL and returns top-k
        matching SHL assessments.

        Args:
            query: User query string or URL to job description.
            top_k: Number of results to return (5-10, default 10).

        Returns:
            List of dicts, each containing:
                - assessment_name (str)
                - url (str)
                - score (float, 0-1)
                - duration (int, -1 if unknown)
                - test_type (list[str])
                - remote_support (str)
                - adaptive_support (str)
        """
        # ── Sanitize ──
        query = self.sanitize_query(query)
        if not query:
            log.warning("Empty query received")
            return []

        # Clamp top_k
        top_k = max(MIN_RESULTS, min(MAX_RESULTS, top_k))

        # ── URL handling ──
        if self.is_url(query):
            log.info(f"  Query is URL - fetching JD...")
            original_url = query
            jd_text = self.fetch_jd_from_url(query)
            if jd_text:
                query = jd_text
            else:
                # Fallback: extract keywords from URL slug
                # e.g. "glassdoor.com/job/software-engineer-python" -> "software engineer python"
                from urllib.parse import urlparse
                path = urlparse(original_url).path
                # Get the most descriptive path segment (usually the last meaningful one)
                segments = [s for s in path.split("/") if s and len(s) > 3]
                if segments:
                    # Use the longest segment (likely the job title slug)
                    best_slug = max(segments, key=len)
                    # Convert slug to words: "software-engineer-python" -> "software engineer python"
                    slug_words = re.sub(r'[^a-zA-Z0-9]', ' ', best_slug).strip()
                    # Remove common URL noise words
                    noise = {"job", "listing", "view", "htm", "html", "php", "aspx", "new", "apply"}
                    cleaned = " ".join(w for w in slug_words.split() if w.lower() not in noise and len(w) > 1)
                    if cleaned and len(cleaned) > 5:
                        query = cleaned
                        log.info(f"  Extracted from URL slug: \"{query}\"")
                    else:
                        log.warning("  Could not extract JD from URL, using URL as query")

        log.info(f"  Query: \"{query[:100]}{'...' if len(query) > 100 else ''}\"")

        # ── Extract constraints ──
        constraints = self.extract_constraints(query)
        log.info(f"  Constraints: {constraints}")

        # ── Clean query for embedding ──
        clean_query = self.clean_query_for_embedding(query)

        # ── Expand query with Gemini LLM ──
        expanded_query = self.expand_query(clean_query)

        # ── Inject domain synonyms for better semantic matching ──
        synonym_query = self.expand_query_synonyms(expanded_query)
        log.info(f"  Embedding query: \"{synonym_query[:100]}{'...' if len(synonym_query) > 100 else ''}\"")

        # ── Compute hybrid scores ──
        # Use synonym-expanded query for semantic/keyword (broad recall)
        # Use original clean query for name matching (precision)
        scores = self.compute_hybrid_scores(synonym_query, clean_query)

        # ── Adaptive/IRT boost ──
        if constraints["adaptive_required"]:
            adaptive_mask = self.df["adaptive_support"].str.lower() == "yes"
            scores[adaptive_mask.values] += 0.15
            log.info(f"  Adaptive boost applied: {adaptive_mask.sum()} assessments boosted")

        # ── Job Level boost ──
        if constraints["job_level"]:
            level_kw = JOB_LEVEL_KEYWORDS.get(constraints["job_level"], [])
            if level_kw:
                # Add 0.15 boost to items that explicitly match the requested job level
                # This helps them without blindly overriding perfect tech/name matches
                job_level_boost_count = 0
                for i, levels_str in enumerate(self.df["job_levels"]):
                    levels_lower = str(levels_str).lower()
                    if any(kw in levels_lower for kw in level_kw):
                        scores[i] = min(1.0, scores[i] + 0.15)
                        job_level_boost_count += 1
                log.info(f"  Job level boost applied: {job_level_boost_count} assessments boosted")

        # ── Get ranked indices ──
        ranked_indices = np.argsort(scores)[::-1]  # Descending

        # ── Apply hard filters ──
        filtered = ranked_indices.copy()

        # Time filter
        if constraints["time_limit"] is not None:
            pre_count = len(filtered)
            filtered = self.apply_time_filter(filtered, constraints["time_limit"])
            log.info(f"  Time filter ({constraints['time_limit']}min): {pre_count} → {len(filtered)}")

        # Type filter
        if constraints["test_types"] is not None:
            pre_count = len(filtered)
            filtered_by_type = self.apply_type_filter(filtered, constraints["test_types"])
            # Only apply if it doesn't reduce below MIN_RESULTS
            if len(filtered_by_type) >= MIN_RESULTS:
                filtered = filtered_by_type
                log.info(f"  Type filter ({constraints['test_types']}): {pre_count} → {len(filtered)}")
            else:
                log.info(f"  Type filter would reduce to {len(filtered_by_type)} — relaxing")

        # ── Ensure minimum results ──
        if len(filtered) < MIN_RESULTS:
            log.info(f"  Only {len(filtered)} after filtering — relaxing to top {MIN_RESULTS} unfiltered")
            filtered = ranked_indices[:max(MIN_RESULTS, len(filtered))]

        # ── Build result dicts ──
        results = []
        for idx in filtered:
            row = self.df.iloc[idx]
            results.append({
                "assessment_name": str(row["assessment_name"]),
                "url": str(row["url"]),
                "score": round(float(scores[idx]), 4),
                "duration": int(row["duration"]),
                "test_type": row["test_type_list"],
                "remote_support": str(row.get("remote_support", "Yes")),
                "adaptive_support": str(row.get("adaptive_support", "No")),
            })

        # Multi-domain detection and balancing.
        # SHL explicitly requires that a query like "Java developer who
        # can collaborate" returns BOTH technical (K) AND behavioral (P)
        # results. Without this, pure semantic search just returns all K.
        is_multi = False
        if constraints["test_types"] and self.detect_multi_domain_query(query, constraints["test_types"]):
            is_multi = True
        else:
            # Fallback: check query text directly for behavioral signals
            # even if the type extractor didn't categorize them formally.
            behavioral_signals = [
                "collaborat", "teamwork", "team skills", "interpersonal",
                "soft skills", "people skills", "leadership", "personality",
                "behavioral", "communication", "work with",
                "cultural fit", "team player",
            ]
            query_lower = query.lower()
            has_tech = any(kw in query_lower for kw in [
                "java", "python", "sql", "javascript", "developer", "coding",
                "technical", ".net", "html", "css", "software", "programming",
            ])
            has_behav = any(sig in query_lower for sig in behavioral_signals)
            if has_tech and has_behav:
                is_multi = True

        if is_multi:
            log.info("  Multi-domain query detected — applying balance")
            tech_types = {"K", "A"}
            behav_types = {"P", "B", "C"}

            # Take the candidate top_k and check if both families appear
            top_slice = results[:top_k]
            top_types = set()
            for r in top_slice:
                top_types.update(r["test_type"])

            has_tech_top = bool(top_types & tech_types)
            has_behav_top = bool(top_types & behav_types)

            if has_tech_top and not has_behav_top:
                # Top results are all tech. Reserve slots for behavioral.
                slots_for_behav = max(2, int(top_k * 0.3))
                slots_for_tech = top_k - slots_for_behav
                tech_slice = top_slice[:slots_for_tech]
                used_urls = {r["url"] for r in tech_slice}

                # Gather behavioral candidates from ALL results + full catalog
                behav_pool = []
                for r in results:
                    if set(r["test_type"]) & behav_types and r["url"] not in used_urls:
                        behav_pool.append(r)
                        used_urls.add(r["url"])

                # If not enough in filtered results, scan full catalog
                if len(behav_pool) < slots_for_behav:
                    for i in range(len(self.df)):
                        if len(behav_pool) >= slots_for_behav * 2:
                            break
                        row = self.df.iloc[i]
                        row_types = set(row["test_type_list"])
                        url = str(row["url"])
                        if row_types & behav_types and url not in used_urls:
                            behav_pool.append({
                                "assessment_name": str(row["assessment_name"]),
                                "url": url,
                                "score": round(float(scores[i]), 4),
                                "duration": int(row["duration"]),
                                "test_type": row["test_type_list"],
                                "remote_support": str(row.get("remote_support", "Yes")),
                                "adaptive_support": str(row.get("adaptive_support", "No")),
                            })
                            used_urls.add(url)

                behav_pool.sort(key=lambda x: x["score"], reverse=True)
                behav_top = behav_pool[:slots_for_behav]

                results = tech_slice + behav_top
                results.sort(key=lambda x: x["score"], reverse=True)
                log.info(f"  Balanced: {len(tech_slice)} tech + {len(behav_top)} behavioral")

            elif has_behav_top and not has_tech_top:
                # Opposite case: all behavioral, inject tech
                slots_for_tech = max(2, int(top_k * 0.3))
                slots_for_behav = top_k - slots_for_tech
                behav_slice = top_slice[:slots_for_behav]
                used_urls = {r["url"] for r in behav_slice}

                tech_pool = [r for r in results if set(r["test_type"]) & tech_types and r["url"] not in used_urls]
                tech_pool.sort(key=lambda x: x["score"], reverse=True)
                tech_top = tech_pool[:slots_for_tech]

                results = behav_slice + tech_top
                results.sort(key=lambda x: x["score"], reverse=True)
                log.info(f"  Balanced: {len(tech_top)} tech + {len(behav_slice)} behavioral")

        # Final slice
        results = results[:top_k]
        log.info(f"  Returning {len(results)} results\n")

        return results



def main():
    """Quick standalone test."""
    engine = AssessmentEngine()

    test_queries = [
        "I need a cognitive ability test for entry-level candidates under 30 minutes",
        "Looking for a personality assessment for senior managers",
        "Java developer coding test that supports remote testing",
        "We need a simulation for customer service roles",
        "Short technical assessment for .NET developers max 20 minutes",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"QUERY: {q}")
        print(f"{'='*70}")
        results = engine.recommend(q)
        for i, r in enumerate(results, 1):
            types = ",".join(r["test_type"])
            dur = r["duration"] if r["duration"] != -1 else "?"
            print(
                f"  {i:2d}. [{r['score']:.3f}] {r['assessment_name'][:50]:<50} "
                f"type={types:<10} dur={dur}"
            )


if __name__ == "__main__":
    main()
