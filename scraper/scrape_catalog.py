"""
Scrapes the SHL product catalog for Individual Test Solutions (type=1).

SHL's catalog is paginated at 12 items per page with server-rendered HTML,
so I used requests + BeautifulSoup rather than a headless browser.
The scraper paginates through list pages, collects assessment URLs, then
visits each detail page to pull out structured data (description, test type,
duration, remote/adaptive support, job levels, languages).

I added polite delays between requests and retry logic because SHL
occasionally returns 5xx during heavy scraping.

    $ python scraper/scrape_catalog.py

Author: Mohammad Inayat Hussain
"""

import os
import re
import sys
import time
import random
import logging
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path

BASE_URL = "https://www.shl.com/products/product-catalog/"
DETAIL_BASE = "https://www.shl.com"
CATALOG_TYPE = 1          # 1 = Individual Test Solutions, 2 = Pre-packaged Job Solutions
ITEMS_PER_PAGE = 12
MAX_PAGES = 35            # Safety cap (actual ~32 pages)
MIN_DELAY = 1.0           # seconds between requests
MAX_DELAY = 2.5
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_FILE = OUTPUT_DIR / "shl_catalog.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("shl_scraper")



def polite_delay():
    """Random delay between requests to be respectful."""
    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))


def fetch_page(url: str, session: requests.Session, retries: int = 3) -> BeautifulSoup | None:
    """Fetch a URL and return parsed BeautifulSoup, with retry logic."""
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "lxml")
        except requests.RequestException as e:
            log.warning(f"  Attempt {attempt}/{retries} failed for {url}: {e}")
            if attempt < retries:
                time.sleep(3 * attempt)
    log.error(f"  FAILED after {retries} attempts: {url}")
    return None



def scrape_catalog_listing(session: requests.Session) -> list[dict]:
    """
    Iterate over all paginated catalog list pages for type=1.
    Returns list of dicts with 'name' and 'url' for each assessment.
    """
    assessments = []
    seen_urls = set()
    page_num = 0

    while page_num < MAX_PAGES:
        start = page_num * ITEMS_PER_PAGE
        url = f"{BASE_URL}?start={start}&type={CATALOG_TYPE}"
        log.info(f"Scraping catalog page {page_num + 1} (start={start})...")

        soup = fetch_page(url, session)
        if soup is None:
            log.warning(f"  Skipping page {page_num + 1} due to fetch failure.")
            page_num += 1
            polite_delay()
            continue

        # Find the Individual Test Solutions section
        # The assessments are links to /products/product-catalog/view/...
        links = soup.find_all("a", href=re.compile(r"/products/product-catalog/view/"))
        
        page_found = 0
        for link in links:
            href = link.get("href", "")
            name = link.get_text(strip=True)
            
            if not name or not href:
                continue
                
            # Build full URL
            full_url = href if href.startswith("http") else DETAIL_BASE + href
            
            # Deduplicate
            if full_url in seen_urls:
                continue
            seen_urls.add(full_url)
            
            assessments.append({"name": name, "url": full_url})
            page_found += 1

        log.info(f"  Found {page_found} new assessments (total: {len(assessments)})")

        if page_found == 0:
            log.info("  No new assessments found — reached end of catalog.")
            break

        page_num += 1
        polite_delay()

    return assessments



def parse_detail_page(soup: BeautifulSoup, name: str, url: str) -> dict:
    """
    Extract structured data from an individual assessment detail page.
    
    Structure observed on SHL detail pages:
      - Description paragraph
      - Job levels
      - Languages
      - Assessment length / Completion time
      - Test Type (K, P, C, A, B, E, S, D)
      - Remote Testing: Yes/No
      - Adaptive/IRT: Yes/No (sometimes shown as checkbox/icon)
    """
    record = {
        "assessment_name": name,
        "url": url,
        "description": "",
        "test_type": "",
        "duration_minutes": "",
        "remote_testing": "No",
        "adaptive_irt": "No",
        "job_levels": "",
        "languages": "",
    }

    try:
        # Get the main content area
        body_text = soup.get_text(separator="\n")
        
        # ── Description ──
        desc_section = soup.find("h4", string=re.compile(r"Description", re.I))
        if desc_section:
            # Get the next sibling paragraph or text
            next_el = desc_section.find_next_sibling()
            if next_el:
                record["description"] = next_el.get_text(strip=True)
            else:
                # Try getting text between this h4 and the next h4
                desc_parts = []
                for sibling in desc_section.next_siblings:
                    if sibling.name and sibling.name.startswith("h"):
                        break
                    text = sibling.get_text(strip=True) if hasattr(sibling, 'get_text') else str(sibling).strip()
                    if text:
                        desc_parts.append(text)
                record["description"] = " ".join(desc_parts)
        
        if not record["description"]:
            # Fallback: look for og:description meta tag
            og_desc = soup.find("meta", property="og:description")
            if og_desc:
                content = og_desc.get("content", "")
                # Remove the assessment name prefix if present
                if ":" in content:
                    record["description"] = content.split(":", 1)[1].strip()
                else:
                    record["description"] = content.strip()

        # ── Test Type ──
        test_type_match = re.search(r"Test\s*Type\s*:?\s*([A-Z](?:\s*,\s*[A-Z])*)", body_text)
        if test_type_match:
            record["test_type"] = test_type_match.group(1).strip()
        else:
            # Look for test type categories in the page
            # The catalog uses single letters: A, B, C, D, E, K, P, S
            type_letters = []
            for letter_map in [
                ("A", ["Ability", "Aptitude"]),
                ("B", ["Biodata", "Situational Judgement"]),
                ("C", ["Competencies"]),
                ("D", ["Development", "360"]),
                ("E", ["Exercises"]),
                ("K", ["Knowledge", "Skills"]),
                ("P", ["Personality", "Behavior"]),
                ("S", ["Simulations"]),
            ]:
                letter, keywords = letter_map
                # Check if the assessment page mentions these test categories
                for keyword in keywords:
                    pattern = rf"Test\s*Type.*?{keyword}"
                    if re.search(pattern, body_text, re.I | re.DOTALL):
                        type_letters.append(letter)
                        break
            if type_letters:
                record["test_type"] = ",".join(type_letters)

        # ── Duration ──
        # SHL format: "Approximate Completion Time in minutes = 9"
        duration_match = re.search(
            r"(?:Completion\s*Time\s*in\s*minutes|Duration|Assessment\s*[Ll]ength)\s*[=:]\s*(\d+)",
            body_text, re.I
        )
        if not duration_match:
            # Fallback: "Completion Time ... N minutes"
            duration_match = re.search(
                r"(?:Completion\s*Time|Duration|Assessment\s*[Ll]ength).*?(\d+)\s*(?:minutes|mins?)",
                body_text, re.I | re.DOTALL
            )
        if not duration_match:
            # Fallback: look for "minutes = N" anywhere in assessment length section
            duration_match = re.search(r"minutes\s*=\s*(\d+)", body_text, re.I)
        if duration_match:
            record["duration_minutes"] = duration_match.group(1)

        # ── Remote Testing ──
        remote_section = re.search(r"Remote\s*Testing\s*:?\s*(Yes|No)?", body_text, re.I)
        if remote_section:
            if remote_section.group(1):
                record["remote_testing"] = remote_section.group(1).strip().title()
            else:
                # If "Remote Testing" header exists but no explicit Yes/No,
                # check for checkmark icons or "Yes" nearby
                record["remote_testing"] = "Yes"
        
        # Also check for remote testing icons/checkmarks
        remote_icons = soup.find_all(string=re.compile(r"Remote\s*Testing", re.I))
        for icon_el in remote_icons:
            parent = icon_el.parent if hasattr(icon_el, 'parent') else None
            if parent:
                # Look for checkmark or "Yes" indicators
                parent_text = parent.get_text() if hasattr(parent, 'get_text') else str(parent)
                if any(word in parent_text.lower() for word in ["yes", "✓", "✔", "true"]):
                    record["remote_testing"] = "Yes"

        # ── Adaptive/IRT ──
        adaptive_match = re.search(r"(?:Adaptive|IRT)\s*(?:Testing)?\s*:?\s*(Yes|No)?", body_text, re.I)
        if adaptive_match:
            if adaptive_match.group(1):
                record["adaptive_irt"] = adaptive_match.group(1).strip().title()
            else:
                record["adaptive_irt"] = "Yes"

        # ── Job Levels ──
        job_section = soup.find("h4", string=re.compile(r"Job\s*[Ll]evels?", re.I))
        if job_section:
            job_parts = []
            for sibling in job_section.next_siblings:
                if sibling.name and sibling.name.startswith("h"):
                    break
                text = sibling.get_text(strip=True) if hasattr(sibling, 'get_text') else str(sibling).strip()
                if text:
                    job_parts.append(text)
            record["job_levels"] = " ".join(job_parts).strip().rstrip(",")

        # ── Languages ──
        lang_section = soup.find("h4", string=re.compile(r"Languages?", re.I))
        if lang_section:
            lang_parts = []
            for sibling in lang_section.next_siblings:
                if sibling.name and sibling.name.startswith("h"):
                    break
                text = sibling.get_text(strip=True) if hasattr(sibling, 'get_text') else str(sibling).strip()
                if text:
                    lang_parts.append(text)
            record["languages"] = " ".join(lang_parts).strip().rstrip(",")

    except Exception as e:
        log.warning(f"  Error parsing detail page for '{name}': {e}")

    return record


def scrape_detail_pages(session: requests.Session, assessments: list[dict]) -> list[dict]:
    """
    Visit each assessment's detail page and extract structured data.
    """
    results = []
    total = len(assessments)

    for i, assessment in enumerate(assessments, 1):
        name = assessment["name"]
        url = assessment["url"]
        log.info(f"  [{i}/{total}] Scraping detail: {name}")

        soup = fetch_page(url, session)
        if soup is None:
            log.warning(f"  Skipping '{name}' — could not fetch detail page.")
            # Still record it with what we have
            results.append({
                "assessment_name": name,
                "url": url,
                "description": "",
                "test_type": "",
                "duration_minutes": "",
                "remote_testing": "",
                "adaptive_irt": "",
                "job_levels": "",
                "languages": "",
            })
            continue

        record = parse_detail_page(soup, name, url)
        results.append(record)

        if i % 25 == 0:
            log.info(f"  Progress: {i}/{total} detail pages scraped ({i*100//total}%)")

        polite_delay()

    return results



def main():
    log.info("=" * 65)
    log.info("SHL Product Catalog Scraper — Individual Test Solutions")
    log.info("=" * 65)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()

    # ── Phase 1: Catalog listing ──
    log.info("\n>>> PHASE 1: Scraping catalog listing pages...")
    assessments = scrape_catalog_listing(session)
    log.info(f"\n>>> Found {len(assessments)} individual test solution assessments.")

    if len(assessments) < 300:
        log.warning(
            f"Expected at least 377 assessments but found {len(assessments)}. "
            "Continuing anyway, but results may be incomplete."
        )

    # ── Phase 2: Detail pages ──
    log.info("\n>>> PHASE 2: Scraping detail pages for each assessment...")
    results = scrape_detail_pages(session, assessments)

    # ── Save results ──
    df = pd.DataFrame(results)
    
    # Clean up columns
    df["description"] = df["description"].str.strip()
    df["test_type"] = df["test_type"].str.strip()
    df["duration_minutes"] = df["duration_minutes"].str.strip()
    
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    log.info(f"\n>>> SAVED: {len(df)} assessments → {OUTPUT_FILE}")

    # ── Summary stats ──
    log.info("\n" + "=" * 65)
    log.info("SCRAPING COMPLETE — Summary:")
    log.info(f"  Total assessments:     {len(df)}")
    log.info(f"  With description:      {(df['description'] != '').sum()}")
    log.info(f"  With test type:        {(df['test_type'] != '').sum()}")
    log.info(f"  With duration:         {(df['duration_minutes'] != '').sum()}")
    log.info(f"  Remote testing = Yes:  {(df['remote_testing'] == 'Yes').sum()}")
    log.info(f"  Adaptive/IRT = Yes:    {(df['adaptive_irt'] == 'Yes').sum()}")
    log.info("=" * 65)

    return df


if __name__ == "__main__":
    main()
