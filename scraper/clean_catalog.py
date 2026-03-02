"""
Cleans the raw scraped catalog data and produces shl_catalog_clean.csv.

The raw scraper output had a few issues I needed to fix:
  - Column names didn't match the API spec (duration_minutes vs duration)
  - Test types were space-separated on SHL's site ("C P A B") but I need
    them comma-separated for filtering. Had to re-scrape detail pages
    because the initial scraper only caught the first letter.
  - Some durations were missing (set to -1 as a sentinel)
  - Needed a combined_text column for embedding generation
  - Deduplication, URL validation, pre-packaged contamination check

    $ python scraper/clean_catalog.py

Author: Mohammad Inayat Hussain
"""

import re
import sys
import time
import random
import logging
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path

INPUT_FILE = Path(__file__).resolve().parent.parent / "data" / "shl_catalog.csv"
OUTPUT_FILE = Path(__file__).resolve().parent.parent / "data" / "shl_catalog_clean.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Test type letter → full description mapping
TEST_TYPE_MAP = {
    "A": "Ability and Aptitude test",
    "B": "Biodata and Situational Judgement",
    "C": "Competency assessment",
    "D": "Development and 360 feedback",
    "E": "Assessment Exercise",
    "K": "Knowledge and Skills test",
    "P": "Personality and Behavioral assessment",
    "S": "Simulation exercise",
}

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("shl_cleaner")



def fetch_page(url: str, session: requests.Session, retries: int = 3):
    """Fetch a URL and return parsed BeautifulSoup."""
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "lxml")
        except requests.RequestException as e:
            if attempt < retries:
                time.sleep(3 * attempt)
    return None


def extract_test_type_from_page(soup: BeautifulSoup) -> str:
    """
    Extract test type from detail page.
    SHL format: "Test Type: C P A B" (space-separated single letters)
    """
    body_text = soup.get_text(separator="\n")
    
    # Match "Test Type: C P A B" or "Test Type: K"
    # The letters are space-separated single uppercase letters
    match = re.search(r"Test\s*Type\s*:\s*([A-Z](?:\s+[A-Z])*)", body_text)
    if match:
        raw = match.group(1).strip()
        # Convert space-separated to comma-separated
        letters = raw.split()
        # Validate each is a known test type letter
        valid_letters = [l for l in letters if l in TEST_TYPE_MAP]
        if valid_letters:
            return ",".join(valid_letters)
    
    return ""


def rescrape_test_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Re-scrape test_type for all assessments to fix the space-separated
    letter parsing bug. Only re-scrapes rows where test_type appears
    to be a single letter (likely truncated).
    """
    session = requests.Session()
    total = len(df)
    fixed_count = 0
    
    log.info(f"\n>>> RE-SCRAPING test_type for {total} assessments...")
    
    for idx, row in df.iterrows():
        url = row["url"]
        old_type = str(row.get("test_type", "")).strip()
        
        if idx % 50 == 0:
            log.info(f"  Progress: {idx}/{total} ({idx*100//total}%)")
        
        soup = fetch_page(url, session)
        if soup is None:
            continue
        
        new_type = extract_test_type_from_page(soup)
        if new_type and new_type != old_type:
            df.at[idx, "test_type"] = new_type
            fixed_count += 1
        
        # Be polite
        time.sleep(random.uniform(0.8, 1.5))
    
    log.info(f"  Fixed test_type for {fixed_count}/{total} assessments")
    return df



def main():
    log.info("=" * 65)
    log.info("SHL Catalog Data Cleaner")
    log.info("=" * 65)

    # ── Load raw data ──
    df = pd.read_csv(INPUT_FILE)
    rows_before = len(df)
    log.info(f"\nLoaded {rows_before} rows from {INPUT_FILE}")
    log.info(f"Original columns: {list(df.columns)}")

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 0: Re-scrape test_type to fix truncation bug
    # ═══════════════════════════════════════════════════════════════════════
    df = rescrape_test_types(df)

    # ═══════════════════════════════════════════════════════════════════════
    # FIX 1: Column Renaming (API spec alignment)
    # ═══════════════════════════════════════════════════════════════════════
    log.info("\n>>> FIX 1: Renaming columns to match API spec...")
    df = df.rename(columns={
        "duration_minutes": "duration",
        "remote_testing": "remote_support",
        "adaptive_irt": "adaptive_support",
    })
    log.info(f"  Columns after rename: {list(df.columns)}")

    # ═══════════════════════════════════════════════════════════════════════
    # FIX 2: Duration handling
    # ═══════════════════════════════════════════════════════════════════════
    log.info("\n>>> FIX 2: Fixing duration values...")
    
    # Convert to numeric, coerce errors to NaN
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
    
    nan_count = df["duration"].isna().sum()
    known_durations = df["duration"].dropna()
    
    log.info(f"  Known durations: {len(known_durations)}")
    log.info(f"  Missing durations: {nan_count}")
    if len(known_durations) > 0:
        log.info(f"  Min (known): {known_durations.min():.0f}")
        log.info(f"  Max (known): {known_durations.max():.0f}")
        log.info(f"  Mean (known): {known_durations.mean():.1f}")
    
    # Replace NaN with -1 (flag for "unknown duration")
    df["duration"] = df["duration"].fillna(-1).astype(int)
    log.info(f"  Set {nan_count} missing durations to -1 (unknown)")

    # ═══════════════════════════════════════════════════════════════════════
    # FIX 3: test_type cleanup
    # ═══════════════════════════════════════════════════════════════════════
    log.info("\n>>> FIX 3: Cleaning test_type values...")
    
    # Strip whitespace
    df["test_type"] = df["test_type"].astype(str).str.strip()
    
    # Replace 'nan' strings
    df.loc[df["test_type"] == "nan", "test_type"] = ""
    
    # Verify all values are valid letters
    valid_letters = set("ABCDEKPS")
    invalid_types = []
    for idx, val in df["test_type"].items():
        if val:
            letters = [l.strip() for l in val.split(",")]
            invalid = [l for l in letters if l and l not in valid_letters]
            if invalid:
                invalid_types.append((idx, val, invalid))
    
    if invalid_types:
        log.warning(f"  Found {len(invalid_types)} rows with invalid test_type values:")
        for idx, val, inv in invalid_types[:5]:
            log.warning(f"    Row {idx}: '{val}' — invalid: {inv}")
    else:
        log.info(f"  All test_type values are valid")
    
    unique_types = sorted(set(
        letter 
        for val in df["test_type"] 
        for letter in val.split(",") 
        if letter.strip()
    ))
    log.info(f"  Unique test type letters: {unique_types}")

    # ═══════════════════════════════════════════════════════════════════════
    # FIX 4: Deduplication
    # ═══════════════════════════════════════════════════════════════════════
    log.info("\n>>> FIX 4: Checking for duplicates...")
    
    dupes = df[df.duplicated(subset=["url"], keep="first")]
    if len(dupes) > 0:
        log.info(f"  Removing {len(dupes)} duplicate URLs:")
        for _, row in dupes.iterrows():
            log.info(f"    - {row['assessment_name']} ({row['url']})")
        df = df.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
    else:
        log.info(f"  No duplicate URLs found")

    # ═══════════════════════════════════════════════════════════════════════
    # FIX 5: URL validation
    # ═══════════════════════════════════════════════════════════════════════
    log.info("\n>>> FIX 5: Validating URLs...")
    
    invalid_urls = df[~df["url"].str.startswith("https://www.shl.com")]
    if len(invalid_urls) > 0:
        log.warning(f"  Found {len(invalid_urls)} invalid URLs:")
        for _, row in invalid_urls.iterrows():
            log.warning(f"    - {row['assessment_name']}: {row['url']}")
        df = df[df["url"].str.startswith("https://www.shl.com")].reset_index(drop=True)
    else:
        log.info(f"  All {len(df)} URLs are valid (start with https://www.shl.com)")

    # ═══════════════════════════════════════════════════════════════════════
    # FIX 6: Pre-packaged contamination check
    # ═══════════════════════════════════════════════════════════════════════
    log.info("\n>>> FIX 6: Checking for pre-packaged contamination...")
    
    suspect_keywords = ["pre-packaged", "bundle", "package", "pre packaged"]
    suspects = df[df["assessment_name"].str.lower().str.contains(
        "|".join(suspect_keywords), na=False
    )]
    
    if len(suspects) > 0:
        log.warning(f"  ⚠️ Found {len(suspects)} potentially pre-packaged items:")
        for _, row in suspects.iterrows():
            log.warning(f"    - {row['assessment_name']}")
        log.info("  (Not auto-removing — these may be false positives. Review manually.)")
    else:
        log.info(f"  No pre-packaged keywords found. Data is clean.")

    # ═══════════════════════════════════════════════════════════════════════
    # FIX 7: Missing value handling
    # ═══════════════════════════════════════════════════════════════════════
    log.info("\n>>> FIX 7: Handling missing values...")
    
    # If description is empty, fill with assessment_name
    empty_desc = (df["description"].isna()) | (df["description"].str.strip() == "")
    if empty_desc.sum() > 0:
        log.info(f"  Filling {empty_desc.sum()} empty descriptions with assessment_name")
        df.loc[empty_desc, "description"] = df.loc[empty_desc, "assessment_name"]
    else:
        log.info(f"  All descriptions present")
    
    # If remote_support is empty, set to "Unknown"
    empty_remote = (df["remote_support"].isna()) | (df["remote_support"].str.strip() == "")
    if empty_remote.sum() > 0:
        log.info(f"  Setting {empty_remote.sum()} empty remote_support to 'Unknown'")
        df.loc[empty_remote, "remote_support"] = "Unknown"
    
    # If adaptive_support is empty, set to "Unknown"
    empty_adaptive = (df["adaptive_support"].isna()) | (df["adaptive_support"].str.strip() == "")
    if empty_adaptive.sum() > 0:
        log.info(f"  Setting {empty_adaptive.sum()} empty adaptive_support to 'Unknown'")
        df.loc[empty_adaptive, "adaptive_support"] = "Unknown"

    # ═══════════════════════════════════════════════════════════════════════
    # FIX 8: Create combined_text column for embeddings
    # ═══════════════════════════════════════════════════════════════════════
    log.info("\n>>> FIX 8: Creating combined_text column for embeddings...")
    
    def build_combined_text(row):
        """Build rich text representation for semantic embedding."""
        parts = [row["assessment_name"]]
        
        # Add description
        if pd.notna(row["description"]) and str(row["description"]).strip():
            parts.append(str(row["description"]).strip())
        
        # Add test type descriptions
        test_types = str(row.get("test_type", "")).strip()
        if test_types:
            type_descriptions = []
            for letter in test_types.split(","):
                letter = letter.strip()
                if letter in TEST_TYPE_MAP:
                    type_descriptions.append(TEST_TYPE_MAP[letter])
            if type_descriptions:
                parts.append("Test categories: " + ", ".join(type_descriptions))
        
        # Add duration info
        duration = row.get("duration", -1)
        if duration > 0:
            parts.append(f"Duration: {duration} minutes")
        
        # Add job levels
        if pd.notna(row.get("job_levels")) and str(row["job_levels"]).strip():
            parts.append(f"Job levels: {row['job_levels']}")
        
        # Add remote support
        remote = str(row.get("remote_support", "")).strip()
        if remote and remote != "Unknown":
            parts.append(f"Remote testing: {remote}")
        
        # Add adaptive support
        adaptive = str(row.get("adaptive_support", "")).strip()
        if adaptive == "Yes":
            parts.append("Adaptive/IRT testing supported")
        
        return " . ".join(parts)
    
    df["combined_text"] = df.apply(build_combined_text, axis=1)
    
    avg_len = df["combined_text"].str.len().mean()
    log.info(f"  Created combined_text — average length: {avg_len:.0f} chars")

    # ═══════════════════════════════════════════════════════════════════════
    # SAVE
    # ═══════════════════════════════════════════════════════════════════════
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    rows_after = len(df)
    
    # ═══════════════════════════════════════════════════════════════════════
    # VERIFICATION OUTPUT
    # ═══════════════════════════════════════════════════════════════════════
    log.info("\n" + "=" * 65)
    log.info("VERIFICATION REPORT")
    log.info("=" * 65)
    
    log.info(f"\n  Row count before cleaning: {rows_before}")
    log.info(f"  Row count after cleaning:  {rows_after}")
    log.info(f"  Rows removed:              {rows_before - rows_after}")
    
    log.info(f"\n  Columns: {list(df.columns)}")
    
    # Verify column names
    required_cols = {"duration", "remote_support", "adaptive_support"}
    forbidden_cols = {"duration_minutes", "remote_testing", "adaptive_irt"}
    
    present_required = required_cols.intersection(set(df.columns))
    present_forbidden = forbidden_cols.intersection(set(df.columns))
    
    log.info(f"\n  Required columns present:  {present_required} ({'✅ ALL' if present_required == required_cols else '❌ MISSING'})")
    log.info(f"  Forbidden columns absent:  {len(present_forbidden) == 0} ({'✅ CLEAN' if len(present_forbidden) == 0 else '❌ ' + str(present_forbidden)})")
    
    log.info(f"\n  Unique test_type values: {unique_types}")
    
    # Test type distribution (considering multi-type assessments)
    all_types = []
    for val in df["test_type"]:
        if val:
            all_types.extend([l.strip() for l in val.split(",") if l.strip()])
    type_counts = pd.Series(all_types).value_counts()
    log.info(f"\n  Test type distribution (including multi-type):")
    for t, c in type_counts.items():
        desc = TEST_TYPE_MAP.get(t, "Unknown")
        log.info(f"    {t} ({desc}): {c}")
    
    # Multi-type stats
    multi_type = df[df["test_type"].str.contains(",", na=False)]
    log.info(f"\n  Assessments with multiple test types: {len(multi_type)}")
    
    # Duration stats
    known = df[df["duration"] != -1]["duration"]
    unknown = (df["duration"] == -1).sum()
    log.info(f"\n  Duration stats:")
    log.info(f"    Unknown (-1): {unknown}")
    log.info(f"    Known count:  {len(known)}")
    if len(known) > 0:
        log.info(f"    Min (known):  {known.min()}")
        log.info(f"    Max (known):  {known.max()}")
        log.info(f"    Mean (known): {known.mean():.1f}")
        log.info(f"    Median:       {known.median():.1f}")
    
    # Remote support distribution
    log.info(f"\n  Remote support: {df['remote_support'].value_counts().to_dict()}")
    log.info(f"  Adaptive support: {df['adaptive_support'].value_counts().to_dict()}")
    
    # Sample rows
    log.info(f"\n  Sample of 5 rows:")
    sample = df[["assessment_name", "test_type", "duration", "remote_support", "adaptive_support"]].head(5)
    for _, row in sample.iterrows():
        log.info(f"    {row['assessment_name'][:45]:<45} type={row['test_type']:<10} dur={row['duration']:<4} remote={row['remote_support']:<4} adaptive={row['adaptive_support']}")
    
    log.info(f"\n>>> SAVED: {OUTPUT_FILE}")
    log.info("=" * 65)

    return df


if __name__ == "__main__":
    main()
