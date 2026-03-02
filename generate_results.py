"""
Generates the final results.csv for SHL submission.

Loads the 9 unlabeled test queries, runs each through the recommendation
engine, and writes out the CSV in the exact format SHL's automated
pipeline expects: two columns, 'Query' and 'Assessment_url', with
5-10 URL rows per query.

    $ python generate_results.py

Author: Mohammad Inayat Hussain
"""

import sys
import os
import pandas as pd
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.engine import AssessmentEngine


TEST_SET_PATH = Path("data/test_set.csv")
OUTPUT_DIR = Path("results")
OUTPUT_PATH = OUTPUT_DIR / "results.csv"
TOP_K = 10  # Max 10 recommendations per query


def main():
    # ── Validate test set exists ──
    if not TEST_SET_PATH.exists():
        print(f"ERROR: Test set not found at {TEST_SET_PATH}")
        sys.exit(1)

    # ── Load test queries ──
    test_df = pd.read_csv(TEST_SET_PATH)
    queries = test_df["query"].dropna().tolist()
    print(f"Loaded {len(queries)} test queries from {TEST_SET_PATH}")

    # ── Initialize engine ──
    print("Initializing AssessmentEngine...")
    engine = AssessmentEngine()
    print("Engine ready.\n")

    # ── Generate predictions ──
    rows = []
    for i, query in enumerate(queries, 1):
        query = query.strip()
        print(f"[{i}/{len(queries)}] Processing: \"{query[:80]}{'...' if len(query) > 80 else ''}\"")

        results = engine.recommend(query, top_k=TOP_K)

        if not results:
            print(f"  WARNING: No results for query {i}")
            continue

        for r in results:
            url = r.get("url", "").strip()
            if url and url.startswith("http"):
                rows.append({
                    "Query": query,
                    "Assessment_url": url,
                })

        print(f"  -> {len([r for r in results if r.get('url', '').startswith('http')])} URLs added")

    # ── Create output DataFrame ──
    result_df = pd.DataFrame(rows, columns=["Query", "Assessment_url"])

    # ── Validate ──
    print(f"\n{'='*60}")
    print(f"VALIDATION")
    print(f"{'='*60}")
    print(f"  Total rows: {len(result_df)}")
    print(f"  Unique queries: {result_df['Query'].nunique()}")
    print(f"  Column names: {list(result_df.columns)}")

    # Check URLs are valid
    bad_urls = result_df[~result_df["Assessment_url"].str.startswith("https://")]
    if len(bad_urls) > 0:
        print(f"  WARNING: {len(bad_urls)} non-HTTPS URLs found!")
    else:
        print(f"  All URLs are valid HTTPS links [OK]")

    # Check no local paths leaked
    local_patterns = ["C:\\", "c:\\", "/tmp/", "/home/", "\\Users\\"]
    for pat in local_patterns:
        leaked = result_df[result_df["Assessment_url"].str.contains(pat, na=False, regex=False)]
        if len(leaked) > 0:
            print(f"  ERROR: Local path '{pat}' found in URLs!")
            sys.exit(1)
    print(f"  No local paths in URLs [OK]")

    # Rows per query
    for query in queries:
        count = len(result_df[result_df["Query"] == query.strip()])
        status = "[OK]" if 5 <= count <= 10 else "[!]"
        print(f"  Query {queries.index(query)+1}: {count} recommendations {status}")

    # ── Save ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"\n{'='*60}")
    print(f"SAVED: {OUTPUT_PATH}")
    print(f"{'='*60}")

    # ── Show first few rows ──
    print(f"\nPreview (first 15 rows):")
    print(result_df.head(15).to_string(index=False))

    print(f"\nDone! File ready for submission: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
