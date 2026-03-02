"""
Evaluation script - measures how well the engine's recommendations
match the ground truth labels in train_set.csv.

SHL scores submissions on Mean Recall@10, so that's the primary metric
I'm optimizing for. I also track Recall at smaller K values and MAP@10
to understand where the engine is strong vs weak.

The trickiest part was resolving assessment names from the train set to
catalog URLs, since they don't always match exactly (unicode dashes,
missing prefixes, etc.). I built a fuzzy matcher with manual overrides
for the worst cases.

    $ python -m src.evaluate

Author: Mohammad Inayat Hussain
"""

import sys
import os
import re
import csv
from pathlib import Path
from difflib import SequenceMatcher

# Fix Windows console encoding
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.engine import AssessmentEngine

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAIN_FILE = DATA_DIR / "train_set.csv"
CATALOG_FILE = DATA_DIR / "shl_catalog_clean.csv"



# Manual overrides for train set names that fuzzy matching can't handle.
# These are assessed by comparing the train set against the actual catalog.
TRAIN_NAME_OVERRIDES: dict[str, str] = {
    "Verify Interactive - Numerical Ability": "SHL Verify Interactive \u2013 Numerical Reasoning",
    "Verify Interactive - Verbal Ability": "Verify - Verbal Ability - Next Generation",
    "Verify Interactive - Inductive Reasoning": "SHL Verify Interactive - Inductive Reasoning",
    "Verify Interactive - Deductive Reasoning": "SHL Verify Interactive \u2013 Deductive Reasoning",
    "OPQ32r": "Occupational Personality Questionnaire OPQ32r",
    "Motivation Questionnaire": "Motivation Questionnaire MQM5",
    "Contact Center Simulation": "Contact Center Call Simulation (New)",
    "Customer Service Simulation": "Customer Service Phone Simulation",
    "Computer Literacy (New)": "Basic Computer Literacy (Windows 10) (New)",
    "Occupational Personality Questionnaire": "Occupational Personality Questionnaire OPQ32r",
    "Graduate 8.0 Job Focused Assessment": "Apprentice 8.0 Job Focused Assessment",
    "Sales Representative Solution": "Sales & Service Phone Solution",
    "dot NET Framework 4.5": ".NET Framework 4.5",
    "Managerial and Professional Profiler": "OPQ Manager Plus Report",
    "CCSQ Managerial Skills Profile": "OPQ Manager Plus Report 2.0",
    "General Clerical Test (New)": "Data Entry (New)",
    "Leadership Edge 360": "Enterprise Leadership Report 1.0",
}


def build_name_url_map(catalog_df: pd.DataFrame) -> dict[str, str]:
    """Build a lookup from assessment name -> URL."""
    return dict(zip(catalog_df["assessment_name"], catalog_df["url"]))


def normalize_name(name: str) -> str:
    """Normalize an assessment name for comparison."""
    n = name.lower().strip()
    # Normalize unicode dashes
    n = n.replace("\u2013", "-").replace("\u2014", "-").replace("\u2012", "-")
    # Remove common prefixes
    for prefix in ["shl ", "shl- "]:
        if n.startswith(prefix):
            n = n[len(prefix):]
    # Normalize whitespace
    n = re.sub(r"\s+", " ", n).strip()
    return n


def fuzzy_match_name(target_name: str, catalog_names: list[str], threshold: float = 0.45) -> str | None:
    """
    Find the best fuzzy match for a target name in the catalog.
    
    Priority: manual override > exact match > normalized substring > fuzzy score.
    """
    # Check manual override first
    if target_name.strip() in TRAIN_NAME_OVERRIDES:
        override = TRAIN_NAME_OVERRIDES[target_name.strip()]
        if override in catalog_names:
            return override
    
    target_norm = normalize_name(target_name)
    
    best_match = None
    best_score = 0.0
    
    for catalog_name in catalog_names:
        catalog_norm = normalize_name(catalog_name)
        
        # Exact match after normalization
        if target_norm == catalog_norm:
            return catalog_name
        
        # Substring check (after normalization)
        if target_norm in catalog_norm or catalog_norm in target_norm:
            score = 0.92
        else:
            # SequenceMatcher on normalized names
            score = SequenceMatcher(None, target_norm, catalog_norm).ratio()
        
        # Bonus for matching key tokens
        target_tokens = set(re.findall(r'\w+', target_norm))
        catalog_tokens = set(re.findall(r'\w+', catalog_norm))
        if target_tokens and catalog_tokens:
            token_overlap = len(target_tokens & catalog_tokens) / max(len(target_tokens), 1)
        else:
            token_overlap = 0.0
        
        # Weighted score
        combined_score = 0.55 * score + 0.45 * token_overlap
        
        if combined_score > best_score:
            best_score = combined_score
            best_match = catalog_name
    
    if best_score >= threshold:
        return best_match
    
    return None


def resolve_assessment_urls(
    assessment_names: list[str],
    catalog_df: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """
    Resolve assessment names from train set to catalog URLs.
    
    Does manual override > exact match > fuzzy matching.
    """
    name_url_map = build_name_url_map(catalog_df)
    catalog_names = list(name_url_map.keys())
    
    found_urls = []
    unresolved = []
    
    for name in assessment_names:
        name = name.strip()
        
        # Exact match
        if name in name_url_map:
            found_urls.append(name_url_map[name])
            continue
        
        # Fuzzy match (includes manual overrides)
        match = fuzzy_match_name(name, catalog_names)
        if match:
            found_urls.append(name_url_map[match])
        else:
            unresolved.append(name)
    
    return found_urls, unresolved



def recall_at_k(recommended_urls: list[str], relevant_urls: list[str], k: int) -> float:
    """
    Compute Recall@K.
    
    Recall@K = |relevant found in top-K| / |total relevant|
    
    Args:
        recommended_urls: List of URL strings from engine recommendations.
        relevant_urls: List of ground-truth relevant URLs.
        k: Number of top results to consider.
    
    Returns:
        Recall score (0.0 to 1.0).
    """
    if not relevant_urls:
        return 0.0
    
    top_k_urls = set(recommended_urls[:k])
    relevant_set = set(relevant_urls)
    
    found = len(top_k_urls & relevant_set)
    return found / len(relevant_set)


def average_precision_at_k(recommended_urls: list[str], relevant_urls: list[str], k: int) -> float:
    """
    Compute Average Precision@K.
    
    AP@K = (1/|relevant|) * sum(P@i * rel(i)) for i = 1..K
    where P@i = precision at position i, rel(i) = 1 if item at i is relevant.
    
    Args:
        recommended_urls: List of URL strings from engine recommendations.
        relevant_urls: List of ground-truth relevant URLs.
        k: Number of top results to consider.
    
    Returns:
        Average precision score (0.0 to 1.0).
    """
    if not relevant_urls:
        return 0.0
    
    relevant_set = set(relevant_urls)
    hits = 0
    precision_sum = 0.0
    
    for i, url in enumerate(recommended_urls[:k]):
        if url in relevant_set:
            hits += 1
            precision_sum += hits / (i + 1)
    
    return precision_sum / len(relevant_set)



def evaluate():
    """Run full evaluation and print report."""
    
    print("\n" + "#" * 70)
    print("#  SHL SmartMatch AI — Evaluation Report")
    print("#" * 70)
    
    # Load train set
    if not TRAIN_FILE.exists():
        print(f"ERROR: Train set not found at {TRAIN_FILE}")
        return
    
    train_df = pd.read_csv(TRAIN_FILE)
    print(f"\nLoaded {len(train_df)} labeled queries from train_set.csv")
    
    # Load catalog for name resolution
    catalog_df = pd.read_csv(CATALOG_FILE)
    print(f"Loaded {len(catalog_df)} assessments from catalog")
    
    # Initialize engine
    engine = AssessmentEngine()
    
    # Collect metrics
    recalls_at_1 = []
    recalls_at_3 = []
    recalls_at_5 = []
    recalls_at_10 = []
    aps_at_10 = []
    
    total_relevant = 0
    total_found_at_10 = 0
    total_unresolvable = 0
    
    print("\n" + "=" * 70)
    print("  PER-QUERY RESULTS")
    print("=" * 70)
    
    for idx, row in train_df.iterrows():
        query = str(row["query"])
        raw_assessments = str(row["relevant_assessments"])
        
        # Parse relevant assessment names
        relevant_names = [n.strip() for n in raw_assessments.split(",") if n.strip()]
        
        # Resolve names to URLs
        relevant_urls, unresolved = resolve_assessment_urls(relevant_names, catalog_df)
        
        if unresolved:
            total_unresolvable += len(unresolved)
        
        # Get recommendations
        results = engine.recommend(query, top_k=10)
        recommended_urls = [r["url"] for r in results]
        recommended_names = [r["assessment_name"] for r in results]
        
        # Compute metrics
        r_at_1 = recall_at_k(recommended_urls, relevant_urls, 1)
        r_at_3 = recall_at_k(recommended_urls, relevant_urls, 3)
        r_at_5 = recall_at_k(recommended_urls, relevant_urls, 5)
        r_at_10 = recall_at_k(recommended_urls, relevant_urls, 10)
        ap_at_10 = average_precision_at_k(recommended_urls, relevant_urls, 10)
        
        recalls_at_1.append(r_at_1)
        recalls_at_3.append(r_at_3)
        recalls_at_5.append(r_at_5)
        recalls_at_10.append(r_at_10)
        aps_at_10.append(ap_at_10)
        
        total_relevant += len(relevant_urls)
        found_count = len(set(recommended_urls[:10]) & set(relevant_urls))
        total_found_at_10 += found_count
        
        # Print query results
        print(f"\n--- Query {idx+1}: \"{query[:80]}{'...' if len(query)>80 else ''}\"")
        print(f"    Relevant assessments: {len(relevant_names)} "
              f"(resolved to {len(relevant_urls)} URLs, {len(unresolved)} unresolvable)")
        print(f"    Recall@1={r_at_1:.2f}  @3={r_at_3:.2f}  @5={r_at_5:.2f}  @10={r_at_10:.2f}  AP@10={ap_at_10:.2f}")
        print(f"    Found {found_count}/{len(relevant_urls)} relevant in top 10")
        
        # Show what was found
        relevant_url_set = set(relevant_urls)
        for i, r in enumerate(results[:10], 1):
            is_match = "[*]" if r["url"] in relevant_url_set else "[ ]"
            print(f"      {i:2d}. {is_match} [{r['score']:.3f}] {r['assessment_name'][:50]}")
        
        # Show what was missed
        found_urls_set = set(recommended_urls[:10])
        missed_urls = [u for u in relevant_urls if u not in found_urls_set]
        if missed_urls:
            print(f"    MISSED ({len(missed_urls)}):")
            for url in missed_urls:
                name = catalog_df[catalog_df["url"] == url]["assessment_name"].values
                name_str = name[0] if len(name) > 0 else "unknown"
                print(f"      - {name_str}")
        
        if unresolved:
            print(f"    UNRESOLVABLE (not in catalog):")
            for name in unresolved:
                print(f"      ? {name}")
    
    # ═════════════════════════════════════════════════════════════════════
    # FINAL METRICS SUMMARY
    # ═════════════════════════════════════════════════════════════════════
    
    mean_r1 = sum(recalls_at_1) / len(recalls_at_1) if recalls_at_1 else 0
    mean_r3 = sum(recalls_at_3) / len(recalls_at_3) if recalls_at_3 else 0
    mean_r5 = sum(recalls_at_5) / len(recalls_at_5) if recalls_at_5 else 0
    mean_r10 = sum(recalls_at_10) / len(recalls_at_10) if recalls_at_10 else 0
    map_10 = sum(aps_at_10) / len(aps_at_10) if aps_at_10 else 0
    
    print("\n" + "=" * 70)
    print("  FINAL METRICS SUMMARY")
    print("=" * 70)
    
    print(f"""
    Queries evaluated:    {len(train_df)}
    Total relevant:       {total_relevant} (resolvable to catalog URLs)
    Total found @10:      {total_found_at_10}
    Unresolvable names:   {total_unresolvable} (not in catalog)
    
    ┌──────────────────────────────────────┐
    │  Mean Recall@1:    {mean_r1:.4f}            │
    │  Mean Recall@3:    {mean_r3:.4f}            │
    │  Mean Recall@5:    {mean_r5:.4f}            │
    │  Mean Recall@10:   {mean_r10:.4f}            │
    │  MAP@10:           {map_10:.4f}            │
    └──────────────────────────────────────┘
    """)
    
    # Grade
    if mean_r10 >= 0.8:
        grade = "EXCELLENT"
    elif mean_r10 >= 0.6:
        grade = "GOOD"
    elif mean_r10 >= 0.4:
        grade = "ACCEPTABLE"
    else:
        grade = "NEEDS IMPROVEMENT"
    
    print(f"    Grade: {grade}")
    print(f"    Target: Mean Recall@10 >= 0.50 for submission")
    
    # Per-query recall table
    print(f"\n    Per-Query Recall@10:")
    print(f"    {'Query':<55}  R@10")
    print(f"    {'-'*55}  ----")
    for i, (r10, row) in enumerate(zip(recalls_at_10, train_df.itertuples())):
        q = str(row.query)[:55]
        bar = "|" + "#" * int(r10 * 20) + " " * (20 - int(r10 * 20)) + "|"
        print(f"    {q:<55}  {r10:.2f} {bar}")
    
    print()
    return mean_r10


if __name__ == "__main__":
    evaluate()
