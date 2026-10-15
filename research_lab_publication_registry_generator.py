"""
================================================================================
SCENARIO 3: APPLIED DATA SCIENCE & STATISTICS RESEARCH LAB REGISTRY
================================================================================
Narrative Context:
This script simulates the bibliometric and publication tracking registry of a
university research collaborative where graduate students, postdocs, and faculty
publish papers across statistics and machine learning venues.

HOW THE MESSINESS DIAL WORKS:
We use a parametric Bernoulli corruption model based on data cleaning literature
(Arocena et al., PVLDB 2015; DAMA-DMBOK Data Quality Dimensions):
    P(corruption) = messiness * BASE_RATE

Messiness Levels Guide:
  - 0.0 (Clean)    : 0% noise. Publication dates strictly follow submission dates,
                     and citation counts follow a clean Negative Binomial distribution.
  - 0.3 (Light)    : ~2% to 10% noise. Standard NaNs, minor venue/domain casing
                     issues, alternate date formats, and rare exact duplicates.
  - 0.6 (Moderate) : ~5% to 20% noise. Blank strings ("") and "N/A", inconsistent
                     boolean representations ("Yes", 1, "Y"), and chronological
                     inversions (publication date earlier than submission date).
  - 1.0 (Heavy)    : ~10% to 35% max ceiling. Sentinel values (-999, "NULL"),
                     impossible calendar dates, negative citation counts, and
                     conflicting duplicate paper IDs.
================================================================================
"""

import numpy as np
import pandas as pd

# ==============================================================================
# 1. USER CONFIGURATION (Change these three values to test different outputs)
# ==============================================================================
num_rows = 1000       # Number of research paper records to generate
random_seed = 42      # Seed for reproducible random numbers
messiness = 0.6       # Messiness dial: choose any value from 0.0 to 1.0

# Ensure messiness stays within the valid [0.0, 1.0] probability scale
messiness = max(0.0, min(1.0, float(messiness)))
np.random.seed(random_seed)

# ==============================================================================
# 2. FIXED BASE RATES (Maximum error ceilings when messiness = 1.0)
# ==============================================================================
BASE_MISSING_RATE = 0.30     # Up to 30% missing/sentinel values per target column
BASE_TEXT_NOISE_RATE = 0.35  # Up to 35% text casing, spacing, or boolean mix-ups
BASE_DATE_NOISE_RATE = 0.25  # Up to 25% mixed date formats or chronological inversions
BASE_OUTLIER_RATE = 0.08     # Up to 8% impossible numeric outliers
BASE_DUPLICATE_RATE = 0.10   # Up to 10% duplicate paper records

# ==============================================================================
# 3. GENERATE CLEAN BASELINE DATASET (Messiness = 0.0)
# ==============================================================================
print(f"Generating {num_rows} clean baseline research publication records...")

# Generate unique Paper IDs
paper_ids = [f"LAB-PUB-{1001 + i}" for i in range(num_rows)]

# Lead author academic roles and research sub-domains
roles = ["Master Student", "PhD Candidate", "Postdoc", "Faculty Member"]
lead_author_roles = np.random.choice(roles, size=num_rows, p=[0.35, 0.35, 0.15, 0.15])

domains = ["Time Series", "Biostatistics", "Machine Learning", "Econometrics", "Spatial Statistics"]
research_domains = np.random.choice(domains, size=num_rows, p=[0.20, 0.25, 0.30, 0.15, 0.10])

venues = ["Journal", "Conference Proceedings", "Symposium"]
target_venues = np.random.choice(venues, size=num_rows, p=[0.55, 0.35, 0.10])

index_list = ["Scopus", "Web of Science", "IEEE Xplore", "PubMed"]
indexing = np.random.choice(index_list, size=num_rows, p=[0.40, 0.30, 0.20, 0.10])

# Generate clean submission dates between 2021 and 2025
sub_years = np.random.choice([2021, 2022, 2023, 2024, 2025], size=num_rows)
sub_months = np.random.randint(1, 13, size=num_rows)
sub_days = np.random.randint(1, 28, size=num_rows)

submission_dates_dt = pd.to_datetime([
    f"{y}-{m:02d}-{d:02d}" for y, m, d in zip(sub_years, sub_months, sub_days)
])

# Publication date is strictly 60 to 300 days AFTER submission date in clean data
review_lag_days = np.random.randint(60, 300, size=num_rows)
publication_dates_dt = submission_dates_dt + pd.to_timedelta(review_lag_days, unit="D")

submission_dates = submission_dates_dt.strftime("%Y-%m-%d").tolist()
publication_dates = publication_dates_dt.strftime("%Y-%m-%d").tolist()

# Peer review rounds (1 to 4) and skewed citation counts (Negative Binomial)
review_rounds = np.random.choice([1, 2, 3, 4], size=num_rows, p=[0.25, 0.45, 0.20, 0.10])
citation_counts = np.random.negative_binomial(n=2, p=0.18, size=num_rows)

# Grant funding status
grant_funded = np.random.choice(["Yes", "No"], size=num_rows, p=[0.40, 0.60])

# Assemble the clean DataFrame
df = pd.DataFrame({
    "paper_id": paper_ids,
    "lead_author_role": lead_author_roles,
    "research_domain": research_domains,
    "target_venue": target_venues,
    "indexing": indexing,
    "submission_date": submission_dates,
    "publication_date": publication_dates,
    "review_rounds": review_rounds,
    "citation_count": citation_counts,
    "grant_funded": grant_funded
})

# Cast numeric columns to object to allow sentinel strings during corruption
df["review_rounds"] = df["review_rounds"].astype(object)
df["citation_count"] = df["citation_count"].astype(object)

# ==============================================================================
# 4. INJECT ERRORS BASED ON MESSINESS DIAL (If messiness > 0.0)
# ==============================================================================
if messiness > 0.0:
    print(f"Applying messiness level {messiness} across all 5 data quality dimensions...")

    # --------------------------------------------------------------------------
    # A. COMPLETENESS: Missing Values (NaN, empty strings, sentinel codes)
    # --------------------------------------------------------------------------
    prob_missing = messiness * BASE_MISSING_RATE
    target_missing_cols = ["lead_author_role", "indexing", "publication_date", "review_rounds", "citation_count", "grant_funded"]

    for col in target_missing_cols:
        mask = np.random.uniform(0.0, 1.0, size=len(df)) < prob_missing
        indices = df[mask].index

        for idx in indices:
            roll = np.random.uniform(0.0, 1.0)
            if messiness < 0.4 or roll < 0.5:
                df.loc[idx, col] = np.nan
            elif messiness < 0.7 or roll < 0.8:
                df.loc[idx, col] = np.random.choice(["", "N/A", "Unrecorded"])
            else:
                if col in ["review_rounds", "citation_count"]:
                    df.loc[idx, col] = np.random.choice([-999, 99999, "NULL"])
                else:
                    df.loc[idx, col] = np.random.choice(["NULL", "NONE", "-999"])

    # --------------------------------------------------------------------------
    # B. CONSISTENCY: Text Casing, Extra Spaces, Typos, & Boolean Mix-ups
    # --------------------------------------------------------------------------
    prob_text = messiness * BASE_TEXT_NOISE_RATE

    for col in ["lead_author_role", "research_domain", "target_venue", "indexing"]:
        mask_txt = np.random.uniform(0.0, 1.0, size=len(df)) < prob_text
        for idx in df[mask_txt].index:
            val = df.loc[idx, col]
            if pd.isna(val) or str(val) in ["", "N/A", "Unrecorded", "NULL", "NONE", "-999"]:
                continue
            val_str = str(val)
            mode = np.random.choice(["lower", "upper", "spaces", "abbrev"])
            if mode == "lower":
                df.loc[idx, col] = val_str.lower()
            elif mode == "upper":
                df.loc[idx, col] = val_str.upper()
            elif mode == "spaces":
                df.loc[idx, col] = f"  {val_str}   "
            elif mode == "abbrev":
                if val_str == "Conference Proceedings":
                    df.loc[idx, col] = np.random.choice(["Conf. Proc.", "Conference", "Conf"])
                elif val_str == "Web of Science":
                    df.loc[idx, col] = np.random.choice(["WoS", "WOS", "Web Of Sci"])
                elif val_str == "Scopus":
                    df.loc[idx, col] = np.random.choice(["Scopos", "SCOPUS ", "scopus"])
                elif val_str == "Machine Learning":
                    df.loc[idx, col] = np.random.choice(["ML", "Machne Learning", "machine  learning"])

    # Corrupt grant_funded boolean consistency ("Yes"/"No" mixed with 1/0, True/False, Y/N)
    mask_bool = np.random.uniform(0.0, 1.0, size=len(df)) < prob_text
    for idx in df[mask_bool].index:
        val = df.loc[idx, "grant_funded"]
        if val == "Yes":
            df.loc[idx, "grant_funded"] = np.random.choice(["yes", "Y", "True", "1", "  Yes "])
        elif val == "No":
            df.loc[idx, "grant_funded"] = np.random.choice(["no", "N", "False", "0", "  No "])

    # --------------------------------------------------------------------------
    # C. VALIDITY: Mixed Date Formats, Impossible Dates, & Chronological Inversions
    # --------------------------------------------------------------------------
    prob_date = messiness * BASE_DATE_NOISE_RATE
    mask_date = np.random.uniform(0.0, 1.0, size=len(df)) < prob_date

    for idx in df[mask_date].index:
        sub_str = str(df.loc[idx, "submission_date"])
        pub_str = str(df.loc[idx, "publication_date"])
        parts = sub_str.split("-")

        if len(parts) == 3:
            y, m, d = parts[0], parts[1], parts[2]
            style = np.random.choice(["dd/mm/yyyy", "mm-dd-yyyy", "time_travel", "invalid_date"])

            if style == "dd/mm/yyyy":
                df.loc[idx, "submission_date"] = f"{d}/{m}/{y}"
            elif style == "mm-dd-yyyy":
                df.loc[idx, "submission_date"] = f"{m}-{d}-{y}"
            elif style == "time_travel" and messiness >= 0.4:
                # Logical violation: Publication date set BEFORE submission date
                if not pd.isna(pub_str) and "-" in pub_str:
                    df.loc[idx, "publication_date"] = f"{int(y) - 1}-{m}-{d}"
            elif style == "invalid_date" and messiness >= 0.6:
                df.loc[idx, "submission_date"] = np.random.choice([
                    f"{y}-02-30", f"{y}-14-05", "2024-00-00", "TBD"
                ])

    # --------------------------------------------------------------------------
    # D. ACCURACY: Numeric Outliers & Domain Rule Violations
    # --------------------------------------------------------------------------
    prob_outlier = messiness * BASE_OUTLIER_RATE

    # Corrupt review_rounds (valid: 1 to 4)
    mask_rev = np.random.uniform(0.0, 1.0, size=len(df)) < prob_outlier
    for idx in df[mask_rev].index:
        df.loc[idx, "review_rounds"] = np.random.choice([0, -1, -3, 18, 45])

    # Corrupt citation_count (negative citations or extreme corrupted values)
    mask_cit = np.random.uniform(0.0, 1.0, size=len(df)) < prob_outlier
    for idx in df[mask_cit].index:
        df.loc[idx, "citation_count"] = np.random.choice([-15, -1, 85000, 999999])

    # --------------------------------------------------------------------------
    # E. UNIQUENESS: Exact Duplicates and Partial Key Collisions
    # --------------------------------------------------------------------------
    prob_dup = messiness * BASE_DUPLICATE_RATE
    num_duplicates = int(len(df) * prob_dup)

    if num_duplicates > 0:
        dup_indices = np.random.choice(df.index, size=num_duplicates, replace=False)
        dup_rows = df.loc[dup_indices].copy()

        # Partial key collision: same paper_id logged with updated citation count
        half = num_duplicates // 2
        if half > 0 and messiness >= 0.4:
            for i in range(half):
                row_idx = dup_rows.index[i]
                dup_rows.loc[row_idx, "citation_count"] = 999

        df = pd.concat([df, dup_rows], ignore_index=True)
        df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

# ==============================================================================
# 5. SAVE OUTPUT AND PRINT DIAGNOSTIC SUMMARY
# ==============================================================================
output_filename = f"messy_research_registry_M{int(messiness * 100):02d}.csv"
df.to_csv(output_filename, index=False)

print("\n" + "=" * 65)
print("GENERATION COMPLETE: RESEARCH LAB PUBLICATION REGISTRY")
print("=" * 65)
print(f"Configured Rows      : {num_rows}")
print(f"Final Row Count      : {len(df)} (includes {len(df) - num_rows} injected duplicate rows)")
print(f"Messiness Level (M)  : {messiness}")
print(f"Standard NaN Count   : {df.isna().sum().sum()} cells")
print(f"Saved Output File    : {output_filename}")
print("=" * 65)
print("\nSample Preview (First 8 rows):")
print(df.head(8).to_string())