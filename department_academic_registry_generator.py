"""
================================================================================
SCENARIO 1: UNIVERSITY DEPARTMENT GRADUATE ACADEMIC & COURSE EVALUATION LOGS
================================================================================
Narrative Context:
This script simulates the academic database of a university Statistics & Data
Science department where graduate students enroll in courses, earn exam scores,
and submit end-of-semester instructor evaluations.

HOW THE MESSINESS DIAL WORKS:
We use a parametric Bernoulli corruption model based on data cleaning literature
(Arocena et al., PVLDB 2015; DAMA-DMBOK Data Quality Dimensions):
    P(corruption) = messiness * BASE_RATE

Messiness Levels Guide:
  - 0.0 (Clean)    : 0% noise. All values obey strict domain rules and types.
  - 0.3 (Light)    : ~2% to 10% noise. Standard NaNs, minor casing/spacing issues,
                     alternate valid date formats, and rare exact duplicates.
  - 0.6 (Moderate) : ~5% to 20% noise. Empty strings ("") and "N/A" nulls, typos
                     in course/instructor names, scores exceeding bounds, and
                     partial duplicate student IDs.
  - 1.0 (Heavy)    : ~10% to 35% max ceiling. Numeric sentinel nulls (-999),
                     impossible calendar dates (e.g., Feb 30), negative exam
                     scores, and severe text formatting corruption.
================================================================================
"""

import numpy as np
import pandas as pd

# ==============================================================================
# 1. USER CONFIGURATION (Change these three values to test different outputs)
# ==============================================================================
num_rows = 1000       # Number of student-course records to generate
random_seed = 42      # Seed for reproducible random numbers
messiness = 0.6       # Messiness dial: choose any value from 0.0 to 1.0

# Ensure messiness stays within the valid [0.0, 1.0] probability scale
messiness = max(0.0, min(1.0, float(messiness)))
np.random.seed(random_seed)

# ==============================================================================
# 2. FIXED BASE RATES (Maximum error ceilings when messiness = 1.0)
# ==============================================================================
BASE_MISSING_RATE = 0.30     # Up to 30% missing/sentinel values per target column
BASE_TEXT_NOISE_RATE = 0.35  # Up to 35% text casing, spacing, or typo issues
BASE_DATE_NOISE_RATE = 0.25  # Up to 25% mixed date formats or invalid dates
BASE_OUTLIER_RATE = 0.08     # Up to 8% impossible numeric outliers
BASE_DUPLICATE_RATE = 0.10   # Up to 10% duplicate rows (exact + partial key)

# ==============================================================================
# 3. GENERATE CLEAN BASELINE DATASET (Messiness = 0.0)
# ==============================================================================
print(f"Generating {num_rows} clean baseline academic records...")

# Generate sequential and realistic graduate Student IDs (e.g., STU-2024-1001)
cohort_years = np.random.choice([2022, 2023, 2024, 2025], size=num_rows, p=[0.15, 0.25, 0.35, 0.25])
student_numbers = np.random.randint(1001, 1450, size=num_rows)
student_ids = [f"STU-{year}-{num}" for year, num in zip(cohort_years, student_numbers)]

# Generate clean enrollment dates in standard ISO format (YYYY-MM-DD)
months = np.random.choice(["01", "07"], size=num_rows)  # Spring (Jan) or Fall (Jul) intake
days = np.random.randint(1, 28, size=num_rows)
enrollment_dates = [f"{year}-{month}-{day:02d}" for year, month, day in zip(cohort_years, months, days)]

# Define realistic graduate courses and faculty instructors
courses = ["STAT-501", "STAT-502", "DATA-503", "STAT-508", "DATA-510"]
instructors = [
    "Dr. A. Rahman",
    "Prof. S. Karim",
    "Dr. M. Chowdhury",
    "Prof. N. Islam",
    "Dr. T. Ahmed"
]

course_codes = np.random.choice(courses, size=num_rows)
instructor_names = np.random.choice(instructors, size=num_rows)

# Generate realistic exam scores using Normal distributions clipped to valid ranges
# Midterm is out of 30; Final exam is out of 70
midterm_scores = np.round(np.clip(np.random.normal(loc=21.5, scale=4.2, size=num_rows), 0.0, 30.0), 1)
final_scores = np.round(np.clip(np.random.normal(loc=51.0, scale=8.5, size=num_rows), 0.0, 70.0), 1)

# Attendance percentage between 50% and 100%
attendance_pct = np.round(np.clip(np.random.normal(loc=86.0, scale=9.0, size=num_rows), 50.0, 100.0), 1)

# Student evaluation of the course (Likert scale 1 to 5) and feedback tags
course_ratings = np.random.choice([1, 2, 3, 4, 5], size=num_rows, p=[0.05, 0.10, 0.25, 0.35, 0.25])
feedback_options = ["Well Structured", "Challenging", "Fast Paced", "Great Examples", "Heavy Workload"]
student_feedback = np.random.choice(feedback_options, size=num_rows)

# Assemble the clean DataFrame
df = pd.DataFrame({
    "student_id": student_ids,
    "enrollment_date": enrollment_dates,
    "cohort_year": cohort_years,
    "course_code": course_codes,
    "instructor_name": instructor_names,
    "midterm_score": midterm_scores,
    "final_score": final_scores,
    "attendance_pct": attendance_pct,
    "course_rating": course_ratings,
    "student_feedback": student_feedback
})

# Convert columns that will receive mixed types (like sentinel strings) to object/float
df["cohort_year"] = df["cohort_year"].astype(object)
df["midterm_score"] = df["midterm_score"].astype(object)
df["final_score"] = df["final_score"].astype(object)
df["attendance_pct"] = df["attendance_pct"].astype(object)
df["course_rating"] = df["course_rating"].astype(object)

# ==============================================================================
# 4. INJECT ERRORS BASED ON MESSINESS DIAL (If messiness > 0.0)
# ==============================================================================
if messiness > 0.0:
    print(f"Applying messiness level {messiness} across all 5 data quality dimensions...")

    # --------------------------------------------------------------------------
    # A. COMPLETENESS: Missing Values (NaN, empty strings, sentinel codes)
    # --------------------------------------------------------------------------
    prob_missing = messiness * BASE_MISSING_RATE
    target_missing_cols = ["instructor_name", "midterm_score", "final_score", "attendance_pct", "student_feedback"]

    for col in target_missing_cols:
        mask = np.random.uniform(0.0, 1.0, size=len(df)) < prob_missing
        indices = df[mask].index

        for idx in indices:
            # At light messiness, mostly standard NaN. At higher messiness, mix in strings and sentinels.
            roll = np.random.uniform(0.0, 1.0)
            if messiness < 0.4 or roll < 0.5:
                df.loc[idx, col] = np.nan
            elif messiness < 0.7 or roll < 0.8:
                df.loc[idx, col] = np.random.choice(["", "N/A", "missing"])
            else:
                # Heavy messiness introduces numeric or uppercase legacy sentinels
                if col in ["midterm_score", "final_score", "attendance_pct"]:
                    df.loc[idx, col] = np.random.choice([-999, 9999, "NULL"])
                else:
                    df.loc[idx, col] = np.random.choice(["NULL", "NONE", "-999"])

    # --------------------------------------------------------------------------
    # B. CONSISTENCY: Text Casing, Extra Spaces, and Typos
    # --------------------------------------------------------------------------
    prob_text = messiness * BASE_TEXT_NOISE_RATE

    # Corrupt course_code (e.g., "stat-501", "STAT 501", "  STAT-501 ")
    mask_course = np.random.uniform(0.0, 1.0, size=len(df)) < prob_text
    for idx in df[mask_course].index:
        val = str(df.loc[idx, "course_code"])
        choice = np.random.choice(["lower", "space", "pad", "typo"])
        if choice == "lower":
            df.loc[idx, "course_code"] = val.lower()
        elif choice == "space":
            df.loc[idx, "course_code"] = val.replace("-", " ")
        elif choice == "pad":
            df.loc[idx, "course_code"] = f"  {val}   "
        elif choice == "typo" and messiness >= 0.5:
            df.loc[idx, "course_code"] = val.replace("STAT", "STTA").replace("DATA", "DTA")

    # Corrupt instructor_name and student_feedback
    for col in ["instructor_name", "student_feedback"]:
        mask_txt = np.random.uniform(0.0, 1.0, size=len(df)) < prob_text
        for idx in df[mask_txt].index:
            val = df.loc[idx, col]
            if pd.isna(val) or str(val) in ["", "N/A", "NULL", "-999"]:
                continue
            val_str = str(val)
            mode = np.random.choice(["upper", "lower", "spaces", "typo"])
            if mode == "upper":
                df.loc[idx, col] = val_str.upper()
            elif mode == "lower":
                df.loc[idx, col] = val_str.lower()
            elif mode == "spaces":
                df.loc[idx, col] = "  " + val_str.replace(" ", "   ") + " "
            elif mode == "typo" and len(val_str) > 5:
                # Swap two adjacent characters to simulate a human typing error
                pos = np.random.randint(1, len(val_str) - 2)
                chars = list(val_str)
                chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
                df.loc[idx, col] = "".join(chars)

    # --------------------------------------------------------------------------
    # C. VALIDITY: Date Format Mix-ups & Impossible Calendar Dates
    # --------------------------------------------------------------------------
    prob_date = messiness * BASE_DATE_NOISE_RATE
    mask_date = np.random.uniform(0.0, 1.0, size=len(df)) < prob_date

    for idx in df[mask_date].index:
        date_str = str(df.loc[idx, "enrollment_date"])
        parts = date_str.split("-")
        if len(parts) == 3:
            y, m, d = parts[0], parts[1], parts[2]
            style = np.random.choice(["dd/mm/yyyy", "mm-dd-yyyy", "slash_iso", "invalid_date"])
            if style == "dd/mm/yyyy":
                df.loc[idx, "enrollment_date"] = f"{d}/{m}/{y}"
            elif style == "mm-dd-yyyy":
                df.loc[idx, "enrollment_date"] = f"{m}-{d}-{y}"
            elif style == "slash_iso":
                df.loc[idx, "enrollment_date"] = f"{y}/{m}/{d}"
            elif style == "invalid_date" and messiness >= 0.5:
                # Inject impossible calendar dates
                df.loc[idx, "enrollment_date"] = np.random.choice([
                    f"{y}-02-30", f"{y}-13-15", f"{y}-00-10", "Invalid Date"
                ])

    # --------------------------------------------------------------------------
    # D. ACCURACY: Numeric Outliers & Domain Rule Violations
    # --------------------------------------------------------------------------
    prob_outlier = messiness * BASE_OUTLIER_RATE

    # Corrupt midterm_score (valid: 0 to 30)
    mask_mid = np.random.uniform(0.0, 1.0, size=len(df)) < prob_outlier
    for idx in df[mask_mid].index:
        df.loc[idx, "midterm_score"] = np.random.choice([-12.5, -5.0, 45.0, 120.0, 300.0])

    # Corrupt final_score (valid: 0 to 70)
    mask_fin = np.random.uniform(0.0, 1.0, size=len(df)) < prob_outlier
    for idx in df[mask_fin].index:
        df.loc[idx, "final_score"] = np.random.choice([-20.0, 95.5, 150.0, 700.0])

    # Corrupt attendance_pct (valid: 0 to 100)
    mask_att = np.random.uniform(0.0, 1.0, size=len(df)) < prob_outlier
    for idx in df[mask_att].index:
        df.loc[idx, "attendance_pct"] = np.random.choice([-15.0, 112.5, 145.0, 980.0])

    # Corrupt cohort_year and course_rating
    mask_yr = np.random.uniform(0.0, 1.0, size=len(df)) < (prob_outlier * 0.5)
    for idx in df[mask_yr].index:
        df.loc[idx, "cohort_year"] = np.random.choice([1924, 2099, -2024])

    mask_rat = np.random.uniform(0.0, 1.0, size=len(df)) < prob_outlier
    for idx in df[mask_rat].index:
        df.loc[idx, "course_rating"] = np.random.choice([0, 6, 10, -1, "Five", "Good"])

    # --------------------------------------------------------------------------
    # E. UNIQUENESS: Exact Duplicates and Partial Key Collisions
    # --------------------------------------------------------------------------
    prob_dup = messiness * BASE_DUPLICATE_RATE
    num_duplicates = int(len(df) * prob_dup)

    if num_duplicates > 0:
        dup_indices = np.random.choice(df.index, size=num_duplicates, replace=False)
        dup_rows = df.loc[dup_indices].copy()

        # Split duplicates into exact copies vs. partial key variations
        half = num_duplicates // 2
        if half > 0 and messiness >= 0.4:
            # Modify midterm_score slightly or change casing on student_id for partial duplicates
            for i in range(half):
                row_idx = dup_rows.index[i]
                dup_rows.loc[row_idx, "student_id"] = str(dup_rows.loc[row_idx, "student_id"]).lower()

        # Append duplicates and shuffle the dataset
        df = pd.concat([df, dup_rows], ignore_index=True)
        df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

# ==============================================================================
# 5. SAVE OUTPUT AND PRINT DIAGNOSTIC SUMMARY
# ==============================================================================
output_filename = f"messy_department_academics_M{int(messiness * 100):02d}.csv"
df.to_csv(output_filename, index=False)

print("\n" + "=" * 65)
print("GENERATION COMPLETE: UNIVERSITY DEPARTMENT ACADEMIC LOGS")
print("=" * 65)
print(f"Configured Rows      : {num_rows}")
print(f"Final Row Count      : {len(df)} (includes {len(df) - num_rows} injected duplicate rows)")
print(f"Messiness Level (M)  : {messiness}")
print(f"Standard NaN Count   : {df.isna().sum().sum()} cells")
print(f"Saved Output File    : {output_filename}")
print("=" * 65)
print("\nSample Preview (First 8 rows):")
print(df.head(8).to_string())