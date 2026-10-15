"""
================================================================================
SCENARIO 2: E-COMMERCE OPERATIONS & ORDER TRANSACTIONS LOG
================================================================================
Narrative Context:
This script simulates the operational transaction database of an e-commerce
company where our graduate student works part-time as a Data Analyst. It tracks
customer orders, product categories, pricing, discounts, local payment methods,
and delivery statuses.

HOW THE MESSINESS DIAL WORKS:
We use a parametric Bernoulli corruption model based on data cleaning literature
(Arocena et al., PVLDB 2015; DAMA-DMBOK Data Quality Dimensions):
    P(corruption) = messiness * BASE_RATE

Messiness Levels Guide:
  - 0.0 (Clean)    : 0% noise. Clean transaction logs with valid arithmetic bounds.
  - 0.3 (Light)    : ~2% to 10% noise. Standard NaNs, minor payment/category casing
                     variations, alternate date formats, and rare retry duplicates.
  - 0.6 (Moderate) : ~5% to 20% noise. Blank strings ("") and "N/A", typos in
                     categories, negative prices/quantities, and duplicate Order IDs.
  - 1.0 (Heavy)    : ~10% to 35% max ceiling. Sentinel values (-999, "NULL"),
                     impossible calendar dates, 100x price scaling errors, and
                     heavy duplicate records.
================================================================================
"""

import numpy as np
import pandas as pd

# ==============================================================================
# 1. USER CONFIGURATION (Change these three values to test different outputs)
# ==============================================================================
num_rows = 1000       # Number of order transactions to generate
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
BASE_DUPLICATE_RATE = 0.10   # Up to 10% duplicate rows from network retries

# ==============================================================================
# 3. GENERATE CLEAN BASELINE DATASET (Messiness = 0.0)
# ==============================================================================
print(f"Generating {num_rows} clean baseline e-commerce transactions...")

# Generate unique Order IDs and recurring Customer IDs
order_ids = [f"ORD-{100001 + i}" for i in range(num_rows)]
customer_ids = [f"CUST-{np.random.randint(2001, 2450)}" for _ in range(num_rows)]

# Generate clean order dates across 2024 and 2025 in YYYY-MM-DD format
years = np.random.choice([2024, 2025], size=num_rows, p=[0.45, 0.55])
months = np.random.randint(1, 13, size=num_rows)
days = np.random.randint(1, 29, size=num_rows)
order_dates = [f"{y}-{m:02d}-{d:02d}" for y, m, d in zip(years, months, days)]

# Product categories and realistic skewed unit prices (Log-Normal distribution)
categories = ["Electronics", "Home & Living", "Books & Stationery", "Apparel", "Accessories"]
product_categories = np.random.choice(categories, size=num_rows, p=[0.20, 0.25, 0.20, 0.20, 0.15])

# Generate positively skewed unit prices (e.g., between 150 and 25,000 BDT)
raw_prices = np.random.lognormal(mean=6.8, sigma=0.85, size=num_rows)
unit_prices = np.round(np.clip(raw_prices, 120.0, 35000.0), 2)

# Discrete order quantities (mostly 1 to 3 items, occasionally up to 8)
quantities = np.random.choice([1, 2, 3, 4, 5, 8], size=num_rows, p=[0.50, 0.25, 0.12, 0.06, 0.04, 0.03])

# Promotional discount percentages and standard shipping fee tiers
discount_pcts = np.random.choice([0.00, 0.05, 0.10, 0.15, 0.20, 0.25], size=num_rows, p=[0.45, 0.15, 0.15, 0.10, 0.10, 0.05])
shipping_fees = np.random.choice([60.0, 100.0, 150.0], size=num_rows, p=[0.50, 0.35, 0.15])

# Payment methods and order fulfillment statuses
payment_methods = np.random.choice(
    ["bKash", "Nagad", "Credit Card", "Cash on Delivery"],
    size=num_rows,
    p=[0.35, 0.25, 0.15, 0.25]
)
order_statuses = np.random.choice(
    ["Delivered", "In Transit", "Processing", "Cancelled", "Returned"],
    size=num_rows,
    p=[0.65, 0.12, 0.08, 0.09, 0.06]
)

# Assemble the clean DataFrame
df = pd.DataFrame({
    "order_id": order_ids,
    "customer_id": customer_ids,
    "order_date": order_dates,
    "product_category": product_categories,
    "unit_price": unit_prices,
    "quantity": quantities,
    "discount_pct": discount_pcts,
    "shipping_fee": shipping_fees,
    "payment_method": payment_methods,
    "order_status": order_statuses
})

# Cast numeric columns to object so sentinel strings or corruptions can be inserted cleanly
df["unit_price"] = df["unit_price"].astype(object)
df["quantity"] = df["quantity"].astype(object)
df["discount_pct"] = df["discount_pct"].astype(object)
df["shipping_fee"] = df["shipping_fee"].astype(object)

# ==============================================================================
# 4. INJECT ERRORS BASED ON MESSINESS DIAL (If messiness > 0.0)
# ==============================================================================
if messiness > 0.0:
    print(f"Applying messiness level {messiness} across all 5 data quality dimensions...")

    # --------------------------------------------------------------------------
    # A. COMPLETENESS: Missing Values (NaN, empty strings, sentinel codes)
    # --------------------------------------------------------------------------
    prob_missing = messiness * BASE_MISSING_RATE
    target_missing_cols = ["customer_id", "unit_price", "discount_pct", "shipping_fee", "payment_method", "order_status"]

    for col in target_missing_cols:
        mask = np.random.uniform(0.0, 1.0, size=len(df)) < prob_missing
        indices = df[mask].index

        for idx in indices:
            roll = np.random.uniform(0.0, 1.0)
            if messiness < 0.4 or roll < 0.5:
                df.loc[idx, col] = np.nan
            elif messiness < 0.7 or roll < 0.8:
                df.loc[idx, col] = np.random.choice(["", "N/A", "unknown"])
            else:
                if col in ["unit_price", "discount_pct", "shipping_fee"]:
                    df.loc[idx, col] = np.random.choice([-999, -1, 99999, "NULL"])
                else:
                    df.loc[idx, col] = np.random.choice(["NULL", "NONE", "-999"])

    # --------------------------------------------------------------------------
    # B. CONSISTENCY: Text Casing, Extra Spaces, and Typos
    # --------------------------------------------------------------------------
    prob_text = messiness * BASE_TEXT_NOISE_RATE

    for col in ["product_category", "payment_method", "order_status"]:
        mask_txt = np.random.uniform(0.0, 1.0, size=len(df)) < prob_text
        for idx in df[mask_txt].index:
            val = df.loc[idx, col]
            if pd.isna(val) or str(val) in ["", "N/A", "unknown", "NULL", "NONE", "-999"]:
                continue
            val_str = str(val)
            mode = np.random.choice(["lower", "upper", "spaces", "typo"])
            if mode == "lower":
                df.loc[idx, col] = val_str.lower()
            elif mode == "upper":
                df.loc[idx, col] = val_str.upper()
            elif mode == "spaces":
                df.loc[idx, col] = f"   {val_str}  "
            elif mode == "typo" and len(val_str) > 4:
                # Realistic domain typos
                if val_str == "bKash":
                    df.loc[idx, col] = np.random.choice(["bkash ", "bKsh", "B-Kash"])
                elif val_str == "Electronics":
                    df.loc[idx, col] = np.random.choice(["Electrnics", "electronic", "Elctronics"])
                elif val_str == "Delivered":
                    df.loc[idx, col] = np.random.choice(["Deliverd", "delivred", "DELIVERED "])
                else:
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
        date_str = str(df.loc[idx, "order_date"])
        parts = date_str.split("-")
        if len(parts) == 3:
            y, m, d = parts[0], parts[1], parts[2]
            style = np.random.choice(["dd/mm/yyyy", "mm-dd-yyyy", "dot_format", "invalid_date"])
            if style == "dd/mm/yyyy":
                df.loc[idx, "order_date"] = f"{d}/{m}/{y}"
            elif style == "mm-dd-yyyy":
                df.loc[idx, "order_date"] = f"{m}-{d}-{y}"
            elif style == "dot_format":
                df.loc[idx, "order_date"] = f"{d}.{m}.{y}"
            elif style == "invalid_date" and messiness >= 0.5:
                df.loc[idx, "order_date"] = np.random.choice([
                    f"{y}-02-31", f"{y}-15-01", "2099-99-99", "Pending"
                ])

    # --------------------------------------------------------------------------
    # D. ACCURACY: Numeric Outliers & Domain Rule Violations
    # --------------------------------------------------------------------------
    prob_outlier = messiness * BASE_OUTLIER_RATE

    # Corrupt unit_price (negative prices or 100x decimal entry mistakes)
    mask_price = np.random.uniform(0.0, 1.0, size=len(df)) < prob_outlier
    for idx in df[mask_price].index:
        df.loc[idx, "unit_price"] = np.random.choice([-450.0, -1200.50, 0.0, 9500000.0])

    # Corrupt quantity (zero, negative, or impossible fractional quantities)
    mask_qty = np.random.uniform(0.0, 1.0, size=len(df)) < prob_outlier
    for idx in df[mask_qty].index:
        df.loc[idx, "quantity"] = np.random.choice([0, -2, -5, 2.5, 999])

    # Corrupt discount_pct (discounts > 100% or negative)
    mask_disc = np.random.uniform(0.0, 1.0, size=len(df)) < prob_outlier
    for idx in df[mask_disc].index:
        df.loc[idx, "discount_pct"] = np.random.choice([-0.15, 1.50, 25.0, 120.0])

    # Corrupt shipping_fee (negative shipping fees)
    mask_ship = np.random.uniform(0.0, 1.0, size=len(df)) < (prob_outlier * 0.6)
    for idx in df[mask_ship].index:
        df.loc[idx, "shipping_fee"] = np.random.choice([-60.0, -100.0, 50000.0])

    # --------------------------------------------------------------------------
    # E. UNIQUENESS: Exact Duplicates and Partial Key Collisions
    # --------------------------------------------------------------------------
    prob_dup = messiness * BASE_DUPLICATE_RATE
    num_duplicates = int(len(df) * prob_dup)

    if num_duplicates > 0:
        dup_indices = np.random.choice(df.index, size=num_duplicates, replace=False)
        dup_rows = df.loc[dup_indices].copy()

        # Simulate partial duplicates where the same order_id has a different status
        half = num_duplicates // 2
        if half > 0 and messiness >= 0.4:
            for i in range(half):
                row_idx = dup_rows.index[i]
                dup_rows.loc[row_idx, "order_status"] = "Processing"

        df = pd.concat([df, dup_rows], ignore_index=True)
        df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

# ==============================================================================
# 5. SAVE OUTPUT AND PRINT DIAGNOSTIC SUMMARY
# ==============================================================================
output_filename = f"messy_ecommerce_orders_M{int(messiness * 100):02d}.csv"
df.to_csv(output_filename, index=False)

print("\n" + "=" * 65)
print("GENERATION COMPLETE: E-COMMERCE ORDER TRANSACTIONS LOG")
print("=" * 65)
print(f"Configured Rows      : {num_rows}")
print(f"Final Row Count      : {len(df)} (includes {len(df) - num_rows} injected duplicate rows)")
print(f"Messiness Level (M)  : {messiness}")
print(f"Standard NaN Count   : {df.isna().sum().sum()} cells")
print(f"Saved Output File    : {output_filename}")
print("=" * 65)
print("\nSample Preview (First 8 rows):")
print(df.head(8).to_string())