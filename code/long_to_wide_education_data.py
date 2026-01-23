import pandas as pd
import numpy as np
import re

# ----------------------------
# Paths
# ----------------------------
input_path = "GBR_Q-filtered-education-2026-01-20.csv"
output_path = "GBR_Q_education_wide_1995_onwards.csv"

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv(input_path)

# Keep only required columns (education groups are in classif2.label)
df = df[["time", "sex.label", "classif2.label", "obs_value"]].copy()

# ----------------------------
# Keep time from 1995 onwards (expects '1995Q1' format)
# ----------------------------
df["time"] = df["time"].astype(str).str.strip()
df["year"] = df["time"].str.slice(0, 4).astype(int)
df = df[df["year"] >= 1995].copy()
df.drop(columns=["year"], inplace=True)

# Clean labels
df["sex.label"] = df["sex.label"].astype(str).str.strip()
df["classif2.label"] = df["classif2.label"].astype(str).str.strip()

# Prefer overall population (sex = Total) if available
if (df["sex.label"].str.lower() == "total").any():
    df = df[df["sex.label"].str.lower() == "total"].copy()

# ----------------------------
# Combine categories:
# Less than basic + Basic -> Basic and below basic
# ----------------------------
def map_edu_group(x: str) -> str:
    s = str(x).strip()
    s_low = s.lower()
    if "less than basic" in s_low or re.search(r"(^|:\s*)basic$", s_low):
        return "Education (Aggregate levels): Basic and below basic"
    return s

df["edu_group"] = df["classif2.label"].apply(map_edu_group)

# After mapping, sum values within each (time, edu_group)
df_agg = (
    df.groupby(["time", "edu_group"], as_index=False)["obs_value"]
      .sum()
)

# ----------------------------
# Pivot: edu_group -> columns
# ----------------------------
wide = (
    df_agg.pivot_table(
        index="time",
        columns="edu_group",
        values="obs_value",
        aggfunc="sum"
    )
    .reset_index()
)

# ----------------------------
# Identify the "Total" education column robustly
# ----------------------------
def is_total_col(col_name: str) -> bool:
    s = str(col_name).strip().lower()
    return (s == "total") or s.endswith(": total") or re.search(r"(^|[:\s])total$", s) is not None

cat_cols_all = [c for c in wide.columns if c != "time"]
total_candidates = [c for c in cat_cols_all if is_total_col(c)]

total_col = None
if len(total_candidates) == 1:
    total_col = total_candidates[0]
elif len(total_candidates) > 1:
    total_col = max(total_candidates, key=lambda c: wide[c].notna().sum())

# Official total series
if total_col is not None:
    wide["total"] = wide[total_col]
else:
    wide["total"] = np.nan

# Categories excluding total column (these will be used for calculated_total and shares)
cat_cols = [c for c in cat_cols_all if c != total_col]

# ----------------------------
# Totals, shares, difference
# ----------------------------
wide["calculated_total"] = wide[cat_cols].sum(axis=1, skipna=True)

# Use official total when available; otherwise fall back to calculated_total
denom = wide["total"].where(wide["total"].notna(), wide["calculated_total"])

for c in cat_cols:
    wide[f"{c}_share_percent"] = wide[c] / denom * 100

wide["difference_total_vs_calculated"] = wide["total"] - wide["calculated_total"]

# Drop the original total category column (optional but usually cleaner)
if total_col is not None and total_col in wide.columns:
    wide = wide.drop(columns=[total_col])

# ----------------------------
# Reorder columns
# ----------------------------
# ----------------------------
# Reorder columns (FORCE category order)
# ----------------------------

preferred_order = [
    "Education (Aggregate levels): Basic and below basic",
    "Education (Aggregate levels): Intermediate",
    "Education (Aggregate levels): Advanced",
    "Education (Aggregate levels): Level not statedz",
]

# 1. Overwrite cat_cols using preferred order
cat_cols = [c for c in preferred_order if c in wide.columns]

# 2. Safety: append any unexpected categories to the end
extra_cols = [
    c for c in wide.columns
    if c not in (
        ["time", "total", "calculated_total", "difference_total_vs_calculated"]
        + cat_cols
        + [f"{x}_share_percent" for x in cat_cols]
    )
    and not c.endswith("_share_percent")
]

cat_cols = cat_cols + extra_cols

# 3. Rebuild share columns IN THE SAME ORDER
share_cols = [f"{c}_share_percent" for c in cat_cols]

# 4. Final column order
final_cols = (
    ["time"]
    + cat_cols
    + share_cols
    + ["total", "calculated_total", "difference_total_vs_calculated"]
)

# Keep only existing columns (safety)
final_cols = [c for c in final_cols if c in wide.columns]

# Apply order
wide = wide[final_cols].sort_values("time")

# ----------------------------
# Save
# ----------------------------
wide.to_csv(output_path, index=False)
print("Saved:", output_path)
