import pandas as pd
import numpy as np
import re

# ----------------------------
# Paths
# ----------------------------
input_path = "GBR_Q-filtered-age-2026-01-20.csv"
output_path = "GBR_Q_age_wide_1995_onwards.csv"

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv(input_path)

# Keep only required columns
df = df[["time", "sex.label", "classif1.label", "obs_value"]].copy()

# ----------------------------
# Keep time from 1995 onwards (expects '1995Q1' format)
# ----------------------------
df["time"] = df["time"].astype(str).str.strip()
df["year"] = df["time"].str.slice(0, 4).astype(int)
df = df[df["year"] >= 1995].copy()
df.drop(columns=["year"], inplace=True)

# Clean labels
df["sex.label"] = df["sex.label"].astype(str).str.strip()
df["classif1.label"] = df["classif1.label"].astype(str).str.strip()

if (df["sex.label"].str.lower() == "total").any():
    df = df[df["sex.label"].str.lower() == "total"].copy()

# ----------------------------
# Pivot: age group (classif1.label) -> columns
# ----------------------------
wide = (
    df.pivot_table(
        index="time",
        columns="classif1.label",
        values="obs_value",
        aggfunc="sum"
    )
    .reset_index()
)

# ----------------------------
# Identify the "Total" age-band column robustly
# e.g. "Total" or "Age (Aggregate bands): Total"
# ----------------------------
def is_total_age_col(col_name: str) -> bool:
    s = str(col_name).strip().lower()
    # Match exact 'total' OR anything ending with ': total' OR containing ' total' as a suffix token
    return (s == "total") or s.endswith(": total") or re.search(r"(^|[:\s])total$", s) is not None

age_group_cols = [c for c in wide.columns if c != "time"]
total_age_cols = [c for c in age_group_cols if is_total_age_col(c)]

# If multiple candidates exist, choose the one with the most non-missing values
total_col = None
if len(total_age_cols) == 1:
    total_col = total_age_cols[0]
elif len(total_age_cols) > 1:
    non_na_counts = {c: wide[c].notna().sum() for c in total_age_cols}
    total_col = max(non_na_counts, key=non_na_counts.get)

# Create official total series
if total_col is not None:
    wide["total"] = wide[total_col]

# ----------------------------
# Define age columns EXCLUDING the total age-band column
# ----------------------------
age_cols = [c for c in age_group_cols if c != total_col]

# ----------------------------
# Calculate totals and shares
# ----------------------------
wide["calculated_total"] = wide[age_cols].sum(axis=1, skipna=True)

# Use official total as denominator when available; otherwise fall back to calculated_total
denom = wide["total"].where(wide["total"].notna(), wide["calculated_total"])

# Shares for each age group (DO NOT create share for the total column)
for c in age_cols:
    wide[f"{c}_share_percent"] = wide[c] / denom * 100

# Difference must have values when total exists
wide["difference_total_vs_calculated"] = wide["total"] - wide["calculated_total"]

# ----------------------------
# Reorder columns: time, age values..., age shares..., total, calculated_total, difference
# ----------------------------
share_cols = [f"{c}_share_percent" for c in age_cols]
final_cols = ["time"] + age_cols + share_cols + ["total", "calculated_total", "difference_total_vs_calculated"]

# Keep only columns that actually exist (safety)
final_cols = [c for c in final_cols if c in wide.columns]
wide = wide[final_cols].sort_values("time")

# ----------------------------
# Save
# ----------------------------
wide.to_csv(output_path, index=False)
print("Saved:", output_path)

