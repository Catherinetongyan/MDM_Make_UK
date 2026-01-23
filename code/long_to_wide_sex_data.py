import pandas as pd
import numpy as np

# ----------------------------
# Load data
# ----------------------------
input_path = "GBR_Q-filtered-sex-2026-01-20.csv"
df = pd.read_csv(input_path)

# ----------------------------
# Keep only required columns
# ----------------------------
df = df[["sex.label", "time", "obs_value"]].copy()

# ----------------------------
# Keep time from 1995 onwards
# Assumes time like '1995Q1'
# ----------------------------
df["year"] = df["time"].astype(str).str.slice(0, 4).astype(int)
df = df[df["year"] >= 1995].copy()
df.drop(columns=["year"], inplace=True)

# ----------------------------
# Pivot: sex.label -> columns
# ----------------------------
wide = (
    df.pivot_table(
        index="time",
        columns="sex.label",
        values="obs_value",
        aggfunc="sum"
    )
    .reset_index()
)

# Normalize possible case variants (optional safety)
wide.columns = [c.strip().lower() if isinstance(c, str) else c for c in wide.columns]

# After lowercasing, expected columns are: 'time', 'male', 'female', 'total'
# Rename to target output names
rename_map = {
    "male": "male_thousands",
    "female": "female_thousands",
    "total": "total"
}
wide = wide.rename(columns=rename_map)

# ----------------------------
# Calculations
# ----------------------------
wide["calculated_total"] = wide["male_thousands"] + wide["female_thousands"]

# Shares: requested male/total and female/total.
# Use official 'total' when available; otherwise fall back to calculated_total.
denom = wide["total"].where(wide["total"].notna(), wide["calculated_total"])

wide["male_share_percent"] = wide["male_thousands"] / denom * 100
wide["female_share_percent"] = wide["female_thousands"] / denom * 100

wide["difference_total_vs_calculated"] = wide["total"] - wide["calculated_total"]

# ----------------------------
# Reorder columns
# ----------------------------
wide = wide[
    [
        "time",
        "male_thousands",
        "female_thousands",
        "male_share_percent",
        "female_share_percent",
        "total",
        "calculated_total",
        "difference_total_vs_calculated",
    ]
].sort_values("time")

# ----------------------------
# Save output
# ----------------------------
output_path = "`GBR_Q_sex_wide_1995_onwards.csv"
wide.to_csv(output_path, index=False)

print("Saved:", output_path)

