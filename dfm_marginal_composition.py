# dfm_marginal_composition_spline_DAMPED.py
# Fixes "shares go to 0 or 1" by damping the spline trend effect in the forecast horizon.
#
# Install:
#   pip install pandas numpy matplotlib statsmodels patsy
#
# Put this script next to masterquarterly.csv (or edit CSV_PATH).

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
from patsy import dmatrix, build_design_matrices

# ----------------------------
# USER SETTINGS
# ----------------------------
CSV_PATH = "masterquarterly.csv"
FORECAST_END = "2050Q4"
SNAPSHOT_YEARS = [2026, 2030, 2035, 2050]

# DFM
K_FACTORS = 1          # IMPORTANT: start with 1 for stability
FACTOR_ORDER = 1
ERROR_ORDER = 1
MAXITER = 600

# Spline (keep small to avoid runaway)
DF_SPLINE = 4          # try 4 or 5 (6+ can overfit and explode)
USE_SEASONAL_DUMMIES = False  # set True if you really need seasonal quarter effects

# Damping of trend effect (quarters). Larger = less damping.
# Half-life of 40 quarters = trend effect halves every 10 years
DAMP_HALF_LIFE_Q = 40

# Optional safety clip on transformed forecasts:
# 6 corresponds to extreme odds (~0.002 to 0.998 in logistic); 4 is milder (~0.018 to 0.982)
USE_Z_CLIP = True
Z_CLIP = 6.0

# Reference categories (if missing, script picks largest-share category)
AGE_REF_PREF = "Aggregate bands: 25-54"
EDU_REF_PREF = "Aggregate levels: Intermediate"
POB_REF_PREF = "Place of birth: Native-born"

DROP_EDU = {"Aggregate levels: Less than basic"}
DROP_POB = {"Place of birth: Status unknown"}

EPS = 1e-8
MIN_OBS = 40
MIN_STD = 1e-8

OUT_DIR = Path(__file__).resolve().parent / "outputs_dfm_damped"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# HELPERS
# ----------------------------
def clip01(x, eps=EPS):
    return np.clip(x, eps, 1 - eps)

def logit(p):
    p = clip01(p)
    return np.log(p / (1 - p))

def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))

def to_shares(mat: pd.DataFrame) -> pd.DataFrame:
    den = mat.sum(axis=1)
    return mat.div(den, axis=0)

def pick_reference(shares: pd.DataFrame, preferred: str | None) -> str:
    if preferred is not None and preferred in shares.columns:
        return preferred
    return shares.mean(skipna=True).idxmax()

def log_ratios(shares: pd.DataFrame, ref_col: str) -> pd.DataFrame:
    s = shares.clip(EPS, 1 - EPS)
    other = [c for c in s.columns if c != ref_col]
    return np.log(s[other].div(s[ref_col], axis=0))

def softmax_from_logratios(Z_other: pd.DataFrame, ref_name: str) -> pd.DataFrame:
    expz = np.exp(Z_other)
    denom = 1.0 + expz.sum(axis=1)
    s_ref = 1.0 / denom
    out = expz.mul(s_ref, axis=0)
    out[ref_name] = s_ref
    return out[[*Z_other.columns, ref_name]]

def plot_partition(hist: pd.DataFrame, fc: pd.DataFrame, title: str, filename: str, cutoff: pd.Period):
    fig, ax = plt.subplots(figsize=(11, 5))
    x_hist = hist.index.to_timestamp()
    x_fc = fc.index.to_timestamp()

    for col in hist.columns:
        ax.plot(x_hist, hist[col], label=col)
        ax.plot(x_fc, fc[col], linestyle="--")

    ax.axvline(cutoff.to_timestamp(), linewidth=1)
    ax.set_title(title)
    ax.set_ylabel("Share")
    ax.set_ylim(0, 1)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=220)
    plt.show()
    plt.close(fig)

def snapshot_table(shares_all: pd.DataFrame, years: list[int], name: str) -> pd.DataFrame:
    rows = []
    for y in years:
        p = pd.Period(f"{y}Q4", freq="Q-DEC")
        if p in shares_all.index:
            for cat, val in shares_all.loc[p].items():
                rows.append({"Partition": name, "Year": y, "Category": cat,
                             "Share": float(val), "Share_pct": 100*float(val)})
    return pd.DataFrame(rows)

def make_exog_spline(index: pd.PeriodIndex, t_vals: np.ndarray, df_spline: int,
                     design_info=None, lower=0.0, upper=1.0):
    """
    exog = const + bspline(t) + optional quarter dummies
    t must lie in [lower, upper] for BOTH in-sample and future.
    """
    df_spline = int(df_spline)
    if df_spline < 4:
        raise ValueError(f"DF_SPLINE must be >= 4, got {df_spline}.")

    formula = (
        f"0 + bs(t, df={df_spline}, degree=3, include_intercept=False, "
        f"lower_bound={lower}, upper_bound={upper})"
    )

    if design_info is None:
        spline = dmatrix(formula, {"t": t_vals}, return_type="dataframe")
        spline_info = spline.design_info
        spline.index = index
    else:
        mats = build_design_matrices([design_info], {"t": t_vals})
        spline = pd.DataFrame(np.asarray(mats[0]), index=index, columns=design_info.column_names)
        spline_info = design_info

    X = pd.DataFrame({"const": 1.0}, index=index)
    X = pd.concat([X, spline], axis=1)

    if USE_SEASONAL_DUMMIES:
        q = pd.Series(index=index, data=index.quarter)
        season = pd.get_dummies(q, prefix="Q", drop_first=True)
        season = season.reindex(columns=["Q_2", "Q_3", "Q_4"], fill_value=0)
        season.index = index
        X = pd.concat([X, season], axis=1)

    return X.astype(float), spline_info

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv(CSV_PATH)
df["period"] = pd.PeriodIndex(df["Quarter"].astype(str), freq="Q-DEC")
df = df.sort_values(["period", "Sex"])

age_cols = [c for c in df.columns if c.startswith("Aggregate bands:")]
edu_cols = [c for c in df.columns if c.startswith("Aggregate levels:") and c not in DROP_EDU]
pob_cols = [c for c in df.columns if c.startswith("Place of birth:") and c not in DROP_POB]

for c in age_cols + edu_cols + pob_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

levels = df.groupby("period")[age_cols + edu_cols + pob_cols].sum(min_count=1)

full_periods = pd.period_range(levels.index.min(), levels.index.max(), freq="Q-DEC")
levels = levels.reindex(full_periods)

# Sex totals from age bands
sex_totals = (
    df.groupby(["period", "Sex"])[age_cols].sum(min_count=1).sum(axis=1).unstack("Sex")
).reindex(full_periods)
male_share = sex_totals["Male"] / sex_totals.sum(axis=1)

# Shares
age_shares = to_shares(levels[age_cols])
edu_shares = to_shares(levels[edu_cols])
pob_shares = to_shares(levels[pob_cols])

AGE_REF = pick_reference(age_shares, AGE_REF_PREF)
EDU_REF = pick_reference(edu_shares, EDU_REF_PREF)

# ----------------------------
# TRANSFORM -> Z (unconstrained)
# ----------------------------
Z = pd.DataFrame(index=full_periods)
Z["sex_logit"] = logit(male_share)
Z = pd.concat([Z, log_ratios(age_shares, AGE_REF)], axis=1)
Z = pd.concat([Z, log_ratios(edu_shares, EDU_REF)], axis=1)

Z = Z.replace([np.inf, -np.inf], np.nan)

if len(pob_cols) == 2:
    foreign_candidates = [c for c in pob_cols if "Foreign-born" in c]
    foreign_col = foreign_candidates[0] if foreign_candidates else pob_cols[0]
    Z["pob_logit"] = logit(pob_shares[foreign_col])
    pob_mode = ("binary", foreign_col)
else:
    POB_REF = POB_REF_PREF if POB_REF_PREF in pob_cols else pick_reference(pob_shares, None)
    Z = pd.concat([Z, log_ratios(pob_shares, POB_REF)], axis=1)
    pob_mode = ("multiclass", POB_REF)

Z = Z.replace([np.inf, -np.inf], np.nan)

# ----------------------------
# FILTER: model only decent series; keep the rest constant
# ----------------------------
good_cols, const_cols, const_values = [], [], {}

for c in Z.columns:
    n_obs = int(Z[c].notna().sum())
    s = float(Z[c].std(skipna=True)) if Z[c].notna().any() else np.nan
    if n_obs < MIN_OBS or (np.isnan(s) or s <= MIN_STD):
        const_cols.append(c)
        const_values[c] = float(Z[c].mean(skipna=True)) if Z[c].notna().any() else 0.0
    else:
        good_cols.append(c)

Z_model = Z[good_cols].copy()

# Standardise
Z_mean = Z_model.mean(skipna=True)
Z_std = Z_model.std(skipna=True).replace(0, 1.0)
Zs = (Z_model - Z_mean) / Z_std

# ----------------------------
# FORECAST HORIZON
# ----------------------------
last_period = Zs.index.max()
target_period = pd.Period(FORECAST_END, freq="Q-DEC")
steps = (target_period - last_period).n
if steps <= 0:
    raise ValueError("FORECAST_END must be after the last observed period.")
future_periods = pd.period_range(start=last_period + 1, periods=steps, freq="Q-DEC")

# ----------------------------
# EXOG TIME: spline basis + DAMPING
# ----------------------------
n = len(Zs)
# time scaled on in-sample to [0,1]
t_in = np.linspace(0.0, 1.0, n)

# for future, we allow it to go slightly beyond 1 so the spline can change,
# but we define upper bound accordingly:
t_f = 1.0 + np.arange(1, steps + 1) / max(1, (n - 1))
upper_bound = float(t_f.max())

X, spline_info = make_exog_spline(Zs.index, t_in, DF_SPLINE, design_info=None,
                                 lower=0.0, upper=upper_bound)
X_f, _ = make_exog_spline(future_periods, t_f, DF_SPLINE, design_info=spline_info,
                          lower=0.0, upper=upper_bound)

# Identify spline columns (everything except const and seasonal)
spline_cols = [c for c in X.columns if "bs(" in c]

# Damping weights over horizon: w(h)=0.5 at half-life
h = np.arange(1, steps + 1)
w = np.exp(-np.log(2) * h / DAMP_HALF_LIFE_Q)
w = pd.Series(w, index=future_periods)

# Apply damping only to spline columns (keeps early movement, prevents runaway by 2050)
X_f.loc[:, spline_cols] = X_f.loc[:, spline_cols].mul(w, axis=0)

# Align columns
X_f = X_f.reindex(columns=X.columns, fill_value=0).astype(float)

# ----------------------------
# FIT DFM
# ----------------------------
model = DynamicFactor(
    Zs,
    k_factors=K_FACTORS,
    factor_order=FACTOR_ORDER,
    error_order=ERROR_ORDER,
    exog=X,
    enforce_stationarity=False  # if you can turn this True without errors, do it
)
res = model.fit(disp=False, maxiter=MAXITER)
print("DFM fitted.")
print(f"Outputs saved in: {OUT_DIR}")

# ----------------------------
# FORECAST in Z-space
# ----------------------------
fc = res.get_forecast(steps=steps, exog=X_f).predicted_mean
fc.index = future_periods

Z_fc_model = fc * Z_std + Z_mean

# Add constant cols back
Z_fc_full = Z_fc_model.copy()
for c in const_cols:
    Z_fc_full[c] = const_values[c]
Z_fc_full = Z_fc_full.reindex(columns=Z.columns)

# Optional safety clip in transformed space
if USE_Z_CLIP:
    Z_fc_full = Z_fc_full.clip(-Z_CLIP, Z_CLIP)

# ----------------------------
# BACK-TRANSFORM -> SHARES
# ----------------------------
male_fc = logistic(Z_fc_full["sex_logit"])
sex_hist = pd.DataFrame({"Male share": male_share, "Female share": 1 - male_share}, index=full_periods)
sex_fc = pd.DataFrame({"Male share": male_fc, "Female share": 1 - male_fc}, index=future_periods)

age_other = [c for c in age_shares.columns if c != AGE_REF]
age_hist = age_shares.copy()
age_fc = softmax_from_logratios(Z_fc_full[age_other], AGE_REF)

edu_other = [c for c in edu_shares.columns if c != EDU_REF]
edu_hist = edu_shares.copy()
edu_fc = softmax_from_logratios(Z_fc_full[edu_other], EDU_REF)

if pob_mode[0] == "binary":
    foreign_col = pob_mode[1]
    foreign_fc = logistic(Z_fc_full["pob_logit"])
    other_col = [c for c in pob_cols if c != foreign_col][0]
    pob_hist = pob_shares.copy()
    pob_fc = pd.DataFrame({foreign_col: foreign_fc, other_col: 1 - foreign_fc}, index=future_periods)
else:
    pob_ref = pob_mode[1]
    pob_other = [c for c in pob_shares.columns if c != pob_ref]
    pob_hist = pob_shares.copy()
    pob_fc = softmax_from_logratios(Z_fc_full[pob_other], pob_ref)

# ----------------------------
# PLOTS
# ----------------------------
plot_partition(sex_hist, sex_fc, "Sex composition (historical solid, forecast dashed)",
               "sex_composition_forecast.png", last_period)

plot_partition(age_hist, age_fc, "Age composition (historical solid, forecast dashed)",
               "age_composition_forecast.png", last_period)

plot_partition(edu_hist, edu_fc, "Education composition (historical solid, forecast dashed)",
               "education_composition_forecast.png", last_period)

plot_partition(pob_hist, pob_fc, "Place of birth composition (historical solid, forecast dashed)",
               "place_of_birth_composition_forecast.png", last_period)

# ----------------------------
# SNAPSHOTS (Q4)
# ----------------------------
years = [y for y in SNAPSHOT_YEARS if y <= target_period.year]

sex_all = pd.concat([sex_hist, sex_fc], axis=0)
age_all = pd.concat([age_hist, age_fc], axis=0)
edu_all = pd.concat([edu_hist, edu_fc], axis=0)
pob_all = pd.concat([pob_hist, pob_fc], axis=0)

snap = pd.concat([
    snapshot_table(sex_all, years, "Sex"),
    snapshot_table(age_all, years, "Age"),
    snapshot_table(edu_all, years, "Education"),
    snapshot_table(pob_all, years, "Place of birth"),
], axis=0).sort_values(["Partition", "Year", "Category"]).reset_index(drop=True)

snap_path = OUT_DIR / "forecast_snapshots_q4.csv"
snap.to_csv(snap_path, index=False)

print("\nForecast snapshots (Q4):")
print(snap.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
print(f"\nSaved snapshot CSV: {snap_path}")
