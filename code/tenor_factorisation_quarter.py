import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning


# ============================================================
# Config
# ============================================================
CSV_PATH = "masterquarterly.csv"  # <-- change to your local path

FORECAST_END_YEAR = 2050  # forecast to 2050Q4
TARGET_YEARS = [2026, 2030, 2035, 2050]

RANK = 3
ALS_ITERS = 300
SEED = 42

TIME_COL = "Quarter"
SEX_COL = "Sex"
TOTAL_AGE_COL_CANDIDATE = "Total for age"

AGE_COLS = ["15-24", "25-54", "55-64", "65+"]
EDU_COLS = ["Basic and below basic", "Intermediate", "Advanced", "Level not stated"]
BORN_COLS = [
    "Place of birth: Foreign-born",
    "Place of birth: Native-born",
    "Place of birth: Status unknown",
]

# Quarterly wide outputs
OUT_SEX_WIDE = "sex_wide_quarterly.csv"
OUT_SEX_AGE_WIDE = "sex_age_wide_quarterly.csv"
OUT_SEX_EDU_WIDE = "sex_edu_wide_quarterly.csv"
OUT_SEX_BORN_WIDE = "sex_born_wide_quarterly.csv"

# Annual target (mean over 4 quarters) outputs
OUT_TARGET_SEX = "targets_annual_mean_sex.csv"
OUT_TARGET_SEX_AGE = "targets_annual_mean_sex_age.csv"
OUT_TARGET_SEX_EDU = "targets_annual_mean_sex_edu.csv"
OUT_TARGET_SEX_BORN = "targets_annual_mean_sex_born.csv"


# ============================================================
# Utilities
# ============================================================
def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def norm_col(s: str) -> str:
    s = str(s).lower().strip()
    for ch in [":", "-", "/", "’", "“", "”"]:
        s = s.replace(ch, "")
    s = s.replace(" ", "")
    return s

def find_col(df: pd.DataFrame, target: str) -> str:
    t = norm_col(target)
    for c in df.columns:
        if norm_col(c) == t:
            return c
    for c in df.columns:
        if t in norm_col(c):
            return c
    raise ValueError(f"Cannot find column matching '{target}'. Columns: {df.columns.tolist()}")

def safe_colname(x: str) -> str:
    x = str(x).strip()
    x = x.replace(" ", "_").replace("/", "_").replace("-", "_").replace(":", "")
    while "__" in x:
        x = x.replace("__", "_")
    return x

def parse_quarter_label(q: str) -> pd.Period:
    s = str(q).strip().replace(" ", "")
    return pd.Period(s, freq="Q")

def period_to_label(p: pd.Period) -> str:
    return f"{p.year}Q{p.quarter}"

def forecast_series_quarterly(y, steps: int):
    """
    Quarterly SARIMAX forecast with seasonal period=4.
    Falls back to a linear trend if model fails or does not converge.
    """
    y = np.asarray(y, dtype=float)

    def linear_fc():
        x = np.arange(len(y))
        A = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        x2 = np.arange(len(y), len(y) + steps)
        return coef[0] * x2 + coef[1]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = SARIMAX(
                y,
                order=(1, 1, 0),
                seasonal_order=(1, 0, 0, 4),
                trend="c",
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = model.fit(disp=False)

        converged = True
        if hasattr(res, "mle_retvals") and isinstance(res.mle_retvals, dict):
            converged = bool(res.mle_retvals.get("converged", True))
        if not converged:
            return linear_fc()

        return np.asarray(res.forecast(steps=steps), dtype=float)

    except Exception:
        return linear_fc()

def to_wide(df_long: pd.DataFrame, category_col: str, value_col: str = "Share_overall") -> pd.DataFrame:
    """
    long: Quarter, Sex, category_col, value_col -> wide Quarter + columns Sex_Category
    """
    tmp = df_long.copy()
    tmp["Sex"] = tmp["Sex"].map(safe_colname)
    tmp[category_col] = tmp[category_col].map(safe_colname)
    tmp["col"] = tmp["Sex"] + "_" + tmp[category_col]
    wide = tmp.pivot(index="Quarter", columns="col", values=value_col).reset_index()
    return wide

def plot_wide_trends_quarterly(wide_df: pd.DataFrame, hist_end_period: pd.Period, title: str, ylabel: str = "Overall share"):
    periods = wide_df["Quarter"].map(parse_quarter_label)
    hist_mask = (periods <= hist_end_period).values

    plt.figure(figsize=(12, 6))
    cols = [c for c in wide_df.columns if c != "Quarter"]
    x = np.arange(len(wide_df))

    for c in cols:
        y = wide_df[c].astype(float).values
        plt.plot(x[hist_mask], y[hist_mask], label=c)
        plt.plot(x[~hist_mask], y[~hist_mask], linestyle="--")

    label_to_idx = {q: i for i, q in enumerate(wide_df["Quarter"].tolist())}
    for yr in TARGET_YEARS:
        qlab = f"{yr}Q1"
        if qlab in label_to_idx:
            plt.axvline(label_to_idx[qlab], linestyle=":", linewidth=1)

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Quarter (index)")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.show()

def annual_mean_targets_from_quarterly_wide(wide_df: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """
    Average over 4 quarters within each target year.
    """
    tmp = wide_df.copy()
    tmp["__p__"] = tmp["Quarter"].map(parse_quarter_label)
    tmp["Year"] = tmp["__p__"].apply(lambda p: p.year)
    num_cols = [c for c in tmp.columns if c not in ["Quarter", "__p__", "Year"]]
    out = tmp[tmp["Year"].isin(years)].groupby("Year")[num_cols].mean().reset_index().sort_values("Year")
    return out

def interpolate_on_period_index(df_block: pd.DataFrame, full_index: pd.Index) -> pd.DataFrame:
    """
    Reindex to continuous PeriodIndex, then linearly interpolate across missing quarters.
    """
    out = df_block.reindex(full_index)
    out = out.interpolate(method="linear", limit_direction="both")
    out = out.ffill().bfill()
    return out


# ============================================================
# Coupled CP-ALS
# ============================================================
def khatri_rao(A, B):
    I, R = A.shape
    J, R2 = B.shape
    assert R == R2
    return np.einsum("ir,jr->ijr", A, B).reshape(I * J, R)

def unfold_mode0(X):
    T, S, K = X.shape
    return X.reshape(T, S * K)

def unfold_mode1(X):
    T, S, K = X.shape
    return np.transpose(X, (1, 0, 2)).reshape(S, T * K)

def unfold_mode2(X):
    T, S, K = X.shape
    return np.transpose(X, (2, 0, 1)).reshape(K, T * S)

def solve_ls(Xmat, KR):
    G = KR.T @ KR + 1e-8 * np.eye(KR.shape[1])
    return (Xmat @ KR) @ np.linalg.inv(G)

def coupled_cp_als(X_list, rank=3, n_iter=200, seed=0, nonneg=True):
    rng = np.random.default_rng(seed)
    T, S, _ = X_list[0].shape
    R = rank

    U_time = rng.random((T, R)) + 1e-3
    U_sex = rng.random((S, R)) + 1e-3
    U_modes = [rng.random((X.shape[2], R)) + 1e-3 for X in X_list]

    for _ in range(n_iter):
        # mode-specific
        for k, Xk in enumerate(X_list):
            X2 = unfold_mode2(Xk)             # (K, T*S)
            KR = khatri_rao(U_sex, U_time)    # (S*T, R)
            Uk = solve_ls(X2, KR)
            if nonneg:
                Uk = np.clip(Uk, 1e-10, None)
            U_modes[k] = Uk

        # shared time
        num = np.zeros((T, R))
        den = np.zeros((R, R))
        for k, Xk in enumerate(X_list):
            X0 = unfold_mode0(Xk)                 # (T, S*K)
            KR = khatri_rao(U_modes[k], U_sex)    # (K*S, R)
            num += X0 @ KR
            den += KR.T @ KR
        U_time = num @ np.linalg.inv(den + 1e-8 * np.eye(R))
        if nonneg:
            U_time = np.clip(U_time, 1e-10, None)

        # shared sex
        num = np.zeros((S, R))
        den = np.zeros((R, R))
        for k, Xk in enumerate(X_list):
            X1 = unfold_mode1(Xk)                 # (S, T*K)
            KR = khatri_rao(U_modes[k], U_time)   # (K*T, R)
            num += X1 @ KR
            den += KR.T @ KR
        U_sex = num @ np.linalg.inv(den + 1e-8 * np.eye(R))
        if nonneg:
            U_sex = np.clip(U_sex, 1e-10, None)

    return U_time, U_sex, U_modes

def reconstruct_tensor(Ut, Us, Uc):
    return np.einsum("tr,sr,kr->tsk", Ut, Us, Uc)

def renormalise_conditional(Xhat):
    Xhat = np.clip(Xhat, 0.0, None)
    denom = Xhat.sum(axis=2, keepdims=True)
    denom = np.where(denom <= 0, 1.0, denom)
    return Xhat / denom


# ============================================================
# 1) Load + continuous historical quarter axis
# ============================================================
df = clean_cols(pd.read_csv(CSV_PATH))

TOTAL_AGE_COL = find_col(df, TOTAL_AGE_COL_CANDIDATE)

# Keep only existing columns
AGE_COLS = [c for c in AGE_COLS if c in df.columns]
EDU_COLS = [c for c in EDU_COLS if c in df.columns]
BORN_COLS = [c for c in BORN_COLS if c in df.columns]

df[TIME_COL] = df[TIME_COL].astype(str).str.strip()
df["__period__"] = df[TIME_COL].map(parse_quarter_label)

for c in [TOTAL_AGE_COL] + AGE_COLS + EDU_COLS + BORN_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df[SEX_COL] = df[SEX_COL].astype(str).str.strip()
sex_levels = sorted(df[SEX_COL].unique().tolist())
if len(sex_levels) < 2:
    raise ValueError("Sex has <2 unique values; cannot proceed.")

period_obs = sorted(df["__period__"].dropna().unique().tolist())
if not period_obs:
    raise ValueError("No valid quarters parsed from 'Quarter' column.")

hist_start = period_obs[0]
hist_end = period_obs[-1]
hist_end_period = hist_end

# continuous historical quarters
period_hist_full = list(pd.period_range(hist_start, hist_end, freq="Q"))

# future quarters up to 2050Q4
end_period = pd.Period(f"{FORECAST_END_YEAR}Q4", freq="Q")
period_future = list(pd.period_range(hist_end + 1, end_period, freq="Q")) if hist_end < end_period else []

period_all = period_hist_full + period_future
quarter_all = [period_to_label(p) for p in period_all]


# ============================================================
# 2) Sex shares (overall) with interpolation (NOT zero-fill)
# ============================================================
tot_sex_obs = df.pivot_table(index="__period__", columns=SEX_COL, values=TOTAL_AGE_COL, aggfunc="sum")

tot_sex_hist = tot_sex_obs.reindex(period_hist_full)
tot_sex_hist = tot_sex_hist.interpolate(method="linear", limit_direction="both").ffill().bfill()
tot_sex_hist = tot_sex_hist.fillna(0.0)  # last resort

tot_total_hist = tot_sex_hist.sum(axis=1).replace(0.0, np.nan)
sex_share_hist = tot_sex_hist.div(tot_total_hist, axis=0).fillna(0.0)

male_label = next((s for s in sex_levels if str(s).lower().startswith("male")), sex_levels[0])


# ============================================================
# 3) Conditional shares within sex + interpolation + renormalise
# ============================================================
def build_conditional_share_long(cols):
    block = df[["__period__", SEX_COL, TOTAL_AGE_COL] + cols].copy()
    denom = block[TOTAL_AGE_COL].replace(0.0, np.nan)
    for c in cols:
        block[c] = block[c] / denom
    return block[["__period__", SEX_COL] + cols]

age_share_long = build_conditional_share_long(AGE_COLS)
edu_share_long = build_conditional_share_long(EDU_COLS)
born_share_long = build_conditional_share_long(BORN_COLS)

def to_tensor_with_interpolation(share_long_df, cat_cols, periods, sexes):
    """
    Build (T,S,K) tensor. For each sex:
      - reindex to continuous periods
      - interpolate missing quarters
      - ffill/bfill
      - renormalise across categories so sum=1 each quarter (conditional share)
    """
    full_index = pd.Index(periods, name="__period__")
    T = len(periods)
    S = len(sexes)
    K = len(cat_cols)

    X = np.zeros((T, S, K), dtype=float)

    for sj, s in enumerate(sexes):
        sub = share_long_df[share_long_df[SEX_COL] == s].set_index("__period__")[cat_cols]
        sub = interpolate_on_period_index(sub, full_index)
        sub = sub.fillna(0.0)

        vals = sub.values.astype(float)
        vals = np.clip(vals, 0.0, None)
        denom = vals.sum(axis=1, keepdims=True)
        denom = np.where(denom <= 0, 1.0, denom)
        vals = vals / denom

        X[:, sj, :] = vals

    return X

X_age = to_tensor_with_interpolation(age_share_long, AGE_COLS, period_hist_full, sex_levels)
X_edu = to_tensor_with_interpolation(edu_share_long, EDU_COLS, period_hist_full, sex_levels)
X_born = to_tensor_with_interpolation(born_share_long, BORN_COLS, period_hist_full, sex_levels)


# ============================================================
# 4) Fit coupled CP on historical tensors
# ============================================================
U_time_hist, U_sex, (U_age, U_edu, U_born) = coupled_cp_als(
    [X_age, X_edu, X_born],
    rank=RANK,
    n_iter=ALS_ITERS,
    seed=SEED,
    nonneg=True
)


# ============================================================
# 5) Forecast time factors and sex shares (quarterly)
# ============================================================
steps = len(period_future)

U_time_future = np.zeros((steps, RANK))
for r in range(RANK):
    U_time_future[:, r] = forecast_series_quarterly(U_time_hist[:, r], steps)

U_time_all = np.vstack([U_time_hist, U_time_future])

male_hist = sex_share_hist[male_label].reindex(period_hist_full).ffill().bfill().values
male_future = forecast_series_quarterly(male_hist, steps)
male_all = np.clip(np.concatenate([male_hist, male_future]), 1e-6, 1 - 1e-6)

sex_share_pred = pd.DataFrame(index=period_all, columns=sex_levels, dtype=float)
if len(sex_levels) == 2:
    other = [s for s in sex_levels if s != male_label][0]
    for p, pm in zip(period_all, male_all):
        sex_share_pred.loc[p, male_label] = pm
        sex_share_pred.loc[p, other] = 1.0 - pm
else:
    other_levels = [s for s in sex_levels if s != male_label]
    avg_other = sex_share_hist[other_levels].mean(axis=0)
    avg_other = avg_other / avg_other.sum()
    for p, pm in zip(period_all, male_all):
        sex_share_pred.loc[p, male_label] = pm
        rest = (1.0 - pm)
        for s in other_levels:
            sex_share_pred.loc[p, s] = rest * float(avg_other[s])


# ============================================================
# 6) Reconstruct conditional shares for all quarters and build overall shares
# ============================================================
Xhat_age = renormalise_conditional(reconstruct_tensor(U_time_all, U_sex, U_age))
Xhat_edu = renormalise_conditional(reconstruct_tensor(U_time_all, U_sex, U_edu))
Xhat_born = renormalise_conditional(reconstruct_tensor(U_time_all, U_sex, U_born))

sex_to_j = {s: j for j, s in enumerate(sex_levels)}
p_to_i_all = {p: i for i, p in enumerate(period_all)}

def make_long_overall(periods, cat_names, X_cond_hat, category_col_name: str):
    rows = []
    for p in periods:
        i = p_to_i_all[p]
        qlab = period_to_label(p)
        for s in sex_levels:
            j = sex_to_j[s]
            p_sex = float(sex_share_pred.loc[p, s])
            for k, cat in enumerate(cat_names):
                p_cond = float(X_cond_hat[i, j, k])
                rows.append({
                    "Quarter": qlab,
                    "Sex": s,
                    category_col_name: cat,
                    "Share_overall": p_sex * p_cond
                })
    return pd.DataFrame(rows)

# Sex wide (overall share)
sex_wide = sex_share_pred.copy()
sex_wide.index = [period_to_label(p) for p in sex_wide.index]
sex_wide = sex_wide.reset_index().rename(columns={"index": "Quarter"})
sex_wide = sex_wide.rename(columns={c: safe_colname(c) for c in sex_levels})

age_long = make_long_overall(period_all, AGE_COLS, Xhat_age, "Age")
edu_long = make_long_overall(period_all, EDU_COLS, Xhat_edu, "Education")
born_long = make_long_overall(period_all, BORN_COLS, Xhat_born, "Birthplace")

sex_age_wide = to_wide(age_long, category_col="Age")
sex_edu_wide = to_wide(edu_long, category_col="Education")
sex_born_wide = to_wide(born_long, category_col="Birthplace")

# Ensure correct quarter order
order = pd.DataFrame({"Quarter": [period_to_label(p) for p in period_all]})
sex_wide = order.merge(sex_wide, on="Quarter", how="left")
sex_age_wide = order.merge(sex_age_wide, on="Quarter", how="left")
sex_edu_wide = order.merge(sex_edu_wide, on="Quarter", how="left")
sex_born_wide = order.merge(sex_born_wide, on="Quarter", how="left")

# Save quarterly wide
sex_wide.to_csv(OUT_SEX_WIDE, index=False)
sex_age_wide.to_csv(OUT_SEX_AGE_WIDE, index=False)
sex_edu_wide.to_csv(OUT_SEX_EDU_WIDE, index=False)
sex_born_wide.to_csv(OUT_SEX_BORN_WIDE, index=False)

print("Saved quarterly wide tables:")
print(" -", OUT_SEX_WIDE)
print(" -", OUT_SEX_AGE_WIDE)
print(" -", OUT_SEX_EDU_WIDE)
print(" -", OUT_SEX_BORN_WIDE)


# ============================================================
# 7) Annual target tables (mean over 4 quarters)
# ============================================================
target_years_in_range = [y for y in TARGET_YEARS if (period_all[0].year <= y <= FORECAST_END_YEAR)]

targets_sex = annual_mean_targets_from_quarterly_wide(sex_wide, target_years_in_range)
targets_sex_age = annual_mean_targets_from_quarterly_wide(sex_age_wide, target_years_in_range)
targets_sex_edu = annual_mean_targets_from_quarterly_wide(sex_edu_wide, target_years_in_range)
targets_sex_born = annual_mean_targets_from_quarterly_wide(sex_born_wide, target_years_in_range)

targets_sex.to_csv(OUT_TARGET_SEX, index=False)
targets_sex_age.to_csv(OUT_TARGET_SEX_AGE, index=False)
targets_sex_edu.to_csv(OUT_TARGET_SEX_EDU, index=False)
targets_sex_born.to_csv(OUT_TARGET_SEX_BORN, index=False)

print("Saved annual target tables (mean over 4 quarters):")
print(" -", OUT_TARGET_SEX)
print(" -", OUT_TARGET_SEX_AGE)
print(" -", OUT_TARGET_SEX_EDU)
print(" -", OUT_TARGET_SEX_BORN)


# ============================================================
# 8) Plot
# ============================================================
plot_wide_trends_quarterly(sex_wide, hist_end_period, "Sex composition (quarterly, historical & projected)")
plot_wide_trends_quarterly(sex_age_wide, hist_end_period, "Sex × Age composition (quarterly, historical & projected)")
plot_wide_trends_quarterly(sex_edu_wide, hist_end_period, "Sex × Education composition (quarterly, historical & projected)")
plot_wide_trends_quarterly(sex_born_wide, hist_end_period, "Sex × Place of birth composition (quarterly, historical & projected)")
