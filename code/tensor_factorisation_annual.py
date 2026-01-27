import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning


# ============================================================
# Config
# ============================================================
CSV_PATH = "master annual data.csv"  # <-- 改成你本地路径
FORECAST_END_YEAR = 2050
MARK_YEARS = [2026, 2030, 2035, 2050]
RANK = 3
ALS_ITERS = 300
SEED = 42

YEAR_COL = "Year"
SEX_COL = "Sex"
TOTAL_AGE_COL = "total for age"

AGE_COLS = ["15-24", "25-54", "55-64", "65+"]
EDU_COLS = ["Basic and below basic", "Intermediate", "Advanced", "Level not stated"]
BORN_COLS = [
    "Place of birth: Foreign-born",
    "Place of birth: Native-born",
    "Place of birth: Status unknown",
]

# Output files
OUT_SEX_WIDE = "sex_wide.csv"
OUT_SEX_AGE_WIDE = "sex_age_wide.csv"
OUT_SEX_EDU_WIDE = "sex_edu_wide.csv"
OUT_SEX_BORN_WIDE = "sex_born_wide.csv"

# ============================================================
# Utilities
# ============================================================
def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def safe_colname(x: str) -> str:
    x = str(x).strip()
    x = x.replace(" ", "_").replace("/", "_").replace("-", "_").replace(":", "")
    while "__" in x:
        x = x.replace("__", "_")
    return x

def forecast_series_arima(y, steps: int):
    """
    Forecast 1D series with SARIMAX(1,1,0) + trend; fallback to linear if non-converged / error.
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
                y, order=(1, 1, 0), trend="c",
                enforce_stationarity=False, enforce_invertibility=False
            )
            res = model.fit(disp=False)

        converged = True
        if hasattr(res, "mle_retvals") and isinstance(res.mle_retvals, dict):
            converged = bool(res.mle_retvals.get("converged", True))
        if not converged:
            return linear_fc()

        fc = res.forecast(steps=steps)
        return np.asarray(fc, dtype=float)
    except Exception:
        return linear_fc()

def to_wide(df_long: pd.DataFrame, category_col: str, value_col: str = "Share_overall") -> pd.DataFrame:
    """
    long: Year, Sex, category_col, value_col -> wide Year + columns Sex_Category
    """
    tmp = df_long.copy()
    tmp["Sex"] = tmp["Sex"].map(safe_colname)
    tmp[category_col] = tmp[category_col].map(safe_colname)
    tmp["col"] = tmp["Sex"] + "_" + tmp[category_col]
    wide = (
        tmp.pivot(index="Year", columns="col", values=value_col)
           .reset_index()
           .sort_values("Year")
    )
    return wide

def plot_wide_trends(wide_df: pd.DataFrame, hist_end_year: int, title: str, ylabel: str = "Share"):
    years = wide_df["Year"].values
    cols = [c for c in wide_df.columns if c != "Year"]

    plt.figure(figsize=(11, 6))
    hist_mask = years <= hist_end_year

    for c in cols:
        y = wide_df[c].values.astype(float)
        plt.plot(years[hist_mask], y[hist_mask], label=c)
        plt.plot(years[~hist_mask], y[~hist_mask], linestyle="--")

    for yv in MARK_YEARS:
        plt.axvline(yv, linestyle=":", linewidth=1)

    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel(ylabel)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.show()


# ============================================================
# Coupled CP-ALS (3 tensors share time + sex factors)
# ============================================================
def khatri_rao(A, B):
    # A: (I,R), B: (J,R) -> (I*J,R)
    I, R = A.shape
    J, R2 = B.shape
    assert R == R2
    return np.einsum("ir,jr->ijr", A, B).reshape(I * J, R)

def unfold_mode0(X):
    # (T,S,K) -> (T, S*K)
    T, S, K = X.shape
    return X.reshape(T, S * K)

def unfold_mode1(X):
    # (T,S,K) -> (S, T*K)
    T, S, K = X.shape
    return np.transpose(X, (1, 0, 2)).reshape(S, T * K)

def unfold_mode2(X):
    # (T,S,K) -> (K, T*S)
    T, S, K = X.shape
    return np.transpose(X, (2, 0, 1)).reshape(K, T * S)

def solve_ls(Xmat, KR):
    # Xmat: (n, m), KR: (m, R) => U: (n, R)
    G = KR.T @ KR
    G = G + 1e-8 * np.eye(G.shape[0])
    return (Xmat @ KR) @ np.linalg.inv(G)

def coupled_cp_als(X_list, rank=3, n_iter=200, seed=0, nonneg=True):
    rng = np.random.default_rng(seed)
    T, S, _ = X_list[0].shape
    R = rank

    U_time = rng.random((T, R)) + 1e-3
    U_sex = rng.random((S, R)) + 1e-3
    U_modes = [rng.random((X.shape[2], R)) + 1e-3 for X in X_list]

    for _ in range(n_iter):
        # Update each mode-specific factor
        for k, Xk in enumerate(X_list):
            X2 = unfold_mode2(Xk)             # (K, T*S)
            KR = khatri_rao(U_sex, U_time)    # (S*T, R) corresponds to (T*S)
            Uk = solve_ls(X2, KR)
            if nonneg:
                Uk = np.clip(Uk, 1e-10, None)
            U_modes[k] = Uk

        # Update shared U_time
        num = np.zeros((T, R))
        den = np.zeros((R, R))
        for k, Xk in enumerate(X_list):
            X0 = unfold_mode0(Xk)                 # (T, S*K)
            KR = khatri_rao(U_modes[k], U_sex)    # (K*S, R) matches S*K
            num += X0 @ KR
            den += KR.T @ KR
        den = den + 1e-8 * np.eye(R)
        U_time = num @ np.linalg.inv(den)
        if nonneg:
            U_time = np.clip(U_time, 1e-10, None)

        # Update shared U_sex
        num = np.zeros((S, R))
        den = np.zeros((R, R))
        for k, Xk in enumerate(X_list):
            X1 = unfold_mode1(Xk)                 # (S, T*K)
            KR = khatri_rao(U_modes[k], U_time)   # (K*T, R) matches T*K
            num += X1 @ KR
            den += KR.T @ KR
        den = den + 1e-8 * np.eye(R)
        U_sex = num @ np.linalg.inv(den)
        if nonneg:
            U_sex = np.clip(U_sex, 1e-10, None)

    return U_time, U_sex, U_modes

def reconstruct_tensor(Ut, Us, Uc):
    # (T,R)(S,R)(K,R) -> (T,S,K)
    return np.einsum("tr,sr,kr->tsk", Ut, Us, Uc)

def renormalise_conditional(Xhat):
    Xhat = np.clip(Xhat, 0.0, None)
    denom = Xhat.sum(axis=2, keepdims=True)
    denom = np.where(denom <= 0, 1.0, denom)
    return Xhat / denom


# ============================================================
# 1) Load + shares
# ============================================================
df = clean_cols(pd.read_csv(CSV_PATH))

# Keep only columns that exist
AGE_COLS = [c for c in AGE_COLS if c in df.columns]
EDU_COLS = [c for c in EDU_COLS if c in df.columns]
BORN_COLS = [c for c in BORN_COLS if c in df.columns]

# Make numeric
for c in [TOTAL_AGE_COL] + AGE_COLS + EDU_COLS + BORN_COLS:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

df[SEX_COL] = df[SEX_COL].astype(str).str.strip()

year_hist = sorted(df[YEAR_COL].dropna().astype(int).unique().tolist())
hist_end_year = max(year_hist)
year_all = list(range(min(year_hist), FORECAST_END_YEAR + 1))
year_future = [y for y in year_all if y > hist_end_year]

sex_levels = sorted(df[SEX_COL].unique().tolist())
if len(sex_levels) < 2:
    raise ValueError("Sex column appears to have <2 unique values; cannot proceed.")

# Totals by year×sex
tot_sex = df.pivot_table(index=YEAR_COL, columns=SEX_COL, values=TOTAL_AGE_COL, aggfunc="sum")
tot_sex = tot_sex.reindex(year_hist).copy()
tot_year = tot_sex.sum(axis=1)

# Sex share over total manufacturing that year
sex_share_hist = tot_sex.div(tot_year, axis=0)

# Choose a "male" label if present; else fallback to first
male_label = None
for s in sex_levels:
    if str(s).lower().startswith("male"):
        male_label = s
        break
if male_label is None:
    male_label = sex_levels[0]

# Build conditional shares within sex using total_for_age as denominator
def build_conditional_share(cols):
    block = df[[YEAR_COL, SEX_COL, TOTAL_AGE_COL] + cols].copy()
    # conditional share per row
    for c in cols:
        block[c] = block[c] / block[TOTAL_AGE_COL]
    return block[[YEAR_COL, SEX_COL] + cols]

age_share_long = build_conditional_share(AGE_COLS)
edu_share_long = build_conditional_share(EDU_COLS)
born_share_long = build_conditional_share(BORN_COLS)

def to_tensor(share_long_df, cat_cols, years, sexes):
    T = len(years)
    S = len(sexes)
    K = len(cat_cols)
    year_to_i = {y: i for i, y in enumerate(years)}
    sex_to_j = {s: j for j, s in enumerate(sexes)}
    X = np.full((T, S, K), np.nan, dtype=float)

    for _, row in share_long_df.iterrows():
        y = int(row[YEAR_COL])
        s = row[SEX_COL]
        if y not in year_to_i or s not in sex_to_j:
            continue
        i = year_to_i[y]
        j = sex_to_j[s]
        X[i, j, :] = np.array([row[c] for c in cat_cols], dtype=float)

    return np.nan_to_num(X, nan=0.0)

X_age = to_tensor(age_share_long, AGE_COLS, year_hist, sex_levels)    # (T,S,A)
X_edu = to_tensor(edu_share_long, EDU_COLS, year_hist, sex_levels)    # (T,S,E)
X_born = to_tensor(born_share_long, BORN_COLS, year_hist, sex_levels) # (T,S,B)


# ============================================================
# 2) Fit coupled CP on historical conditional shares
# ============================================================
U_time_hist, U_sex, U_modes = coupled_cp_als(
    [X_age, X_edu, X_born],
    rank=RANK,
    n_iter=ALS_ITERS,
    seed=SEED,
    nonneg=True
)
U_age, U_edu, U_born = U_modes

# ============================================================
# 3) Forecast latent time factors to 2050
# ============================================================
steps = len(year_future)
U_time_future = np.zeros((steps, RANK))
for r in range(RANK):
    U_time_future[:, r] = forecast_series_arima(U_time_hist[:, r], steps)

U_time_all = np.vstack([U_time_hist, U_time_future])  # aligned with year_all

# ============================================================
# 4) Forecast sex shares (male share); re-normalise if >2 sexes
# ============================================================
male_hist = sex_share_hist[male_label].reindex(year_hist).ffill().bfill().values
male_future = forecast_series_arima(male_hist, steps)
male_all = np.clip(np.concatenate([male_hist, male_future]), 1e-6, 1 - 1e-6)

sex_share_pred = pd.DataFrame(index=year_all, columns=sex_levels, dtype=float)

if len(sex_levels) == 2:
    other = [s for s in sex_levels if s != male_label][0]
    for y, pm in zip(year_all, male_all):
        sex_share_pred.loc[y, male_label] = pm
        sex_share_pred.loc[y, other] = 1.0 - pm
else:
    # If more than 2 sex labels exist, keep historical average proportions for non-male
    # and scale them to (1 - male_share)
    other_levels = [s for s in sex_levels if s != male_label]
    avg_other = sex_share_hist[other_levels].mean(axis=0)
    avg_other = avg_other / avg_other.sum()
    for y, pm in zip(year_all, male_all):
        sex_share_pred.loc[y, male_label] = pm
        rest = (1.0 - pm)
        for s in other_levels:
            sex_share_pred.loc[y, s] = rest * float(avg_other[s])

# ============================================================
# 5) Reconstruct conditional shares for age/edu/born (all years)
# ============================================================
Xhat_age = renormalise_conditional(reconstruct_tensor(U_time_all, U_sex, U_age))
Xhat_edu = renormalise_conditional(reconstruct_tensor(U_time_all, U_sex, U_edu))
Xhat_born = renormalise_conditional(reconstruct_tensor(U_time_all, U_sex, U_born))

# ============================================================
# 6) Build outputs: overall shares for sex×age / sex×edu / sex×born
# ============================================================
sex_to_j = {s: j for j, s in enumerate(sex_levels)}
year_to_i_all = {y: i for i, y in enumerate(year_all)}

def make_long_overall(years, kind, cat_names, X_cond_hat):
    rows = []
    for y in years:
        i = year_to_i_all[y]
        for s in sex_levels:
            j = sex_to_j[s]
            p_sex = float(sex_share_pred.loc[y, s])
            for k, cat in enumerate(cat_names):
                p_cond = float(X_cond_hat[i, j, k])      # p(cat | sex)
                rows.append({
                    "Year": y,
                    "Sex": s,
                    "Kind": kind,
                    "Category": cat,
                    "Share_conditional": p_cond,
                    "Share_overall": p_sex * p_cond       # p(sex, cat)
                })
    return pd.DataFrame(rows)

# Sex-only long (overall share)
sex_long = (
    sex_share_pred.reset_index()
    .rename(columns={"index": "Year"})
    .melt(id_vars="Year", var_name="Sex", value_name="Share_overall")
)
sex_long["Kind"] = "sex"
sex_long["Category"] = sex_long["Sex"]
sex_long["Share_conditional"] = sex_long["Share_overall"]

age_long = make_long_overall(year_all, "age", AGE_COLS, Xhat_age)
edu_long = make_long_overall(year_all, "education", EDU_COLS, Xhat_edu)
born_long = make_long_overall(year_all, "birthplace", BORN_COLS, Xhat_born)

# For wide conversion we want explicit column names, so create specific long tables with consistent category columns
sex_age_long = age_long.rename(columns={"Category": "Age"})[["Year", "Sex", "Age", "Share_overall"]]
sex_edu_long = edu_long.rename(columns={"Category": "Education"})[["Year", "Sex", "Education", "Share_overall"]]
sex_born_long = born_long.rename(columns={"Category": "Birthplace"})[["Year", "Sex", "Birthplace", "Share_overall"]]

# Sex wide
sex_wide = (
    sex_long.assign(col=lambda x: x["Sex"].map(safe_colname))
            .pivot(index="Year", columns="col", values="Share_overall")
            .reset_index()
            .sort_values("Year")
)

sex_age_wide = to_wide(sex_age_long, category_col="Age", value_col="Share_overall")
sex_edu_wide = to_wide(sex_edu_long, category_col="Education", value_col="Share_overall")
sex_born_wide = to_wide(sex_born_long, category_col="Birthplace", value_col="Share_overall")

# Save
sex_wide.to_csv(OUT_SEX_WIDE, index=False)
sex_age_wide.to_csv(OUT_SEX_AGE_WIDE, index=False)
sex_edu_wide.to_csv(OUT_SEX_EDU_WIDE, index=False)
sex_born_wide.to_csv(OUT_SEX_BORN_WIDE, index=False)

print("Saved:")
print(f" - {OUT_SEX_WIDE}")
print(f" - {OUT_SEX_AGE_WIDE}")
print(f" - {OUT_SEX_EDU_WIDE}")
print(f" - {OUT_SEX_BORN_WIDE}")

# Also export target years only (optional, handy for report tables)
targets = sorted(set([y for y in MARK_YEARS if y in year_all]))
sex_age_wide[sex_age_wide["Year"].isin(targets)].to_csv("sex_age_targets.csv", index=False)
sex_edu_wide[sex_edu_wide["Year"].isin(targets)].to_csv("sex_edu_targets.csv", index=False)
sex_born_wide[sex_born_wide["Year"].isin(targets)].to_csv("sex_born_targets.csv", index=False)
sex_wide[sex_wide["Year"].isin(targets)].to_csv("sex_targets.csv", index=False)
print("Saved target-year slices: sex_targets.csv, sex_age_targets.csv, sex_edu_targets.csv, sex_born_targets.csv")


# ============================================================
# 7) Plot trends (history solid, forecast dashed)
# ============================================================
plot_wide_trends(sex_wide, hist_end_year, "Sex composition in manufacturing (historical & projected)", ylabel="Overall share")

plot_wide_trends(sex_age_wide, hist_end_year, "Sex × Age composition in manufacturing (historical & projected)", ylabel="Overall share")

plot_wide_trends(sex_edu_wide, hist_end_year, "Sex × Education composition in manufacturing (historical & projected)", ylabel="Overall share")

plot_wide_trends(sex_born_wide, hist_end_year, "Sex × Place of birth composition in manufacturing (historical & projected)", ylabel="Overall share")
