import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# =========================
# Config
# =========================
CSV_PATH = "masterquarterly.csv"   

TIME_COL = "Quarter"
SEX_COL = "Sex"
TOTAL_AGE_COL = "Total for age"

AGE_COLS = ["15-24", "25-54", "55-64", "65+"]
EDU_COLS = ["Basic and below basic", "Intermediate", "Advanced", "Level not stated"]
BORN_COLS = ["Place of birth: Foreign-born", "Place of birth: Native-born", "Place of birth: Status unknown"]

TRAIN_N = 99
VAL_N = 20

FORECAST_END = "2050Q4"
MARK_YEARS = [2026, 2030, 2035, 2050]
TICK_STEP_YEARS = 5

HORIZONS = [1, 4, 8]                 # in quarters
WEIGHTS = {1: 0.6, 4: 0.25, 8: 0.15}
USE_LOGIT = True

RANK = 3
ALS_ITERS = 250
SEED = 42

# Outputs
OUT_GRID = "grid_results_multistep.csv"
OUT_SEX_Q = "sex_share_forecast_quarterly.csv"
OUT_TARGET_SEX = "targets_annual_mean_sex.csv"
OUT_SEX_AGE = "pred_sex_age.csv"
OUT_SEX_EDU = "pred_sex_edu.csv"
OUT_SEX_BORN = "pred_sex_born.csv"


# =========================
# Helpers
# =========================
def parse_q(s): return pd.Period(str(s).strip().replace(" ", ""), freq="Q")
def qlabel(p): return f"{p.year}Q{p.quarter}"

def clip01(x, eps=1e-6):
    x = np.asarray(x, dtype=float)
    return np.clip(x, eps, 1 - eps)

def logit(p):
    p = clip01(p)
    return np.log(p / (1 - p))

def inv_logit(z):
    z = np.asarray(z, dtype=float)
    return 1 / (1 + np.exp(-z))

def mae(a, b): return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))

def bern_kl(p, q):
    p = clip01(p); q = clip01(q)
    return float(np.mean(p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))))

def fit_sarimax(y, order, seasonal_order, trend):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        res = SARIMAX(
            y, order=order, seasonal_order=seasonal_order, trend=trend,
            enforce_stationarity=False, enforce_invertibility=False
        ).fit(disp=False)
    if hasattr(res, "mle_retvals") and isinstance(res.mle_retvals, dict):
        if not bool(res.mle_retvals.get("converged", True)):
            return None
    return res

def annual_mean_from_quarterly(df_q, year_col="Year"):
    # df_q has Quarter labels; compute mean of all quarters per year
    tmp = df_q.copy()
    tmp["__p__"] = tmp["Quarter"].map(parse_q)
    tmp[year_col] = tmp["__p__"].apply(lambda p: p.year)
    num_cols = [c for c in tmp.columns if c not in ["Quarter", "__p__", year_col]]
    return tmp.groupby(year_col)[num_cols].mean().reset_index()

def year_ticks(periods, step=5):
    start_y = periods[0].year
    end_y = periods[-1].year
    y0 = start_y - (start_y % step)
    pos, lab = [], []
    plist = list(periods)
    for y in range(y0, end_y + 1, step):
        q = pd.Period(f"{y}Q1", freq="Q")
        if q in plist:
            pos.append(plist.index(q))
            lab.append(str(y))
    return pos, lab

def plot_series_with_marks(periods, y_hist, y_all, title, ylabel, marks):
    x = np.arange(len(periods))
    hist_len = len(y_hist)
    plt.figure(figsize=(13, 6))
    plt.plot(x[:hist_len], y_all[:hist_len], label="Historical")
    plt.plot(x[hist_len-1:], y_all[hist_len-1:], linestyle="--", label="Forecast")
    for y in marks:
        q = pd.Period(f"{y}Q1", freq="Q")
        if q in periods:
            plt.axvline(list(periods).index(q), linestyle=":", linewidth=1)
    pos, lab = year_ticks(periods, step=TICK_STEP_YEARS)
    plt.xticks(pos, lab)
    plt.xlabel("Year")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================
# 1) Load + build continuous history
# =========================
df = pd.read_csv(CSV_PATH)
df.columns = [c.strip() for c in df.columns]
df[TIME_COL] = df[TIME_COL].astype(str).str.strip()
df["__p__"] = df[TIME_COL].map(parse_q)
df[SEX_COL] = df[SEX_COL].astype(str).str.strip()
df[TOTAL_AGE_COL] = pd.to_numeric(df[TOTAL_AGE_COL], errors="coerce")

for c in AGE_COLS + EDU_COLS + BORN_COLS:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

sex_levels = sorted(df[SEX_COL].dropna().unique().tolist())
male_label = next((s for s in sex_levels if str(s).lower().startswith("male")), sex_levels[0])
female_label = next((s for s in sex_levels if s != male_label), None)
if female_label is None:
    raise ValueError("Need at least two sex categories.")

p_obs = sorted(df["__p__"].dropna().unique().tolist())
p_hist = list(pd.period_range(p_obs[0], p_obs[-1], freq="Q"))

# totals by sex -> interpolate -> male share
tot = df.pivot_table(index="__p__", columns=SEX_COL, values=TOTAL_AGE_COL, aggfunc="sum").reindex(p_hist)
tot = tot.interpolate("linear", limit_direction="both").ffill().bfill()
total_all = tot.sum(axis=1).replace(0.0, np.nan)
male_share = (tot[male_label] / total_all).fillna(0.0)

if len(male_share) < TRAIN_N + VAL_N:
    raise ValueError(f"Not enough points: got {len(male_share)}, need {TRAIN_N+VAL_N}.")


# =========================
# 2) Sanity check (recent 5y)
# =========================
last_5y = 20
recent = male_share.values[-last_5y:]
x = np.arange(last_5y)
slope = np.polyfit(x, recent, 1)[0]          # per quarter
std = float(np.std(recent))
print("\nSanity check (last 5 years / 20 quarters):")
print(f"  std(male_share) = {std:.6f}")
print(f"  slope per quarter = {slope:.6e}  (per year approx {4*slope:.6e})")


# =========================
# 3) Rolling multi-step grid search
# =========================
def rolling_multistep_eval(series, order, seas, trend):
    y = series.values.astype(float)
    start = TRAIN_N
    end = TRAIN_N + VAL_N
    metrics = {h: {"pred": [], "true": []} for h in HORIZONS}

    for t in range(start, end):
        y_train = y[:t]
        y_fit = logit(y_train) if USE_LOGIT else y_train
        res = fit_sarimax(y_fit, order, seas, trend)
        if res is None:
            return None
        for h in HORIZONS:
            idx = t + h - 1
            if idx >= end:
                continue
            fc = res.forecast(steps=h)[-1]
            pred = inv_logit(fc) if USE_LOGIT else float(fc)
            metrics[h]["pred"].append(float(pred))
            metrics[h]["true"].append(float(y[idx]))

    out = {}
    for h in HORIZONS:
        if len(metrics[h]["true"]) == 0:
            return None
        p = clip01(metrics[h]["true"])
        q = clip01(metrics[h]["pred"])
        out[h] = {"MAE": mae(p, q), "KL": bern_kl(p, q), "N": len(p)}
    return out

orders = [(0,1,1), (1,1,0), (1,1,1), (0,1,0)]
seasonals = [(0,0,0,0), (1,0,0,4), (0,0,1,4)]
trends = ["n", "c"]

rows = []
for order in orders:
    for seas in seasonals:
        for tr in trends:
            m = rolling_multistep_eval(male_share, order, seas, tr)
            if m is None:
                continue
            loss = sum(WEIGHTS[h] * m[h]["MAE"] for h in HORIZONS)
            row = {"order": str(order), "seasonal_order": str(seas), "trend": tr,
                   "use_logit": USE_LOGIT, "loss_weighted_MAE": loss}
            for h in HORIZONS:
                row[f"MAE_h{h}"] = m[h]["MAE"]
                row[f"KL_h{h}"] = m[h]["KL"]
                row[f"N_h{h}"] = m[h]["N"]
            rows.append(row)

grid = pd.DataFrame(rows).sort_values("loss_weighted_MAE").reset_index(drop=True)
grid.to_csv(OUT_GRID, index=False)
best = grid.iloc[0]
print("\nSaved grid results:", OUT_GRID)
print("Best config:", best.to_dict())

best_order = eval(best["order"])
best_seas = eval(best["seasonal_order"])
best_trend = best["trend"]


# =========================
# 4) Fit best on full history -> forecast to 2050Q4
#    + simulate paths to show "波动" (uncertainty), not just mean
# =========================
p_end = parse_q(FORECAST_END)
p_all = list(pd.period_range(p_hist[0], p_end, freq="Q"))
steps_ahead = len(p_all) - len(p_hist)

y_fit = logit(male_share.values) if USE_LOGIT else male_share.values
res_full = fit_sarimax(y_fit, best_order, best_seas, best_trend)
if res_full is None:
    raise RuntimeError("Best model failed to fit on full history.")

fc_mean = res_full.forecast(steps=steps_ahead)
fc_mean = inv_logit(fc_mean) if USE_LOGIT else np.asarray(fc_mean, float)
fc_mean = clip01(fc_mean)

male_all_mean = np.concatenate([male_share.values, fc_mean])
female_all_mean = 1.0 - male_all_mean

# simulate future paths (this is what visually produces "波动")
SIM_N = 200
sim = res_full.simulate(nsimulations=steps_ahead, repetitions=SIM_N, anchor="end")
# sim shape: (steps, reps) in transformed space if USE_LOGIT
sim = inv_logit(sim) if USE_LOGIT else sim
sim = clip01(sim)
sim = np.asarray(sim)
sim = np.squeeze(sim)

# make it (steps, reps)
if sim.shape[0] != steps_ahead and sim.shape[1] == steps_ahead:
    sim = sim.T
sim_q05 = np.quantile(sim, 0.05, axis=1)
sim_q95 = np.quantile(sim, 0.95, axis=1)

# save quarterly forecast
sex_q = pd.DataFrame({
    "Quarter": [qlabel(p) for p in p_all],
    "male_share_mean": male_all_mean,
    "female_share_mean": female_all_mean
})
sex_q.to_csv(OUT_SEX_Q, index=False)
print("\nSaved:", OUT_SEX_Q)

# annual targets (mean of quarters in each year)
targets = annual_mean_from_quarterly(sex_q[["Quarter", "male_share_mean", "female_share_mean"]], year_col="Year")
targets = targets[targets["Year"].isin(MARK_YEARS)].reset_index(drop=True)
targets.to_csv(OUT_TARGET_SEX, index=False)
print("Saved:", OUT_TARGET_SEX)
print("\nTarget-year annual mean sex shares:")
print(targets.to_string(index=False))

# plot sex share with uncertainty band
x = np.arange(len(p_all))
hist_len = len(p_hist)
plt.figure(figsize=(13, 6))
plt.plot(x[:hist_len], male_all_mean[:hist_len], label="Male (hist)")
plt.plot(x[:hist_len], female_all_mean[:hist_len], label="Female (hist)")

plt.plot(x[hist_len-1:], male_all_mean[hist_len-1:], linestyle="--", label="Male (mean fc)")
plt.plot(x[hist_len-1:], female_all_mean[hist_len-1:], linestyle="--", label="Female (mean fc)")

# uncertainty band (future only)
xf = x[hist_len:]
plt.fill_between(xf, sim_q05, sim_q95, alpha=0.2, label="Male 90% band (sim)")

for y in MARK_YEARS:
    q = pd.Period(f"{y}Q1", freq="Q")
    if q in p_all:
        plt.axvline(p_all.index(q), linestyle=":", linewidth=1)

pos, lab = year_ticks(p_all, step=TICK_STEP_YEARS)
plt.xticks(pos, lab)
plt.xlabel("Year")
plt.ylabel("Share")
plt.title("Sex composition (quarterly, historical & projected)")
plt.legend()
plt.tight_layout()
plt.show()


# =========================
# 5) Coupled CP tensor factorisation (sex×age / sex×edu / sex×born)
# =========================
def build_conditional_long(cols):
    use = ["__p__", SEX_COL, TOTAL_AGE_COL] + [c for c in cols if c in df.columns]
    tmp = df[use].copy()
    denom = tmp[TOTAL_AGE_COL].replace(0.0, np.nan)
    for c in cols:
        if c in tmp.columns:
            tmp[c] = tmp[c] / denom
    return tmp[["__p__", SEX_COL] + [c for c in cols if c in tmp.columns]]

def to_tensor(long_df, cols, periods, sexes):
    idx = pd.Index(periods, name="__p__")
    X = np.zeros((len(periods), len(sexes), len(cols)), float)
    for j, s in enumerate(sexes):
        sub = long_df[long_df[SEX_COL] == s].set_index("__p__")[cols].reindex(idx)
        sub = sub.interpolate("linear", limit_direction="both").ffill().bfill().fillna(0.0)
        v = np.clip(sub.values, 0.0, None)
        d = v.sum(axis=1, keepdims=True); d[d <= 0] = 1.0
        X[:, j, :] = v / d
    return X

def khatri_rao(A, B):
    return np.einsum("ir,jr->ijr", A, B).reshape(A.shape[0]*B.shape[0], A.shape[1])

def unfold0(X): return X.reshape(X.shape[0], -1)
def unfold1(X): return np.transpose(X, (1,0,2)).reshape(X.shape[1], -1)
def unfold2(X): return np.transpose(X, (2,0,1)).reshape(X.shape[2], -1)

def solve_ls(Xm, KR):
    G = KR.T @ KR + 1e-8*np.eye(KR.shape[1])
    return (Xm @ KR) @ np.linalg.inv(G)

def coupled_cp_als(Xs, rank=3, iters=200, seed=0):
    rng = np.random.default_rng(seed)
    T, S, _ = Xs[0].shape
    R = rank
    U = rng.random((T,R)) + 1e-3
    V = rng.random((S,R)) + 1e-3
    Ws = [rng.random((X.shape[2],R)) + 1e-3 for X in Xs]

    for _ in range(iters):
        for k, X in enumerate(Xs):
            KR = khatri_rao(V, U)          # (S*T,R)
            W = solve_ls(unfold2(X), KR)
            Ws[k] = np.clip(W, 1e-10, None)

        num = np.zeros((T,R)); den = np.zeros((R,R))
        for k, X in enumerate(Xs):
            KR = khatri_rao(Ws[k], V)      # (K*S,R)
            num += unfold0(X) @ KR
            den += KR.T @ KR
        U = np.clip(num @ np.linalg.inv(den + 1e-8*np.eye(R)), 1e-10, None)

        num = np.zeros((S,R)); den = np.zeros((R,R))
        for k, X in enumerate(Xs):
            KR = khatri_rao(Ws[k], U)      # (K*T,R)
            num += unfold1(X) @ KR
            den += KR.T @ KR
        V = np.clip(num @ np.linalg.inv(den + 1e-8*np.eye(R)), 1e-10, None)

    return U, V, Ws

def reconstruct(U, V, W):
    X = np.einsum("tr,sr,kr->tsk", U, V, W)
    X = np.clip(X, 0.0, None)
    d = X.sum(axis=2, keepdims=True); d[d <= 0] = 1.0
    return X / d

# build tensors on historical quarters
age_long = build_conditional_long(AGE_COLS)
edu_long = build_conditional_long(EDU_COLS)
born_long = build_conditional_long(BORN_COLS)

AGE_USE = [c for c in AGE_COLS if c in df.columns]
EDU_USE = [c for c in EDU_COLS if c in df.columns]
BORN_USE = [c for c in BORN_COLS if c in df.columns]

X_age = to_tensor(age_long, AGE_USE, p_hist, sex_levels)
X_edu = to_tensor(edu_long, EDU_USE, p_hist, sex_levels)
X_born = to_tensor(born_long, BORN_USE, p_hist, sex_levels)

U_hist, V_sex, (W_age, W_edu, W_born) = coupled_cp_als(
    [X_age, X_edu, X_born], rank=RANK, iters=ALS_ITERS, seed=SEED
)

# forecast latent time factors with a simple default SARIMAX (keep compact)
def forecast_latent(u_hist, steps):
    y = np.asarray(u_hist, float)
    res = fit_sarimax(y, order=(1,1,0), seasonal_order=(1,0,0,4), trend="c")
    if res is None:
        # linear fallback
        x = np.arange(len(y))
        a, b = np.polyfit(x, y, 1)
        return a*np.arange(len(y), len(y)+steps) + b
    return np.asarray(res.forecast(steps=steps), float)

U_fut = np.vstack([U_hist] + [np.zeros((steps_ahead, RANK))])  # placeholder to keep shapes
U_future = np.zeros((steps_ahead, RANK))
for r in range(RANK):
    U_future[:, r] = forecast_latent(U_hist[:, r], steps_ahead)
U_all = np.vstack([U_hist, U_future])

Xhat_age = reconstruct(U_all, V_sex, W_age)
Xhat_edu = reconstruct(U_all, V_sex, W_edu)
Xhat_born = reconstruct(U_all, V_sex, W_born)

# build overall shares = sex_share_mean * conditional
sex_share_pred = pd.DataFrame({
    male_label: male_all_mean,
    female_label: 1.0 - male_all_mean
}, index=p_all)

def long_overall(Xhat, cats, kind):
    rows = []
    sex_to_j = {s:j for j,s in enumerate(sex_levels)}
    for i, p in enumerate(p_all):
        for s in sex_levels:
            j = sex_to_j[s]
            ps = float(sex_share_pred.loc[p, s])
            for k, c in enumerate(cats):
                rows.append({"Quarter": qlabel(p), "Sex": s, "Category": c, "Share": ps*float(Xhat[i,j,k]), "Kind": kind})
    return pd.DataFrame(rows)

def wide_sex_cat(long_df, cats_name):
    tmp = long_df.copy()
    tmp["col"] = tmp["Sex"].map(lambda x: x.replace(" ", "_")) + "_" + tmp["Category"].map(lambda x: str(x).replace(" ", "_").replace(":", ""))
    return tmp.pivot(index="Quarter", columns="col", values="Share").reset_index()

sex_age = wide_sex_cat(long_overall(Xhat_age, AGE_USE, "age"), "Age")
sex_edu = wide_sex_cat(long_overall(Xhat_edu, EDU_USE, "edu"), "Education")
sex_born = wide_sex_cat(long_overall(Xhat_born, BORN_USE, "born"), "Birthplace")

sex_age.to_csv(OUT_SEX_AGE, index=False)
sex_edu.to_csv(OUT_SEX_EDU, index=False)
sex_born.to_csv(OUT_SEX_BORN, index=False)
print("\nSaved:", OUT_SEX_AGE, OUT_SEX_EDU, OUT_SEX_BORN)

# plots (overall composition in each table)
def plot_wide(wide_df, title):
    periods = [parse_q(q) for q in wide_df["Quarter"]]
    x = np.arange(len(periods))
    hist_end = p_hist[-1]
    hist_mask = np.array([p <= hist_end for p in periods])

    plt.figure(figsize=(13,6))
    for c in wide_df.columns:
        if c == "Quarter": continue
        y = wide_df[c].astype(float).values
        plt.plot(x[hist_mask], y[hist_mask])
        plt.plot(x[~hist_mask], y[~hist_mask], linestyle="--")

    for y in MARK_YEARS:
        q = pd.Period(f"{y}Q1", freq="Q")
        if q in periods:
            plt.axvline(periods.index(q), linestyle=":", linewidth=1)

    pos, lab = year_ticks(periods, step=TICK_STEP_YEARS)
    plt.xticks(pos, lab)
    plt.xlabel("Year")
    plt.ylabel("Overall share")
    plt.title(title)
    plt.tight_layout()
    plt.show()

plot_wide(sex_age, "Sex × Age composition (quarterly, historical & projected)")
plot_wide(sex_edu, "Sex × Education composition (quarterly, historical & projected)")
plot_wide(sex_born, "Sex × Place of birth composition (quarterly, historical & projected)")

