import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import ast
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# =========================
# Config
# =========================
CSV_PATH = "masterquarterly.csv"   # <-- change to your local path

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

# rolling multi-step validation
HORIZONS = [1, 4, 8, 12, 16]
WEIGHTS = {1: 0.35, 4: 0.25, 8: 0.20, 12: 0.12, 16: 0.08}

USE_LOGIT = True

# grid search
ORDERS = [(0, 1, 1), (1, 1, 0), (1, 1, 1), (0, 1, 0)]
SEASONALS = [(0, 0, 0, 0), (1, 0, 0, 4), (0, 0, 1, 4)]
TRENDS = ["n", "c"]

# uncertainty band from simulation
ALPHA_LOW, ALPHA_HIGH = 0.05, 0.95     # 90% interval
SIM_N = 300
BAND_ALPHA = 0.15                      # visual transparency only (same for male/female)

# coupled CP
RANK = 3
ALS_ITERS = 250
SEED = 42

# outputs
OUT_GRID = "grid_results_multistep.csv"
OUT_SEX_Q = "sex_share_forecast_quarterly_with_band.csv"
OUT_TARGET_SEX = "targets_annual_mean_sex_with_band.csv"
OUT_SEX_AGE = "pred_sex_age_with_band.csv"
OUT_SEX_EDU = "pred_sex_edu_with_band.csv"
OUT_SEX_BORN = "pred_sex_born_with_band.csv"

# =========================
# Helpers
# =========================
def parse_q(s) -> pd.Period:
    return pd.Period(str(s).strip().replace(" ", ""), freq="Q")

def qlabel(p: pd.Period) -> str:
    return f"{p.year}Q{p.quarter}"

def clip01(x, eps=1e-6):
    x = np.asarray(x, dtype=float)
    return np.clip(x, eps, 1 - eps)

def logit(p):
    p = clip01(p)
    return np.log(p / (1 - p))

def inv_logit(z):
    z = np.asarray(z, dtype=float)
    return 1 / (1 + np.exp(-z))

def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))

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
    # if optimizer says not converged, skip
    if hasattr(res, "mle_retvals") and isinstance(res.mle_retvals, dict):
        if not bool(res.mle_retvals.get("converged", True)):
            return None
    return res

def year_ticks(periods, step=5):
    start_y, end_y = periods[0].year, periods[-1].year
    y0 = start_y - (start_y % step)
    plist = list(periods)
    pos, lab = [], []
    for y in range(y0, end_y + 1, step):
        q = pd.Period(f"{y}Q1", freq="Q")
        if q in plist:
            pos.append(plist.index(q))
            lab.append(str(y))
    return pos, lab

def band_percent(alpha_low, alpha_high):
    return int(round((alpha_high - alpha_low) * 100))

def annual_mean_with_band(df_q):
    tmp = df_q.copy()
    tmp["__p__"] = tmp["Quarter"].map(parse_q)
    tmp["Year"] = tmp["__p__"].apply(lambda p: p.year)
    num_cols = [c for c in tmp.columns if c not in ["Quarter", "__p__", "Year"]]
    out = tmp.groupby("Year")[num_cols].mean().reset_index()
    return out


# =========================
# Plotting functions
# =========================
def plot_sex_share(p_all, hist_len,
                   male_mean, female_mean, male_p05, male_p95, female_p05, female_p95,
                   train_end_p, val_end_p,
                   mark_years, tick_step_years,
                   alpha_low, alpha_high, band_alpha):
    x = np.arange(len(p_all))
    plt.figure(figsize=(13, 6))

    # Male: same color for hist and forecast (solid vs dashed)
    m_line, = plt.plot(x[:hist_len], male_mean[:hist_len], label="Male", linestyle="-")
    c_m = m_line.get_color()
    plt.plot(x[hist_len-1:], male_mean[hist_len-1:], linestyle="--", color=c_m)

    # Female
    f_line, = plt.plot(x[:hist_len], female_mean[:hist_len], label="Female", linestyle="-")
    c_f = f_line.get_color()
    plt.plot(x[hist_len-1:], female_mean[hist_len-1:], linestyle="--", color=c_f)

    pct = band_percent(alpha_low, alpha_high)
    plt.fill_between(x[hist_len:], male_p05[hist_len:], male_p95[hist_len:], alpha=band_alpha, color=c_m,
                     label=f"Male {pct}% band")
    plt.fill_between(x[hist_len:], female_p05[hist_len:], female_p95[hist_len:], alpha=band_alpha, color=c_f,
                     label=f"Female {pct}% band")

    # Splits
    plt.axvline(p_all.index(train_end_p), color="k", linestyle="--", linewidth=1, label="Train/Val split")
    plt.axvline(p_all.index(val_end_p), color="k", linestyle="-.", linewidth=1, label="Val/Forecast split")

    # Target years
    for y in mark_years:
        q = pd.Period(f"{y}Q1", freq="Q")
        if q in p_all:
            plt.axvline(p_all.index(q), linestyle=":", linewidth=1)

    pos, lab = year_ticks(p_all, step=tick_step_years)
    plt.xticks(pos, lab)
    plt.xlabel("Year")
    plt.ylabel("Share")
    plt.title("Sex composition (quarterly, historical & projected)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def choose_top_band_keys(wide_df, sex_levels, cats, hist_len, top_k=1):
    # choose category per sex with largest mean share averaged over last 8 historical quarters
    start = max(0, hist_len - 8)
    end = hist_len
    keys = []
    for s in sex_levels:
        bases = [f"{s.replace(' ', '_')}_{c}".replace(" ", "_").replace(":", "") for c in cats]
        vals = []
        for b in bases:
            col = f"{b}_mean"
            if col in wide_df.columns:
                v = wide_df[col].iloc[start:end].astype(float).mean()
                vals.append((b, float(v)))
        vals.sort(key=lambda t: t[1], reverse=True)
        keys += [b for b, _ in vals[:top_k]]
    return keys


def plot_sex_x_all(wide_df, title, sex_levels, cats, hist_len,
                   train_end_p, val_end_p, mark_years, tick_step_years,
                   alpha_low, alpha_high, band_alpha, band_top_k=1,
                   label_fontsize=8, label_dx=6.0, min_label_gap=0.012):
    """
    Line style encodes ONLY: historical (solid) vs forecast (dashed).
    Sex is encoded by color + direct labels.
    Direct-label every line inside the axes near the right edge.
    """
    periods = [parse_q(q) for q in wide_df["Quarter"]]
    x = np.arange(len(periods))
    plt.figure(figsize=(13, 6))

    colors = {}
    end_idx = len(periods) - 1
    series_endpoints = []  # (y_end, base, color)

    # 1) plot all mean lines: historical solid, forecast dashed (same color)
    for s in sex_levels:
        for c in cats:
            base = f"{s.replace(' ', '_')}_{c}".replace(" ", "_").replace(":", "")
            col_mean = f"{base}_mean"
            if col_mean not in wide_df.columns:
                continue

            y = wide_df[col_mean].astype(float).values

            ln, = plt.plot(x[:hist_len], y[:hist_len],
                           linewidth=1.0, linestyle="-", label="_nolegend_")
            color = ln.get_color()
            plt.plot(x[hist_len-1:], y[hist_len-1:],
                     linewidth=1.0, linestyle="--", color=color, label="_nolegend_")

            colors[base] = color
            series_endpoints.append([float(y[end_idx]), base, color])

    # 2) bands only for top categories per sex
    pct = band_percent(alpha_low, alpha_high)
    top_keys = choose_top_band_keys(wide_df, sex_levels, cats, hist_len, top_k=band_top_k)
    for base in top_keys:
        col_p05 = f"{base}_p05"
        col_p95 = f"{base}_p95"
        if col_p05 in wide_df.columns and col_p95 in wide_df.columns:
            lo = wide_df[col_p05].astype(float).values
            hi = wide_df[col_p95].astype(float).values
            plt.fill_between(
                x[hist_len:], lo[hist_len:], hi[hist_len:],
                alpha=band_alpha, color=colors.get(base, None),
                label=f"{base} {pct}% band"
            )

    # 3) split lines with legend
    plt.axvline(periods.index(train_end_p), color="k", linestyle="--", linewidth=1, label="Train/Val split")
    plt.axvline(periods.index(val_end_p), color="k", linestyle="-.", linewidth=1, label="Val/Forecast split")

    # target years (no legend)
    for y in mark_years:
        q = pd.Period(f"{y}Q1", freq="Q")
        if q in periods:
            plt.axvline(periods.index(q), linestyle=":", linewidth=1, label="_nolegend_")

    # 4) direct labels INSIDE plot near right edge (avoid overlap)
    series_endpoints.sort(key=lambda t: t[0])
    adjusted = []
    for y_end, base, color in series_endpoints:
        y_adj = y_end
        if adjusted:
            y_prev = adjusted[-1][0]
            if y_adj - y_prev < min_label_gap:
                y_adj = y_prev + min_label_gap
        adjusted.append([y_adj, y_end, base, color])

    x_text = x[end_idx] - label_dx
    for y_adj, y_end, base, color in adjusted:
        plt.plot([x[end_idx]-0.2, x_text+0.2], [y_end, y_adj], color=color, linewidth=0.6, alpha=0.7)
        plt.text(
            x_text, y_adj, base,
            color=color, fontsize=label_fontsize, va="center", ha="left",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.5)
        )

    # 5) axes
    pos, lab = year_ticks(periods, step=tick_step_years)
    plt.xticks(pos, lab)
    plt.xlabel("Year")
    plt.ylabel("Overall share")
    plt.title(title)

    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


# =========================
# 1) Load + build male share series
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

tot = df.pivot_table(index="__p__", columns=SEX_COL, values=TOTAL_AGE_COL, aggfunc="sum").reindex(p_hist)
tot = tot.interpolate("linear", limit_direction="both").ffill().bfill()
total_all = tot.sum(axis=1).replace(0.0, np.nan)
male_share = (tot[male_label] / total_all).fillna(0.0)

if len(male_share) < TRAIN_N + VAL_N:
    raise ValueError(f"Not enough points: got {len(male_share)}, need {TRAIN_N+VAL_N}.")

# sanity check
recent = male_share.values[-20:]
slope = np.polyfit(np.arange(20), recent, 1)[0]
std = float(np.std(recent))
print("\nSanity check (last 5 years / 20 quarters):")
print(f"  std(male_share) = {std:.6f}")
print(f"  slope per quarter = {slope:.6e}  (per year approx {4*slope:.6e})")


# =========================
# 2) Rolling multi-step evaluation + grid search (trend n/c)
# =========================
def rolling_eval(series, order, seas, trend):
    y = series.values.astype(float)
    start, end = TRAIN_N, TRAIN_N + VAL_N
    store = {h: {"pred": [], "true": []} for h in HORIZONS}

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
            store[h]["pred"].append(float(pred))
            store[h]["true"].append(float(y[idx]))

    out = {}
    for h in HORIZONS:
        if len(store[h]["true"]) == 0:
            return None
        p = clip01(store[h]["true"])
        q = clip01(store[h]["pred"])
        out[h] = {"MAE": mae(p, q), "KL": bern_kl(p, q), "N": len(p)}
    return out

rows = []
for order in ORDERS:
    for seas in SEASONALS:
        for tr in TRENDS:
            m = rolling_eval(male_share, order, seas, tr)
            if m is None:
                continue
            wKL = sum(WEIGHTS[h] * m[h]["KL"] for h in HORIZONS)
            wMAE = sum(WEIGHTS[h] * m[h]["MAE"] for h in HORIZONS)
            row = {"order": str(order), "seasonal_order": str(seas), "trend": tr,
                   "use_logit": USE_LOGIT, "wKL": wKL, "wMAE": wMAE}
            for h in HORIZONS:
                row[f"KL_h{h}"] = m[h]["KL"]
                row[f"MAE_h{h}"] = m[h]["MAE"]
                row[f"N_h{h}"] = m[h]["N"]
            rows.append(row)

grid = pd.DataFrame(rows).sort_values(["wKL", "wMAE"]).reset_index(drop=True)
grid.to_csv(OUT_GRID, index=False)
best = grid.iloc[0]
print("\nSaved grid results:", OUT_GRID)
print("Best config:", best.to_dict())

best_order = ast.literal_eval(best["order"])
best_seas = ast.literal_eval(best["seasonal_order"])
best_trend = best["trend"]


# =========================
# 3) Fit best on full history -> forecast mean + simulate band
# =========================
p_end = parse_q(FORECAST_END)
p_all = list(pd.period_range(p_hist[0], p_end, freq="Q"))
steps_ahead = len(p_all) - len(p_hist)
hist_len = len(p_hist)

y_fit = logit(male_share.values) if USE_LOGIT else male_share.values
res_full = fit_sarimax(y_fit, best_order, best_seas, best_trend)
if res_full is None:
    raise RuntimeError("Best model failed to fit on full history.")

fc = res_full.forecast(steps=steps_ahead)
fc = inv_logit(fc) if USE_LOGIT else np.asarray(fc, float)
fc = clip01(fc)

male_mean = np.concatenate([male_share.values, fc])
female_mean = 1.0 - male_mean

# simulate
sim = res_full.simulate(nsimulations=steps_ahead, repetitions=SIM_N, anchor="end")
sim = np.asarray(sim)
sim = np.squeeze(sim)
if sim.ndim == 1:
    sim = sim.reshape(-1, 1)
if sim.shape[0] != steps_ahead and sim.shape[1] == steps_ahead:
    sim = sim.T

sim = inv_logit(sim) if USE_LOGIT else sim
sim = clip01(sim)

q_lo = np.quantile(sim, ALPHA_LOW, axis=1)
q_hi = np.quantile(sim, ALPHA_HIGH, axis=1)

male_p05 = male_mean.copy()
male_p95 = male_mean.copy()
male_p05[hist_len:] = q_lo
male_p95[hist_len:] = q_hi

female_p05 = 1.0 - male_p95
female_p95 = 1.0 - male_p05

sex_q = pd.DataFrame({
    "Quarter": [qlabel(p) for p in p_all],
    "male_share_mean": male_mean,
    "male_share_p05": male_p05,
    "male_share_p95": male_p95,
    "female_share_mean": female_mean,
    "female_share_p05": female_p05,
    "female_share_p95": female_p95,
})
sex_q.to_csv(OUT_SEX_Q, index=False)
print("\nSaved:", OUT_SEX_Q)

targets = annual_mean_with_band(sex_q)
targets = targets[targets["Year"].isin(MARK_YEARS)].reset_index(drop=True)
targets.to_csv(OUT_TARGET_SEX, index=False)
print("Saved:", OUT_TARGET_SEX)
print("\nTarget-year annual mean sex shares (mean/p05/p95):")
print(targets.to_string(index=False))

# boundaries
train_end_p = p_hist[TRAIN_N-1]
val_end_p = p_hist[TRAIN_N+VAL_N-1]

# plot sex share (with female band + consistent colors)
plot_sex_share(
    p_all, hist_len,
    male_mean, female_mean,
    male_p05, male_p95,
    female_p05, female_p95,
    train_end_p, val_end_p,
    MARK_YEARS, TICK_STEP_YEARS,
    ALPHA_LOW, ALPHA_HIGH, BAND_ALPHA
)


# =========================
# 4) Coupled CP on conditional shares + propagate sex band to sex×(age/edu/born)
# =========================
def build_conditional_long(cols):
    use = ["__p__", SEX_COL, TOTAL_AGE_COL] + [c for c in cols if c in df.columns]
    tmp = df[use].copy()
    denom = tmp[TOTAL_AGE_COL].replace(0.0, np.nan)
    for c in cols:
        tmp[c] = tmp[c] / denom
    return tmp[["__p__", SEX_COL] + cols]

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
            KR = khatri_rao(V, U)
            W = solve_ls(unfold2(X), KR)
            Ws[k] = np.clip(W, 1e-10, None)

        num = np.zeros((T,R)); den = np.zeros((R,R))
        for k, X in enumerate(Xs):
            KR = khatri_rao(Ws[k], V)
            num += unfold0(X) @ KR
            den += KR.T @ KR
        U = np.clip(num @ np.linalg.inv(den + 1e-8*np.eye(R)), 1e-10, None)

        num = np.zeros((S,R)); den = np.zeros((R,R))
        for k, X in enumerate(Xs):
            KR = khatri_rao(Ws[k], U)
            num += unfold1(X) @ KR
            den += KR.T @ KR
        V = np.clip(num @ np.linalg.inv(den + 1e-8*np.eye(R)), 1e-10, None)

    return U, V, Ws

def reconstruct(U, V, W):
    X = np.einsum("tr,sr,kr->tsk", U, V, W)
    X = np.clip(X, 0.0, None)
    d = X.sum(axis=2, keepdims=True); d[d <= 0] = 1.0
    return X / d

def forecast_latent(u_hist, steps):
    # compact default model for latent factors
    res = fit_sarimax(np.asarray(u_hist, float), order=(1,1,0), seasonal_order=(1,0,0,4), trend="c")
    if res is None:
        a, b = np.polyfit(np.arange(len(u_hist)), np.asarray(u_hist, float), 1)
        return a*np.arange(len(u_hist), len(u_hist)+steps) + b
    return np.asarray(res.forecast(steps=steps), float)

AGE_USE = [c for c in AGE_COLS if c in df.columns]
EDU_USE = [c for c in EDU_COLS if c in df.columns]
BORN_USE = [c for c in BORN_COLS if c in df.columns]

X_age = to_tensor(build_conditional_long(AGE_USE), AGE_USE, p_hist, sex_levels)
X_edu = to_tensor(build_conditional_long(EDU_USE), EDU_USE, p_hist, sex_levels)
X_born = to_tensor(build_conditional_long(BORN_USE), BORN_USE, p_hist, sex_levels)

U_hist, V_sex, (W_age, W_edu, W_born) = coupled_cp_als([X_age, X_edu, X_born], rank=RANK, iters=ALS_ITERS, seed=SEED)

U_future = np.zeros((steps_ahead, RANK))
for r in range(RANK):
    U_future[:, r] = forecast_latent(U_hist[:, r], steps_ahead)
U_all = np.vstack([U_hist, U_future])

Xhat_age = reconstruct(U_all, V_sex, W_age)
Xhat_edu = reconstruct(U_all, V_sex, W_edu)
Xhat_born = reconstruct(U_all, V_sex, W_born)

# sex share dfs for propagation
sex_mean_df = pd.DataFrame({male_label: male_mean, female_label: 1.0-male_mean}, index=p_all)
sex_p05_df  = pd.DataFrame({male_label: male_p05,  female_label: 1.0-male_p95}, index=p_all)
sex_p95_df  = pd.DataFrame({male_label: male_p95,  female_label: 1.0-male_p05}, index=p_all)

def build_wide_with_band(Xhat, cats, out_csv):
    rows = []
    sex_to_j = {s: j for j, s in enumerate(sex_levels)}
    for i, p in enumerate(p_all):
        q = qlabel(p)
        for s in sex_levels:
            j = sex_to_j[s]
            cond = Xhat[i, j, :]
            ps_m = float(sex_mean_df.loc[p, s])
            ps_l = float(sex_p05_df.loc[p, s])
            ps_u = float(sex_p95_df.loc[p, s])
            for k, cat in enumerate(cats):
                rows.append({
                    "Quarter": q, "Sex": s, "Category": cat,
                    "mean": ps_m * float(cond[k]),
                    "p05":  ps_l * float(cond[k]),
                    "p95":  ps_u * float(cond[k]),
                })
    long = pd.DataFrame(rows)
    long["col"] = long["Sex"].str.replace(" ", "_") + "_" + long["Category"].astype(str).str.replace(" ", "_").str.replace(":", "", regex=False)

    wide = pd.DataFrame({"Quarter": sorted(long["Quarter"].unique(), key=lambda x: parse_q(x))})
    for suffix in ["mean", "p05", "p95"]:
        mat = long.pivot(index="Quarter", columns="col", values=suffix).reset_index()
        mat = mat.rename(columns={c: f"{c}_{suffix}" for c in mat.columns if c != "Quarter"})
        wide = wide.merge(mat, on="Quarter", how="left")

    wide.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
    return wide

wide_age = build_wide_with_band(Xhat_age, AGE_USE, OUT_SEX_AGE)
wide_edu = build_wide_with_band(Xhat_edu, EDU_USE, OUT_SEX_EDU)
wide_born = build_wide_with_band(Xhat_born, BORN_USE, OUT_SEX_BORN)

# plot all mean lines + band for top category per sex
plot_sex_x_all(wide_age,  "Sex × Age (mean lines for all categories; band for top shares)",
               sex_levels, AGE_USE, hist_len, train_end_p, val_end_p, MARK_YEARS, TICK_STEP_YEARS,
               ALPHA_LOW, ALPHA_HIGH, BAND_ALPHA, band_top_k=1)

plot_sex_x_all(wide_edu,  "Sex × Education (mean lines for all categories; band for top shares)",
               sex_levels, EDU_USE, hist_len, train_end_p, val_end_p, MARK_YEARS, TICK_STEP_YEARS,
               ALPHA_LOW, ALPHA_HIGH, BAND_ALPHA, band_top_k=1)

plot_sex_x_all(wide_born, "Sex × Place of birth (mean lines for all categories; band for top shares)",
               sex_levels, BORN_USE, hist_len, train_end_p, val_end_p, MARK_YEARS, TICK_STEP_YEARS,
               ALPHA_LOW, ALPHA_HIGH, BAND_ALPHA, band_top_k=1)



