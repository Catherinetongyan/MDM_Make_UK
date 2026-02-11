import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import argparse
from pathlib import Path
import pytensor
import pytensor.tensor as pt
import re

CSV_PATH = "masterquarterly.csv"

AGE_COLS = [
    "Aggregate bands: 15-24",
    "Aggregate bands: 25-54",
    "Aggregate bands: 55-64",
    "Aggregate bands: 65+",
]

COVARIATE_COL = "GDP growth"
DEFAULT_GDP_LAGS = 4

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
})

def softmax_np(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)

def extract_category(df, cols, sex):
    sub = df[df["Sex"] == sex].sort_values("Quarter")
    y = sub[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
    y = np.round(y).astype(int)
    W = y.sum(axis=1).astype(int)
    quarters = sub["Quarter"].astype(str).to_numpy()
    return y, W, quarters

def make_lag_matrix(series_std, L):
    T = len(series_std)
    X = np.zeros((T, L + 1))
    for t in range(T):
        row = [series_std[t]]
        for l in range(1, L + 1):
            idx = t - l
            row.append(series_std[idx] if idx >= 0 else 0.0)
        X[t] = row
    return X

def extract_gdp_lags(df, sex, L=DEFAULT_GDP_LAGS):
    sub = df[df["Sex"] == sex].sort_values("Quarter")
    gdp_raw = pd.to_numeric(sub[COVARIATE_COL], errors="coerce").fillna(0.0).to_numpy()
    mean = gdp_raw.mean()
    std = gdp_raw.std() + 1e-6
    gdp_std = (gdp_raw - mean) / std
    X = make_lag_matrix(gdp_std, L)
    return X, gdp_std, mean, std

def build_future_gdp_lags(gdp_std_hist, gdp_future_raw, mean, std, L):
    gdp_future_std = (np.asarray(gdp_future_raw) - mean) / (std)
    buf = list(gdp_std_hist[-L:][::-1])  
    H = len(gdp_future_std)
    Xf = np.zeros((H, L + 1))
    for h in range(H):
        current = float(gdp_future_std[h])
        row = [current] + buf[:L]
        Xf[h] = row
        buf = [current] + buf[:L-1]
    return Xf

def load_scenario_file(path: Path, horizon: int):
    df = pd.read_csv(path)
    col = COVARIATE_COL
    if col not in df.columns:
        raise ValueError(f"Scenario file must contain column '{col}'")
    arr = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()
    if len(arr) >= horizon:
        return arr[:horizon]
    if len(arr) == 0:
        return np.zeros(horizon)
    pad = np.full(horizon - len(arr), arr[-1])
    return np.concatenate([arr, pad])

def make_constant_scenario(value: float, horizon: int):
    return np.full(horizon, float(value))

def make_shock_scenario(value: float, shock_len: int, horizon: int):
    shock_len = max(0, min(horizon, int(shock_len)))
    head = np.full(shock_len, float(value))
    tail = np.zeros(horizon - shock_len)
    return np.concatenate([head, tail])

def make_ramp_scenario(start: float, end: float, horizon: int):
    return np.linspace(float(start), float(end), horizon)

def make_piecewise_scenario(profile: str, horizon: int):
    if not profile:
        return np.zeros(horizon)
    segments = []
    for part in profile.split(';'):
        part = part.strip()
        if not part:
            continue
        try:
            val_str, len_str = part.split(':')
            val = float(val_str)
            dur = int(len_str)
        except Exception:
            raise ValueError("Invalid --scenario-profile format. Use 'value:len;value:len;...'")
        if dur <= 0:
            continue
        segments.append((val, dur))
    seq = []
    for val, dur in segments:
        seq.extend([val] * dur)
    arr = np.array(seq, dtype=float)
    if len(arr) >= horizon:
        return arr[:horizon]
    pad = np.zeros(horizon - len(arr))
    return np.concatenate([arr, pad])

def offset_scenario_vector(vec, offset: int, horizon: int):
    if vec is None:
        return None
    offset = int(max(0, offset))
    head = np.zeros(offset)
    arr = np.concatenate([head, np.asarray(vec, dtype=float)])
    if len(arr) >= horizon:
        return arr[:horizon]
    pad = np.zeros(horizon - len(arr))
    return np.concatenate([arr, pad])

def compute_scenario_offset(quarters, scenario_at: str):
    if not scenario_at:
        return 0
    pi_obs = pd.PeriodIndex(quarters, freq="Q")
    last_p = pi_obs[-1]
    target_p = parse_quarter_str(scenario_at)
    return max(0, int(target_p.ordinal - last_p.ordinal - 1))
    if not scenario_at:
        return 0
    pi_obs = pd.PeriodIndex(quarters, freq="Q")
    last_p = pi_obs[-1]
    target_p = parse_quarter_str(scenario_at)
    return max(0, int(target_p.ordinal - last_p.ordinal - 1))

def parse_quarter_str(s: str):
    s_norm = str(s).strip().upper().replace(" ", "").replace("-", "")
    m_rev = re.match(r'^Q([1-4])(\d{4})$', s_norm)
    if m_rev:
        s_norm = f"{m_rev.group(2)}Q{m_rev.group(1)}"
    m = re.match(r'^(\d{4})Q([1-4])$', s_norm)
    if not m:
        raise ValueError(f"Cannot parse quarter string: {s}")
    year = int(m.group(1))
    q = int(m.group(2))
    return pd.Period(f"{year}Q{q}", freq="Q")

def summarize_samples(samples, quarters, labels, hdi_prob=0.9, observed=None):
    median = np.median(samples, axis=0)
    lo = np.quantile(samples, (1 - hdi_prob) / 2, axis=0)
    hi = np.quantile(samples, 1 - (1 - hdi_prob) / 2, axis=0)

    periods = pd.PeriodIndex(quarters, freq="Q")
    ts = periods.to_timestamp(how="end")

    rows = []
    for t, q in enumerate(quarters):
        for k, lab in enumerate(labels):
            rows.append({
                "quarter": ts[t],
                "category": lab,
                "median": median[t, k],
                "lower": lo[t, k],
                "upper": hi[t, k],
                "is_forecast": t >= (observed or len(quarters)),
            })
    return pd.DataFrame(rows)

def make_observed_df(y, W, quarters, labels):
    shares = y / W[:, None]
    periods = pd.PeriodIndex(quarters, freq="Q")
    ts = periods.to_timestamp(how="end")
    rows = []
    for t, q in enumerate(quarters):
        for k, lab in enumerate(labels):
            rows.append({
                "quarter": ts[t],
                "category": lab,
                "share": float(shares[t, k]),
            })
    return pd.DataFrame(rows)


def fit_model(y, W, X, draws=2000, tune=1000):
    T, K = y.shape

    with pm.Model() as model:
        pm.MutableData("X", X)

        sigma = pm.HalfNormal("sigma", 0.15)
        phi = pm.Beta("phi", 5, 1.5)

        beta0 = pm.Normal("beta0", 0, 1, shape=K)

        mu_raw = pm.Normal("mu_raw", 0, 0.3, shape=K)
        mu = pm.Deterministic("mu", mu_raw - pm.math.mean(mu_raw))

        P = X.shape[1]
        beta_x = pm.Normal("beta_x", 0, 0.2, shape=(P, K))

        dev0 = pm.Normal("dev0", 0, 0.5, shape=K)
        eps = pm.Normal("eps", 0, sigma, shape=(T - 1, K))
        def step(e_t, prev, phi, mu):
            mean = phi * prev + (1 - phi) * mu
            return mean + e_t

        dev_rest, _ = pytensor.scan(
            fn=step,
            sequences=[eps],
            outputs_info=dev0,
            non_sequences=[phi, mu],
        )

        dev = pt.concatenate([dev0[None, :], dev_rest], axis=0)
        dev = pm.Deterministic("dev", dev)

        x_effect = pm.math.dot(model["X"], beta_x) 
        z = beta0 + dev + x_effect
        z -= z.mean(axis=1, keepdims=True)

        p = pm.Deterministic("p", pm.math.softmax(z, axis=1))
        pm.Multinomial("obs", n=W, p=p, observed=y)

        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=4,
            cores=4,
            target_accept=0.97,
            return_inferencedata=True,
        )

    return trace

def forecast(trace, horizon, gdp_future=None, X_train=None, gdp_std_hist=None, gdp_mean=None, gdp_std=None, lags=DEFAULT_GDP_LAGS, seed=123, sigma_mult=1.0, phi_override=None, sigma_growth=0.0, sigma_growth_mode="linear"):
    rng = np.random.default_rng(seed)

    def stack(v):
        da = trace.posterior[v].stack(sample=("chain", "draw"))
        dims = [d for d in da.dims if d != "sample"]
        da = da.transpose("sample", *dims)
        return da.values

    dev = stack("dev")          # (S, T, K)
    beta0 = stack("beta0")      # (S, K)
    mu = stack("mu")            # (S, K)
    beta_x = stack("beta_x")    # (S, P, K)
    sigma = stack("sigma")      # (S,)
    phi = stack("phi")          # (S,)

    if sigma_mult is not None and float(sigma_mult) != 1.0:
        sigma = sigma * float(sigma_mult)
    if phi_override is not None:
        phi = np.full_like(phi, float(phi_override))

    S, T, K = dev.shape

    run = dev[:, -1, :]                 # (S, K)
    fut = np.zeros((S, horizon, K))     # forecast latent states

    if gdp_future is None:
        gdp_future = np.zeros(horizon)

    if X_train is None or gdp_std_hist is None or gdp_mean is None or gdp_std is None:
        P = beta_x.shape[1]
        X_all = np.zeros((T + horizon, P))
    else:
        X_future = build_future_gdp_lags(gdp_std_hist, gdp_future, gdp_mean, gdp_std, lags)
        X_all = np.vstack([X_train, X_future])

    for h in range(horizon):
        shock = rng.normal(size=(S, K))
        if sigma_growth_mode == "exp":
            growth_factor = (1.0 + float(sigma_growth)) ** (h + 1)
        else:
            growth_factor = 1.0 + float(sigma_growth) * (h + 1)
        shock *= (sigma[:, None] * float(sigma_mult) * growth_factor)

        # AR(1) mean
        mean = phi[:, None] * run + (1 - phi[:, None]) * mu

        # update + store
        run = mean + shock
        fut[:, h, :] = run

    dev_all = np.concatenate([dev, fut], axis=1)  # (S, T+h, K)

    x_all = np.einsum('tp,spk->stk', X_all, beta_x)

    z = beta0[:, None, :] + dev_all + x_all
    z -= z.mean(axis=2, keepdims=True)

    return softmax_np(z, axis=2)

def plot(summary, observed=None, save_path=None, title=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {}
    for cat in summary["category"].unique():
        s = summary[summary["category"] == cat]
        line, = ax.plot(s["quarter"], s["median"], label=cat)
        ax.fill_between(s["quarter"], s["lower"], s["upper"], alpha=0.2)
        colors[cat] = line.get_color()

    if observed is not None:
        for cat in summary["category"].unique():
            if cat in colors:
                o = observed[observed["category"] == cat]
                ax.scatter(o["quarter"], o["share"], s=18, facecolors="none", edgecolors=colors[cat], linewidths=1.0, label=None, zorder=3)

    cutoff = summary.loc[~summary["is_forecast"], "quarter"].max()
    ax.axvline(cutoff, ls="--", c="k", alpha=0.4)

    ax.set_ylim(0, 1)
    ax.set_ylabel("Share")
    ax.set_title(title or "Age composition with GDP-driven forecast")
    ax.legend()
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()

def plot_overlay(summary_base, summary_scn, observed=None, save_path=None, title=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {}
    for cat in summary_base["category"].unique():
        s0 = summary_base[summary_base["category"] == cat]
        s1 = summary_scn[summary_scn["category"] == cat]
        obs_mask = ~s0["is_forecast"].values
        fc_mask = s1["is_forecast"].values
        line_base, = ax.plot(s0.loc[obs_mask, "quarter"], s0.loc[obs_mask, "median"], label=f"{cat} (Past)")
        ax.fill_between(s0.loc[obs_mask, "quarter"], s0.loc[obs_mask, "lower"], s0.loc[obs_mask, "upper"], alpha=0.15)
        ax.plot(s1.loc[fc_mask, "quarter"], s1.loc[fc_mask, "median"], ls="--", label=f"{cat} (Scenario)")
        ax.fill_between(s1.loc[fc_mask, "quarter"], s1.loc[fc_mask, "lower"], s1.loc[fc_mask, "upper"], alpha=0.15)
        colors[cat] = line_base.get_color()

    if observed is not None:
        for cat, color in colors.items():
            o = observed[observed["category"] == cat]
            ax.scatter(o["quarter"], o["share"], s=18, facecolors="none", edgecolors=color, linewidths=1.0, label=None, zorder=3)

    cutoff = summary_base.loc[~summary_base["is_forecast"], "quarter"].max()
    ax.axvline(cutoff, ls="--", c="k", alpha=0.4)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share")
    ax.set_title(title or "Baseline vs GDP scenario forecast")
    ax.legend(ncol=2)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()

def plot_dual_overlay(summary_base, summary_up, summary_down, observed=None, save_path=None, title=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {}
    for cat in summary_base["category"].unique():
        s0 = summary_base[summary_base["category"] == cat]
        sup = summary_up[summary_up["category"] == cat]
        sdn = summary_down[summary_down["category"] == cat]
        obs_mask = ~s0["is_forecast"].values
        fc_mask_up = sup["is_forecast"].values
        fc_mask_down = sdn["is_forecast"].values
        line_base, = ax.plot(s0.loc[obs_mask, "quarter"], s0.loc[obs_mask, "median"], label=f"{cat} (base)")
        ax.plot(sup.loc[fc_mask_up, "quarter"], sup.loc[fc_mask_up, "median"], ls="--", color="tab:green", label=f"{cat} (GDP up)")
        ax.plot(sdn.loc[fc_mask_down, "quarter"], sdn.loc[fc_mask_down, "median"], ls="--", color="tab:red", label=f"{cat} (GDP down)")
        ax.fill_between(sup.loc[fc_mask_up, "quarter"], sup.loc[fc_mask_up, "lower"], sup.loc[fc_mask_up, "upper"], alpha=0.12, color="tab:green")
        ax.fill_between(sdn.loc[fc_mask_down, "quarter"], sdn.loc[fc_mask_down, "lower"], sdn.loc[fc_mask_down, "upper"], alpha=0.12, color="tab:red")
        colors[cat] = line_base.get_color()

    if observed is not None:
        for cat, color in colors.items():
            o = observed[observed["category"] == cat]
            ax.scatter(o["quarter"], o["share"], s=18, facecolors="none", edgecolors=color, linewidths=1.0, label=None, zorder=3)

    cutoff = summary_base.loc[~summary_base["is_forecast"], "quarter"].max()
    ax.axvline(cutoff, ls="--", c="k", alpha=0.4)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share")
    ax.set_title(title or "Baseline vs GDP up/down scenarios")
    ax.legend(ncol=2)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()

def summarize_delta(p_base_samples, p_scn_samples, quarters, labels, hdi_prob=0.9, observed=None):
    diff = p_scn_samples - p_base_samples  # (S, T+h, K)
    return summarize_samples(diff, quarters, labels, hdi_prob=hdi_prob, observed=observed)

def impact_delta(summary_base, summary_other):
    base_f = summary_base[summary_base["is_forecast"]].set_index(["quarter", "category"]) [["median"]]
    oth_f = summary_other[summary_other["is_forecast"]].set_index(["quarter", "category"]) [["median"]]
    out = oth_f.copy()
    out["delta"] = out["median"].values - base_f["median"].values
    return out

def plot_delta(summary_delta, save_path=None, title=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    for cat in summary_delta["category"].unique():
        s = summary_delta[summary_delta["category"] == cat]
        ax.plot(s["quarter"], s["median"], label=f"{cat} Δ")
        ax.fill_between(s["quarter"], s["lower"], s["upper"], alpha=0.15)
    cutoff = summary_delta.loc[~summary_delta["is_forecast"], "quarter"].max()
    ax.axvline(cutoff, ls="--", c="k", alpha=0.4)
    ax.axhline(0.0, ls=":", c="gray", alpha=0.6)
    ax.set_ylabel("Scenario − Baseline (share)")
    ax.set_title(title or "Forecast deltas by category")
    ax.legend(ncol=2)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close()
        
    else:
        plt.show()

def run_for_sex(sex, df_train, args, gdp_scn, gdp_up, gdp_down):
    print(f"\n=== Sex: {sex} ===")
    y, W, quarters = extract_category(df_train, AGE_COLS, sex)
    X, gdp_std_hist, gdp_mean, gdp_std = extract_gdp_lags(df_train, sex, L=args.gdp_lags)

    try:
        scenario_offset = compute_scenario_offset(quarters, args.scenario_at)
    except Exception as e:
        raise SystemExit(
            f"Invalid --scenario-at '{args.scenario_at}'. Use format like 2030Q1. "
            f"Last observed quarter is '{quarters[-1]}'. Error: {e}"
        )

    cache_base = Path(args.cache)
    cache = cache_base.parent / (cache_base.stem + f"_{sex.lower()}" + cache_base.suffix)
    if cache.exists() and not args.refit:
        trace = az.from_netcdf(cache)
        print("Loaded cached model")
        if "beta_x" not in trace.posterior.data_vars:
            raise SystemExit("Cached model incompatible (missing beta_x). Re-run with --refit to fit the distributed-lag GDP model.")
        bx_shape = trace.posterior["beta_x"].values.shape  # (chains, draws, P, K)
        P_cached = bx_shape[-2]
        P_current = X.shape[1]
        if P_current != P_cached:
            L_cached = P_cached - 1
            raise SystemExit(
                f"Cached model was fit with gdp-lags={L_cached} (P={P_cached}), "
                f"but current --gdp-lags={args.gdp_lags} produces P={P_current}. "
                "Re-run with --refit to use the requested lags, or set --gdp-lags "
                f"{L_cached} to reuse the cache."
            )
        # Ensure cached model's time dimension matches current training window
        dev_shape = trace.posterior["dev"].values.shape  # (chains, draws, T_cached, K)
        T_cached = dev_shape[-2]
        T_current = X.shape[0]
        if T_current != T_cached:
            raise SystemExit(
                f"Cached model was fit with {T_cached} quarters, but current training window "
                f"produces {T_current}. Re-run with --refit to change the training window, or "
                "remove --train-start/--train-end to reuse the cache."
            )
    else:
        print("Fitting model...")
        trace = fit_model(y, W, X)
        cache.parent.mkdir(exist_ok=True)
        az.to_netcdf(trace, cache)

    print(az.summary(trace, var_names=["phi", "sigma", "beta_x"]))
    if args.loo:
        try:
            loo = az.loo(trace)
            print("\nPSIS-LOO:", loo)
        except Exception as e:
            print("Warning: could not compute LOO:", e)

    gdp_base = np.zeros(args.horizon)  
    p_base = forecast(trace, args.horizon, gdp_base, X_train=X, gdp_std_hist=gdp_std_hist, gdp_mean=gdp_mean, gdp_std=gdp_std, lags=args.gdp_lags)

    # Apply offset to scenarios
    gdp_scn_off = offset_scenario_vector(gdp_scn, scenario_offset, args.horizon) if gdp_scn is not None else None
    gdp_up_off = offset_scenario_vector(gdp_up, scenario_offset, args.horizon) if gdp_up is not None else None
    gdp_down_off = offset_scenario_vector(gdp_down, scenario_offset, args.horizon) if gdp_down is not None else None

    p_scn = forecast(trace, args.horizon, gdp_scn_off, X_train=X, gdp_std_hist=gdp_std_hist, gdp_mean=gdp_mean, gdp_std=gdp_std, lags=args.gdp_lags, sigma_mult=args.forecast_sigma_mult, phi_override=args.forecast_phi, sigma_growth=args.forecast_sigma_growth, sigma_growth_mode=args.forecast_sigma_growth_mode) if gdp_scn_off is not None else None
    p_up = forecast(trace, args.horizon, gdp_up_off, X_train=X, gdp_std_hist=gdp_std_hist, gdp_mean=gdp_mean, gdp_std=gdp_std, lags=args.gdp_lags, sigma_mult=args.forecast_sigma_mult, phi_override=args.forecast_phi, sigma_growth=args.forecast_sigma_growth, sigma_growth_mode=args.forecast_sigma_growth_mode) if gdp_up_off is not None else None
    p_down = forecast(trace, args.horizon, gdp_down_off, X_train=X, gdp_std_hist=gdp_std_hist, gdp_mean=gdp_mean, gdp_std=gdp_std, lags=args.gdp_lags, sigma_mult=args.forecast_sigma_mult, phi_override=args.forecast_phi, sigma_growth=args.forecast_sigma_growth, sigma_growth_mode=args.forecast_sigma_growth_mode) if gdp_down_off is not None else None

    future_q = [str(pd.Period(quarters[-1], freq="Q") + i + 1) for i in range(args.horizon)]
    all_q = list(quarters) + future_q

    summary_base = summarize_samples(p_base, all_q, AGE_COLS, observed=len(quarters))
    observed_df = make_observed_df(y, W, quarters, AGE_COLS)
    plots_dir = Path(args.plots_dir) / sex.lower() if args.save_plots else None

    title_text = f"{sex} age composition forecast"
    if p_up is not None and p_down is not None:
        summary_up = summarize_samples(p_up, all_q, AGE_COLS, observed=len(quarters))
        summary_down = summarize_samples(p_down, all_q, AGE_COLS, observed=len(quarters))
        print("\nGDP up scenario impact preview (median deltas):\n", impact_delta(summary_base, summary_up).head(8))
        print("\nGDP down scenario impact preview (median deltas):\n", impact_delta(summary_base, summary_down).head(8))

        plot_dual_overlay(
            summary_base, summary_up, summary_down,
            observed=observed_df,
            save_path=(plots_dir / "baseline_vs_dual.pdf") if plots_dir else None,
            title=title_text,
        )

        delta_up = summarize_delta(p_base, p_up, all_q, AGE_COLS, observed=len(quarters))
        delta_down = summarize_delta(p_base, p_down, all_q, AGE_COLS, observed=len(quarters))
        print("\nFinal quarter deltas (up/down):")
        print(delta_up[delta_up["is_forecast"]].groupby("category").tail(1)[["category","median"]])
        print(delta_down[delta_down["is_forecast"]].groupby("category").tail(1)[["category","median"]])
        plot_delta(delta_up, save_path=(plots_dir / "delta_up.pdf") if plots_dir else None, title=title_text)
        plot_delta(delta_down, save_path=(plots_dir / "delta_down.pdf") if plots_dir else None, title=title_text)
    elif p_scn is None:
        plot(summary_base, observed=observed_df, save_path=(plots_dir / "baseline.pdf") if plots_dir else None, title=title_text)
    else:
        summary_scn = summarize_samples(p_scn, all_q, AGE_COLS, observed=len(quarters))
        print("\nScenario impact preview (median deltas):\n", impact_delta(summary_base, summary_scn).head(8))
        plot_overlay(summary_base, summary_scn, observed=observed_df, save_path=(plots_dir / "overlay.pdf") if plots_dir else None, title=title_text)

        delta_summary = summarize_delta(p_base, p_scn, all_q, AGE_COLS, observed=len(quarters))
        plot_delta(delta_summary, save_path=(plots_dir / "delta.pdf") if plots_dir else None, title=title_text)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sex", type=str, choices=["Male", "Female", "Both"], default="Male", help="Sex to model; use 'Both' to run for Male and Female.")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--cache", default=".cache/age_gdp.nc")
    parser.add_argument("--refit", action="store_true")
    parser.add_argument("--gdp-lags", type=int, default=DEFAULT_GDP_LAGS)
    parser.add_argument("--scenario-file", type=str, default=None, help="CSV with future GDP path; column name must match 'GDP growth'.")
    parser.add_argument("--scenario-constant", type=float, default=None, help="Constant GDP growth applied across horizon (raw units).")
    parser.add_argument("--shock-len", type=int, default=0, help="Length of initial GDP shock (quarters) for --scenario-constant.")
    parser.add_argument("--scenario-ramp-start", type=float, default=None, help="Ramp start value.")
    parser.add_argument("--scenario-ramp-end", type=float, default=None, help="Ramp end value.")
    parser.add_argument("--two-sides-constant", type=float, default=None, help="Generate both +value and -value constant scenarios (optionally with --shock-len).")
    parser.add_argument("--scenario-profile", type=str, default=None, help="Piecewise GDP path 'value:len;value:len;...' (raw units).")
    parser.add_argument("--scenario-at", type=str, default=None, help="Quarter when scenario starts (e.g., 2030Q1). Pre-fill zeros until then.")
    parser.add_argument("--train-start", type=str, default=None, help="Quarter to start training window (e.g., 2020Q1)")
    parser.add_argument("--train-end", type=str, default=None, help="Quarter to end training window (e.g., 2021Q4)")
    parser.add_argument("--loo", action="store_true", help="Print PSIS-LOO information criterion for the fitted model")
    parser.add_argument("--save-plots", action="store_true", help="Save figures instead of showing interactively.")
    parser.add_argument("--plots-dir", type=str, default="figures", help="Directory to save plots when --save-plots is used.")
    parser.add_argument("--forecast-sigma-mult", type=float, default=1.0, help="Multiply posterior sigma during forecasting to increase volatility (e.g., 1.5–3.0).")
    parser.add_argument("--forecast-phi", type=float, default=None, help="Override phi during forecasting (0–1). Lower values reduce persistence to counter flatness.")
    parser.add_argument("--forecast-sigma-growth", type=float, default=0.0, help="Linear growth rate for forecast innovations per quarter (e.g., 0.02–0.10). 0.0 keeps bands flat.")
    parser.add_argument("--forecast-sigma-growth-mode", type=str, choices=["linear","exp"], default="linear", help="Uncertainty growth mode across horizon: linear or exponential.")
    args = parser.parse_args()
    
    df = pd.read_csv(CSV_PATH)
    df["Quarter"] = df["Quarter"].astype(str)

    if args.train_start or args.train_end:
        pi = pd.PeriodIndex(df["Quarter"], freq="Q")
        mask = np.ones(len(df), dtype=bool)
        if args.train_start:
            start_p = pd.Period(args.train_start, freq="Q")
            mask &= pi >= start_p
        if args.train_end:
            end_p = pd.Period(args.train_end, freq="Q")
            mask &= pi <= end_p
        df_train = df.loc[mask].copy()
    else:
        df_train = df

    gdp_scn = None
    gdp_up = None
    gdp_down = None
    if args.two_sides_constant is not None:
        val = float(args.two_sides_constant)
        if (args.shock_len or 0) > 0:
            gdp_up = make_shock_scenario(+val, args.shock_len, args.horizon)
            gdp_down = make_shock_scenario(-val, args.shock_len, args.horizon)
        else:
            gdp_up = make_constant_scenario(+val, args.horizon)
            gdp_down = make_constant_scenario(-val, args.horizon)
    else:
        if args.scenario_profile:
            gdp_scn = make_piecewise_scenario(args.scenario_profile, args.horizon)
        elif args.scenario_file:
            gdp_scn = load_scenario_file(Path(args.scenario_file), args.horizon)
        elif args.scenario_constant is not None and (args.shock_len or 0) > 0:
            gdp_scn = make_shock_scenario(args.scenario_constant, args.shock_len, args.horizon)
        elif args.scenario_constant is not None:
            gdp_scn = make_constant_scenario(args.scenario_constant, args.horizon)
        elif args.scenario_ramp_start is not None and args.scenario_ramp_end is not None:
            gdp_scn = make_ramp_scenario(args.scenario_ramp_start, args.scenario_ramp_end, args.horizon)

    sexes = [args.sex] if args.sex != "Both" else ["Male", "Female"]
    for sex in sexes:
        run_for_sex(sex, df_train, args, gdp_scn, gdp_up, gdp_down)

if __name__ == "__main__":
    main()
