# =========================
# 4) Coupled CP -> conditional POINT forecast, then JOINT bands propagated ONLY from sex bands
#    Plot: P(sex, factor) with left=Male right=Female
# =========================

OUT_JOINT_AGE  = os.path.join(OUTDIR, "pred_joint_sex_age_with_band.csv")
OUT_JOINT_EDU  = os.path.join(OUTDIR, "pred_joint_sex_edu_with_band.csv")
OUT_JOINT_BORN = os.path.join(OUTDIR, "pred_joint_sex_born_with_band.csv")

FORECAST_END = "2035Q4"
p_end = parse_q(FORECAST_END)
p_all = list(pd.period_range(p_hist[0], p_end, freq="Q"))
hist_len = len(p_hist)
steps_ahead = len(p_all) - hist_len
if steps_ahead <= 0:
    raise ValueError("FORECAST_END is not after the end of historical periods (p_hist).")

ensure_dir(OUTDIR)

# ---------- helpers ----------
def ensure_period_col(df_, time_col):
    if "__p__" in df_.columns:
        return
    q = df_[time_col].astype(str).str.strip()
    q = q.str.replace(r"(\d{4})\s*Q([1-4])", r"\1Q\2", regex=True)
    q = q.str.replace(" ", "", regex=False)
    df_["__p__"] = q.map(parse_q)

def _aggregate_counts(df_, cols, total_col, sex_col):
    g = df_.groupby(["__p__", sex_col], as_index=False)[cols + [total_col]].sum()
    for c in cols + [total_col]:
        g[c] = pd.to_numeric(g[c], errors="coerce")
    return g

def conditional_from_counts(df_, cols, periods, sex_levels, sex_col, total_col):
    """X(t,s,k)=N(s,k)/N(s), sums to 1 across k within each sex."""
    idx = pd.Index(periods, name="__p__")
    T, S, K = len(periods), len(sex_levels), len(cols)
    X = np.zeros((T, S, K), float)

    g = _aggregate_counts(df_, cols, total_col, sex_col)

    for j, s in enumerate(sex_levels):
        sub = g[g[sex_col] == s].set_index("__p__").reindex(idx)

        num = sub[cols].interpolate("linear", limit_direction="both").ffill().bfill().fillna(0.0).to_numpy(float)
        den = sub[total_col].interpolate("linear", limit_direction="both").ffill().bfill().fillna(0.0).to_numpy(float)

        den = np.where(den <= 0.0, np.nan, den)
        cond = num / den[:, None]
        cond = np.nan_to_num(cond, nan=0.0, posinf=0.0, neginf=0.0)

        rs = cond.sum(axis=1, keepdims=True)
        rs[rs <= 0.0] = 1.0
        X[:, j, :] = cond / rs

    return X

def reconstruct_conditional_cp(U, V, W):
    """CP reconstruction then normalise across categories => conditional."""
    X = np.einsum("tr,sr,kr->tsk", U, V, W)
    X = np.clip(X, 0.0, None)
    d = X.sum(axis=2, keepdims=True)
    d[d <= 0.0] = 1.0
    return X / d

def wide_from_joint(p_all, sex_levels, cats, J_mean, J_p05, J_p95, out_csv):
    def clean(s):
        return str(s).replace(" ", "_").replace(":", "")
    rows = []
    for i, p in enumerate(p_all):
        row = {"Quarter": qlabel(p)}
        for j, s in enumerate(sex_levels):
            for k, cat in enumerate(cats):
                base = f"{clean(s)}_{clean(cat)}"
                row[f"{base}_mean"] = float(J_mean[i, j, k])
                row[f"{base}_p05"]  = float(J_p05[i, j, k])
                row[f"{base}_p95"]  = float(J_p95[i, j, k])
        rows.append(row)
    wide = pd.DataFrame(rows)
    wide.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
    return wide

def plot_joint_two_panel(
    wide_df, cats, hist_len, title,
    male_label, female_label,
    train_end_p, val_end_p, mark_years, tick_step_years,
    band_alpha=BAND_ALPHA, outdir=OUTDIR
):
    def clean(s):
        return str(s).replace(" ", "_").replace(":", "")

    periods = [parse_q(q) for q in wide_df["Quarter"]]
    x = np.arange(len(periods))

    TITLE_FZ = 18
    AXIS_FZ = 14
    TICK_FZ = 12
    LEGEND_FZ = 10

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for ax, sex in zip(axes, [male_label, female_label]):  # left male, right female
        for cat in cats:
            base = f"{clean(sex)}_{clean(cat)}"
            mcol, lcol, ucol = f"{base}_mean", f"{base}_p05", f"{base}_p95"
            if mcol not in wide_df.columns:
                continue

            y  = wide_df[mcol].to_numpy(float)
            lo = wide_df[lcol].to_numpy(float)
            hi = wide_df[ucol].to_numpy(float)

            line, = ax.plot(x[:hist_len], y[:hist_len], "-", lw=1.6, label=str(cat))
            c = line.get_color()
            ax.plot(x[hist_len-1:], y[hist_len-1:], "--", lw=1.6, color=c)

            # band (forecast only) — comes ONLY from sex band propagation
            ax.fill_between(x[hist_len:], lo[hist_len:], hi[hist_len:], alpha=band_alpha, color=c)

        if train_end_p in periods:
            ax.axvline(periods.index(train_end_p), color="k", linestyle="--", lw=1)
        if val_end_p in periods:
            ax.axvline(periods.index(val_end_p), color="k", linestyle="-.", lw=1)

        for ymark in mark_years:
            q = pd.Period(f"{ymark}Q1", freq="Q")
            if q in periods:
                ax.axvline(periods.index(q), linestyle=":", lw=1)

        pos, lab = year_ticks(periods, step=tick_step_years)
        ax.set_xticks(pos)
        ax.set_xticklabels(lab, fontsize=TICK_FZ)
        ax.tick_params(axis="y", labelsize=TICK_FZ)

        ax.set_title(sex, fontsize=TITLE_FZ)
        ax.set_xlabel("Year", fontsize=AXIS_FZ)
        ax.legend(fontsize=LEGEND_FZ, loc="upper left", frameon=True)
        ax.grid(False)

    axes[0].set_ylabel("Share", fontsize=AXIS_FZ)
    fig.suptitle(title, fontsize=TITLE_FZ)
    plt.tight_layout()

    fname = re.sub(r"[^\w\-]+", "_", title.lower()).strip("_")
    save_path = os.path.join(outdir, f"{fname}.pdf")
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    print(f"Saved figure: {save_path}")

    plt.show()
    plt.close(fig)

# ---------- run ----------
ensure_period_col(df, TIME_COL)

AGE_USE  = [c for c in AGE_COLS  if c in df.columns]
EDU_USE  = [c for c in EDU_COLS  if c in df.columns]
BORN_USE = [c for c in BORN_COLS if c in df.columns]

if TOTAL_AGE_COL not in df.columns:
    raise ValueError(f"Missing total column: {TOTAL_AGE_COL}")

# (A) fit CP on historical CONDITIONAL shares (as before)
X_age_hist  = conditional_from_counts(df, AGE_USE,  p_hist, sex_levels, SEX_COL, TOTAL_AGE_COL)
X_edu_hist  = conditional_from_counts(df, EDU_USE,  p_hist, sex_levels, SEX_COL, TOTAL_AGE_COL)
X_born_hist = conditional_from_counts(df, BORN_USE, p_hist, sex_levels, SEX_COL, TOTAL_AGE_COL)

U_hist, V_sex, (W_age, W_edu, W_born) = coupled_cp_als(
    [X_age_hist, X_edu_hist, X_born_hist],
    rank=RANK, iters=ALS_ITERS, seed=SEED
)

# (B) forecast latent time factors ONLY as point forecast (no MC here)
#     reuse your chosen SARIMAX config from sex section; DO NOT change sex logic
U_future = np.zeros((steps_ahead, RANK), float)
for r in range(RANK):
    res_u = fit_sarimax(np.asarray(U_hist[:, r], float), best_order, best_seas, best_trend)
    if res_u is None:
        a, b = np.polyfit(np.arange(len(U_hist)), U_hist[:, r], 1)
        U_future[:, r] = a * np.arange(len(U_hist), len(U_hist) + steps_ahead) + b
    else:
        U_future[:, r] = np.asarray(res_u.forecast(steps=steps_ahead), float)

U_all = np.vstack([U_hist, U_future])

# (C) conditional point forecasts from CP
X_age_all  = reconstruct_conditional_cp(U_all, V_sex, W_age)
X_edu_all  = reconstruct_conditional_cp(U_all, V_sex, W_edu)
X_born_all = reconstruct_conditional_cp(U_all, V_sex, W_born)

# (D) build JOINT mean/p05/p95 by propagating ONLY sex uncertainty
sex_mean = np.stack([male_mean,  female_mean], axis=1)   # (T_all,2)
sex_p05  = np.stack([male_p05,   female_p05],  axis=1)
sex_p95  = np.stack([male_p95,   female_p95],  axis=1)

def joint_from_sex_band_only(X_cond, sex_mean, sex_p05, sex_p95):
    J_mean = X_cond * sex_mean[:, :, None]
    J_p05  = X_cond * sex_p05[:,  :, None]
    J_p95  = X_cond * sex_p95[:,  :, None]
    return J_mean, J_p05, J_p95

J_age_mean,  J_age_p05,  J_age_p95  = joint_from_sex_band_only(X_age_all,  sex_mean, sex_p05, sex_p95)
J_edu_mean,  J_edu_p05,  J_edu_p95  = joint_from_sex_band_only(X_edu_all,  sex_mean, sex_p05, sex_p95)
J_born_mean, J_born_p05, J_born_p95 = joint_from_sex_band_only(X_born_all, sex_mean, sex_p05, sex_p95)

# (E) save + plot (left male right female, band per subgroup)
wide_age  = wide_from_joint(p_all, sex_levels, AGE_USE,  J_age_mean,  J_age_p05,  J_age_p95,  OUT_JOINT_AGE)
wide_edu  = wide_from_joint(p_all, sex_levels, EDU_USE,  J_edu_mean,  J_edu_p05,  J_edu_p95,  OUT_JOINT_EDU)
wide_born = wide_from_joint(p_all, sex_levels, BORN_USE, J_born_mean, J_born_p05, J_born_p95, OUT_JOINT_BORN)

plot_joint_two_panel(
    wide_age, AGE_USE, hist_len,
    "Age composition by sex (joint) (historical solid, forecast dashed)",
    male_label, female_label,
    train_end_p, val_end_p, MARK_YEARS, TICK_STEP_YEARS,
    band_alpha=BAND_ALPHA, outdir=OUTDIR
)

plot_joint_two_panel(
    wide_edu, EDU_USE, hist_len,
    "Education composition by sex (joint) (historical solid, forecast dashed)",
    male_label, female_label,
    train_end_p, val_end_p, MARK_YEARS, TICK_STEP_YEARS,
    band_alpha=BAND_ALPHA, outdir=OUTDIR
)

plot_joint_two_panel(
    wide_born, BORN_USE, hist_len,
    "Place-of-birth composition by sex (joint) (historical solid, forecast dashed)",
    male_label, female_label,
    train_end_p, val_end_p, MARK_YEARS, TICK_STEP_YEARS,
    band_alpha=BAND_ALPHA, outdir=OUTDIR
)

print("\nSection 4 (JOINT, sex-band-only propagation) done.")
print("CSVs:", OUT_JOINT_AGE, OUT_JOINT_EDU, OUT_JOINT_BORN)
print("PDFs saved to:", OUTDIR)
