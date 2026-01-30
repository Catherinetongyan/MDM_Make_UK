"""
UK Manufacturing Workforce VECM Forecast Model
===============================================
University of Bristol - Engineering Mathematics Coursework

Vector Error Correction Model forecasting UK manufacturing employment to 2029 Q4
under three policy uncertainty scenarios.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import matplotlib as mpl

mpl.rcParams.update({
    'figure.figsize': (7, 3),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'lines.linewidth': 2.5,
    'lines.markersize': 7,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8
})

warnings.filterwarnings('ignore')

from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson

# ============================================================
# CONFIGURATION
# ============================================================

ENDOG_VARS = ['log_workforce_jobs_k', 'log_output_per_job', 'log_gva_low_level']
EXOG_VARS = ['policy_uncertainty', 'usd_gbp', 'eur_gbp']

TRAINING_WINDOWS = [
    ('Full (1999-2025)', '1999-03-31'),
    ('Post-GFC (2010-2025)', '2010-03-31'),
    ('Post-Brexit (2017-2025)', '2017-03-31'),
]

FORECAST_HORIZON = 17  # quarters to 2029 Q4


# ============================================================
# DATA
# ============================================================

def load_data(filepath):
    df = pd.read_csv(filepath, index_col='date', parse_dates=True)
    df = df.loc['1999-03-31':].copy().interpolate(method='linear')

    for col in ['workforce_jobs_k', 'output_per_job', 'gva_low_level']:
        df[f'log_{col}'] = np.log(df[col])

    return df


def compute_scenarios(data):
    pu = data['policy_uncertainty']
    return {
        'High uncertainty (P90)': {
            'policy_uncertainty': pu.quantile(0.90),
            'usd_gbp': 1.20,
            'eur_gbp': 1.15,
        },
        'Central (P50)': {
            'policy_uncertainty': pu.quantile(0.50),
            'usd_gbp': 1.25,
            'eur_gbp': 1.18,
        },
        'Stability (P10)': {
            'policy_uncertainty': pu.quantile(0.10),
            'usd_gbp': 1.35,
            'eur_gbp': 1.22,
        },
    }


# ============================================================
# MODEL
# ============================================================

def run_johansen_test(data):
    result = coint_johansen(data[ENDOG_VARS], det_order=0, k_ar_diff=2)

    print("\nJohansen Cointegration Test")
    print("-" * 45)
    print(f"{'Null':<10} {'Trace Stat':>12} {'5% CV':>10} {'Result':>12}")
    print("-" * 45)

    for i, (stat, cv) in enumerate(zip(result.lr1, result.cvt[:, 1])):
        res = "Reject" if stat > cv else "Fail to reject"
        print(f"r ≤ {i:<6} {stat:>12.2f} {cv:>10.2f} {res:>12}")

    rank = sum(result.lr1 > result.cvt[:, 1])
    return max(1, rank), result.lr1[0], result.cvt[0, 1]


def get_diagnostics(fit):
    resid = fit.resid[:, 0]
    dw = durbin_watson(resid)
    lb = acorr_ljungbox(resid, lags=[4], return_df=True)
    return {
        'durbin_watson': dw,
        'ljung_box_stat': lb.iloc[0]['lb_stat'],
        'ljung_box_p': lb.iloc[0]['lb_pvalue'],
    }


def generate_ensemble_forecast(data, scenarios, n_periods):
    results = {name: {'forecasts': []} for name in scenarios}

    for window_name, start_date in TRAINING_WINDOWS:
        subset = data.loc[start_date:]

        vecm = VECM(subset[ENDOG_VARS], k_ar_diff=2, coint_rank=1,
                    exog=subset[EXOG_VARS], deterministic='ci')
        fit = vecm.fit()

        for scenario_name, params in scenarios.items():
            exog_fc = np.array([[
                params['policy_uncertainty'],
                params['usd_gbp'],
                params['eur_gbp']
            ]] * n_periods)

            fc_log = fit.predict(steps=n_periods, exog_fc=exog_fc)[:, 0]
            results[scenario_name]['forecasts'].append(np.exp(fc_log))

    for name in results:
        arr = np.array(results[name]['forecasts'])
        results[name]['ensemble'] = arr.mean(axis=0)
        results[name]['std'] = arr.std(axis=0)

    return results


# ============================================================
# VISUALISATION
# ============================================================

def create_forecast_plot(data, results, scenarios, forecast_dates, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    current = data['workforce_jobs_k'].iloc[-1]
    last_date = data.index[-1]

    colors = {
        'High uncertainty (P90)': '#e74c3c',
        'Central (P50)': '#27ae60',
        'Stability (P10)': '#3498db'
    }

    # Left panel: time series
    ax1 = axes[0]
    ax1.plot(data.index, data['workforce_jobs_k'], 'b-', lw=2, label='Historical')

    for name, result in results.items():
        for fc in result['forecasts']:
            ax1.plot(forecast_dates, fc, color=colors[name], alpha=0.2, lw=1)
        ax1.plot(forecast_dates, result['ensemble'], color=colors[name], lw=2.5, label=name)

    ax1.axvline(x=last_date, color='gray', ls=':', alpha=0.7)
    ax1.set_title('UK Manufacturing Employment Forecast', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Jobs (thousands)')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([pd.Timestamp('2015-01-01'), forecast_dates[-1] + pd.DateOffset(months=3)])
    ax1.set_ylim([2300, 2850])

    # Right panel: bar chart
    ax2 = axes[1]
    labels = ['Current\n(2025 Q3)'] + [f'{name.split()[0]}\n({name.split()[-1][1:-1]})' for name in scenarios]
    values = [current] + [results[name]['ensemble'][-1] for name in scenarios]
    errs = [0] + [results[name]['std'][-1] for name in scenarios]
    bar_colors = ['#7f8c8d'] + [colors[name] for name in scenarios]

    bars = ax2.bar(labels, values, color=bar_colors, edgecolor='black', lw=1)
    ax2.errorbar(labels[1:], values[1:], yerr=errs[1:], fmt='none', color='black', capsize=4)

    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                 f'{val:.0f}k', ha='center', fontsize=10, fontweight='bold')

    ax2.set_ylabel('Jobs (thousands)')
    ax2.set_title('2029 Q4 Projections by Scenario', fontsize=11, fontweight='bold')
    ax2.set_ylim([2200, 2850])
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("UK MANUFACTURING VECM FORECAST")
    print("=" * 60)

    # Load data
    data = load_data('uk_manufacturing_clean.csv')
    current = data['workforce_jobs_k'].iloc[-1]
    last_date = data.index[-1]
    print(f"\nData: {len(data)} quarterly observations (1999 Q1 - 2025 Q3)")
    print(f"Current employment: {current:.0f}k")

    # Cointegration test
    coint_rank, trace_stat, crit_val = run_johansen_test(data)
    print(f"\nUsing cointegration rank: {coint_rank}")

    # Fit model on full sample for diagnostics
    vecm_full = VECM(data[ENDOG_VARS], k_ar_diff=2, coint_rank=1,
                     exog=data[EXOG_VARS], deterministic='ci')
    fit_full = vecm_full.fit()

    # Diagnostics
    diag = get_diagnostics(fit_full)
    print(f"\nResidual Diagnostics (jobs equation):")
    print(f"  Durbin-Watson:     {diag['durbin_watson']:.3f}")
    print(f"  Ljung-Box Q(4):    {diag['ljung_box_stat']:.2f} (p = {diag['ljung_box_p']:.3f})")

    # Key coefficients
    gamma = fit_full.gamma[0, :]
    print(f"\nExogenous Coefficients:")
    print(f"  Policy uncertainty: {gamma[0]:.6f}")
    print(f"  USD/GBP:            {gamma[1]:.4f}")
    print(f"  EUR/GBP:            {gamma[2]:.4f}")

    # Scenarios
    scenarios = compute_scenarios(data)
    print(f"\nScenario Definitions:")
    for name, params in scenarios.items():
        print(f"  {name}: PU = {params['policy_uncertainty']:.0f}")

    # Forecast
    forecast_dates = pd.date_range(
        start=last_date + pd.offsets.QuarterEnd(1),
        periods=FORECAST_HORIZON,
        freq='QE'
    )

    print(f"\nGenerating ensemble forecast (3 training windows)...")
    results = generate_ensemble_forecast(data, scenarios, FORECAST_HORIZON)

    # Results
    print("\n" + "=" * 60)
    print("RESULTS: 2029 Q4")
    print("=" * 60)
    print(f"\nCurrent: {current:.0f}k\n")
    print(f"{'Scenario':<25} {'Forecast':>10} {'Change':>10} {'Std':>10}")
    print("-" * 57)

    for name in scenarios:
        fc = results[name]['ensemble'][-1]
        std = results[name]['std'][-1]
        change = (fc / current - 1) * 100
        print(f"{name:<25} {fc:>9.0f}k {change:>+9.1f}% {std:>9.0f}k")

    high = results['High uncertainty (P90)']['ensemble'][-1]
    stab = results['Stability (P10)']['ensemble'][-1]
    print(f"\nScenario spread: {stab - high:.0f}k")

    # Plot
    create_forecast_plot(data, results, scenarios, forecast_dates,
                         'uk_manufacturing_forecast.png')

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
