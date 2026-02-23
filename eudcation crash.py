import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

plt.rcParams.update({
    'font.size': 16,          # Main text size (was 10)
    'axes.titlesize': 18,     # Title size
    'axes.labelsize': 16,     # Axis labels
    'xtick.labelsize': 14,    # Numbers on X axis
    'ytick.labelsize': 14,    # Numbers on Y axis
    'legend.fontsize': 14,    # Legend text
    'figure.figsize': (12, 8) # Makes the graph canvas bigger
})

# LOAD & PREP
master_df = pd.read_csv('master annual data.csv')
master_df = master_df[master_df['Year'] >= 1995]
master_df = master_df.fillna(0)

# Identify columns
edu_cols = [c for c in master_df.columns if 'Aggregate levels' in c]

# Calculate Shares
master_df['Total_Edu_Emp'] = master_df[edu_cols].sum(axis=1)
yearly_data = master_df.groupby('Year').sum(numeric_only=True)

for col in edu_cols:
    yearly_data[f'Pct_{col}'] = (yearly_data[col] / yearly_data['Total_Edu_Emp']) * 100

# Crisis Dummy
def is_crisis_year(year):
    return 1 if year in [2008, 2009, 2020, 2021] else 0

yearly_data['Crisis_Dummy'] = [is_crisis_year(y) for y in yearly_data.index]

# PREDICT WITH CRASH
def predict_future_crash(target_col, df, future_years):
    clean_data = df[['Crisis_Dummy', target_col]].dropna()
    if len(clean_data) < 5: return {y: 0 for y in future_years}

    X = pd.DataFrame({'Year': clean_data.index, 'Crisis_Dummy': clean_data['Crisis_Dummy']})
    y = clean_data[target_col]

    model = LinearRegression()
    model.fit(X, y)

    preds = {}
    for year in future_years:
        # CRASH TRIGGER: Force dummy=1 in 2030
        is_shock = 1 if year == 2030 else 0
        input_data = pd.DataFrame({'Year': [year], 'Crisis_Dummy': [is_shock]})
        preds[year] = max(0, model.predict(input_data)[0])
    return preds

# RUN
target_years = [2026, 2030, 2035]
results_crash = []
cols_to_predict = [f'Pct_{c}' for c in edu_cols]

for col in cols_to_predict:
    preds = predict_future_crash(col, yearly_data, target_years)
    row = {'Variable': col}; row.update(preds)
    results_crash.append(row)

results_crash_df = pd.DataFrame(results_crash)

# NORMALIZE
def normalize_group(df, keyword, year):
    mask = df['Variable'].str.contains(keyword)
    total = df.loc[mask, year].sum()
    if total > 0:
        df.loc[mask, year] = (df.loc[mask, year] / total) * 100

for year in target_years:
    normalize_group(results_crash_df, 'Aggregate levels', year)

# PLOT
def get_plot_data(col_name):
    history = yearly_data[col_name].dropna()
    future_row = results_crash_df[results_crash_df['Variable'] == col_name].iloc[0]
    future = pd.Series([future_row[y] for y in target_years], index=target_years)
    return pd.concat([history, future]).sort_index()

plt.figure(figsize=(10, 6))
labels = [c.replace('Pct_Aggregate levels: ', '') for c in cols_to_predict]

for i, col in enumerate(cols_to_predict):
    if col in results_crash_df['Variable'].values:
        data = get_plot_data(col)
        plt.plot(data.index, data.values, marker='o', label=labels[i])

plt.axvline(x=2024, color='gray', linestyle='--', label='Forecast Start')
plt.axvline(x=2030, color='red', linestyle=':', label='Simulated Crash (2030)')
plt.title("Projected Education Levels (Scenario: 2030 Shock)")
plt.xlabel("Year")
plt.ylabel("% of Workforce")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('education_trend_crash.png')
print("Saved education_trend_crash.png")
plt.show()