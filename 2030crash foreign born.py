import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# LOAD & PREP DATA (Same as Baseline)
master_df = pd.read_csv('master annual data.csv')
master_df = master_df[master_df['Year'] >= 1995]
master_df = master_df.fillna(0)

birth_cols = [c for c in master_df.columns if 'Place of birth' in c]
age_cols = [c for c in master_df.columns if 'Aggregate bands' in c]

master_df['Total_Emp'] = master_df[age_cols].sum(axis=1)
yearly_data = master_df.groupby('Year').sum(numeric_only=True)

for col in birth_cols:
    yearly_data[f'Pct_{col}'] = (yearly_data[col] / yearly_data['Total_Emp']) * 100


# CRISIS DUMMY
def is_crisis_year(year):
    if year in [2008, 2009, 2020, 2021]:
        return 1
    return 0


yearly_data['Crisis_Dummy'] = [is_crisis_year(y) for y in yearly_data.index]


# PREDICTION FUNCTION (WITH CRASH TRIGGER)
def predict_future_crash(target_col, df, future_years):
    clean_data = df[['Crisis_Dummy', target_col]].dropna()

    if len(clean_data) < 5:
        return {y: 0 for y in future_years}

    X = pd.DataFrame({'Year': clean_data.index, 'Crisis_Dummy': clean_data['Crisis_Dummy']})
    y = clean_data[target_col]

    model = LinearRegression()
    model.fit(X, y)

    preds = {}
    for year in future_years:
        # CRASH TRIGGER: Force dummy to 1 if year is 2030
        is_shock = 1 if year == 2030 else 0
        input_data = pd.DataFrame({'Year': [year], 'Crisis_Dummy': [is_shock]})
        preds[year] = max(0, model.predict(input_data)[0])

    return preds


# RUN PREDICTIONS
print("Running Crash Scenario Predictions...")
target_years = [2026, 2030, 2035]
results = []
cols_to_predict = [f'Pct_{c}' for c in birth_cols]

for col in cols_to_predict:
    preds = predict_future_crash(col, yearly_data, target_years)
    row = {'Variable': col}
    row.update(preds)
    results.append(row)

results_df = pd.DataFrame(results)


# NORMALIZATION
def normalize_group(df, keyword, year):
    mask = df['Variable'].str.contains(keyword)
    total = df.loc[mask, year].sum()
    if total > 0:
        df.loc[mask, year] = (df.loc[mask, year] / total) * 100


for year in target_years:
    normalize_group(results_df, 'Place of birth', year)


# VISUALIZATION
def get_plot_data(col_name):
    history = yearly_data[col_name].dropna()
    future_row = results_df[results_df['Variable'] == col_name].iloc[0]
    future = pd.Series([future_row[y] for y in target_years], index=target_years)
    return pd.concat([history, future]).sort_index()


plt.figure(figsize=(10, 6))
labels = [c.replace('Pct_Place of birth: ', '') for c in cols_to_predict]

for i, col in enumerate(cols_to_predict):
    if col in results_df['Variable'].values:
        data = get_plot_data(col)
        plt.plot(data.index, data.values, marker='o', label=labels[i])

plt.axvline(x=2024, color='gray', linestyle='--', label='Forecast Start')
plt.axvline(x=2030, color='red', linestyle=':', label='Simulated Crash (2030)')
plt.title("Projected Place of Birth (Scenario: 2030 Shock)")
plt.xlabel("Year")
plt.ylabel("% of Workforce")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('birth_place_crash.png')
print("Saved birth_place_crash.png")
plt.show()