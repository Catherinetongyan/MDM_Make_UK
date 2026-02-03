import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

master_df = pd.read_csv('master annual data.csv')
ethnicity_df = pd.read_csv('manufacturing_ethnicity_year_share_pivot.csv')

master_df = master_df[master_df['Year'] >= 1995]
ethnicity_df = ethnicity_df[ethnicity_df['Year'] >= 1995]

master_df = master_df.fillna(0)

age_cols = [c for c in master_df.columns if 'Aggregate bands' in c]
master_df['Total_Emp'] = master_df[age_cols].sum(axis=1)

yearly_data = master_df.groupby('Year').sum(numeric_only=True)

female_counts = master_df[master_df['Sex'] == 'Female'].groupby('Year')['Total_Emp'].sum()
yearly_data['Pct_Female'] = (female_counts / yearly_data['Total_Emp']) * 100

for col in age_cols:
    yearly_data[f'Pct_{col}'] = (yearly_data[col] / yearly_data['Total_Emp']) * 100

full_data = yearly_data.join(ethnicity_df.set_index('Year'), how='outer')


def is_crisis_year(year):
    if year in [2008, 2009, 2020, 2021]:
        return 1
    return 0


full_data['Crisis_Dummy'] = [is_crisis_year(y) for y in full_data.index]


def predict_future(target_col, df, future_years):
    clean_data = df[['Crisis_Dummy', target_col]].dropna()

    if len(clean_data) < 5:
        return {y: np.nan for y in future_years}

    X = pd.DataFrame({'Year': clean_data.index, 'Crisis_Dummy': clean_data['Crisis_Dummy']})
    y = clean_data[target_col]

    model = LinearRegression()
    model.fit(X, y)

    preds = {}
    for year in future_years:
        is_shock = 1 if year == 2030 else 0
        input_data = pd.DataFrame({'Year': [year], 'Crisis_Dummy': [is_shock]})
        preds[year] = max(0, model.predict(input_data)[0])

    return preds


print("Running Predictions...")
target_years = [2026, 2030, 2035]
results = []
cols_to_predict = [c for c in full_data.columns if c.startswith('Pct_')] + \
                  [c for c in ethnicity_df.columns if c != 'Year']

for col in cols_to_predict:
    preds = predict_future(col, full_data, target_years)
    row = {'Variable': col}
    row.update(preds)
    results.append(row)

results_df = pd.DataFrame(results)


def normalize_group(df, keyword, year):
    mask = df['Variable'].str.contains(keyword)
    total = df.loc[mask, year].sum()
    if total > 0:
        df.loc[mask, year] = (df.loc[mask, year] / total) * 100


for year in target_years:
    normalize_group(results_df, 'Aggregate bands', year)
    normalize_group(results_df, 'Place of birth', year)

results_df.to_csv('forecast_results_2030_CRASH_SCENARIO.csv', index=False)
print("Data saved to forecast_results_2030_CRASH_SCENARIO.csv")

print("Generating graphs...")


def get_plot_data(col_name):
    history = full_data[col_name].dropna()
    future_row = results_df[results_df['Variable'] == col_name].iloc[0]
    future = pd.Series([future_row[y] for y in target_years], index=target_years)
    return pd.concat([history, future]).sort_index()


plt.figure(figsize=(10, 6))
age_vars = [c for c in full_data.columns if 'Pct_Aggregate bands' in c]
labels = ['15-24 (Young)', '25-54 (Core)', '55-64 (Older)', '65+ (Retirement)']

for i, col in enumerate(age_vars):
    if col in results_df['Variable'].values:
        data = get_plot_data(col)
        plt.plot(data.index, data.values, marker='o', label=labels[i])

plt.axvline(x=2024, color='gray', linestyle='--', label='Forecast Start')
plt.axvline(x=2030, color='red', linestyle=':', label='Simulated Crash (2030)')
plt.title("Projected Age Structure (Scenario: 2030 Shock)")
plt.xlabel("Year")
plt.ylabel("% of Workforce")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('age_trend_graph_crash.png')
plt.show()

plt.figure(figsize=(10, 6))
eth_vars = ['White British', 'White Other', 'Indian', 'Black', 'Mixed']

for col in eth_vars:
    if col in full_data.columns and col in results_df['Variable'].values:
        data = get_plot_data(col)
        plt.plot(data.index, data.values, marker='o', label=col)

plt.axvline(x=2024, color='gray', linestyle='--', label='Forecast Start')
plt.axvline(x=2030, color='red', linestyle=':', label='Simulated Crash (2030)')
plt.title("Projected Ethnicity Trends (Scenario: 2030 Shock)")
plt.xlabel("Year")
plt.ylabel("% of Workforce")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('ethnicity_trend_graph_crash.png')
plt.show()