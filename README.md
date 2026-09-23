# Predictive Analytics Using Historical Data

I built this project to forecast future sales trends using regression models trained on historical time-series data. The goal was to learn predictive modeling, trend analysis, and data-driven forecasting.

## Tools & Libraries Used
- Python
- pandas, NumPy
- scikit-learn (Linear Regression, Random Forest Regressor)
- matplotlib, seaborn

## What I Did
1. **Data Preparation** — Worked with 4 years of monthly historical sales data (Date, Sales).
2. **Exploratory Data Analysis (EDA)** — Plotted the sales trend over time and checked monthly patterns to spot seasonality.
3. **Cleaning & Preprocessing** — Checked for missing values, then engineered time-based features a regression model can learn from: a numeric time index, sin/cos month encoding (to capture seasonality), lag features (previous month, same month last year), and a rolling 3-month average.
4. **Train/Test Split** — Split the data by time order (not randomly, since it's a time series), keeping the last 6 months as the test set.
5. **Modeling** — Trained two models: Linear Regression and Random Forest Regressor, to compare a simple trend-based approach against a more flexible one.
6. **Evaluation** — Measured MAE, RMSE, MAPE, and R² on the test set to see which model predicted more accurately.
7. **Forecasting** — Used the better-performing model to recursively forecast the next 6 months.

## Results

| Model | MAE | RMSE | MAPE (%) | R² |
|---|---|---|---|---|
| Linear Regression | 15.53 | 20.54 | 3.54 | 0.193 |
| Random Forest | 26.61 | 28.87 | 5.93 | -0.594 |

Linear Regression performed better on this dataset (lower error, positive R²), so I used it to generate the final 6-month forecast.

## What I Learned
This project taught me how to turn a raw time series into features a regression model can actually use (lags, rolling averages, seasonal encoding), why time series data must be split by time order and not shuffled, how to evaluate forecast accuracy with multiple metrics instead of relying on just one, and how to do recursive forecasting for future, unseen periods.

## Files in This Repository
- `predictive_analytics.py` — the full code (data loading, EDA, preprocessing, modeling, evaluation, forecasting)
- `model_evaluation.csv` — accuracy metrics for both models
- `future_forecast.csv` — the predicted sales for the next 6 months
- `historical_sales.png` — the historical sales trend
- `seasonality_boxplot.png` — monthly seasonality check
- `actual_vs_predicted.png` — model predictions vs actual values on the test set
- `future_forecast.png` — final 6-month forecast chart

## Note on the dataset
This uses a realistic synthetic dataset since no dataset file was provided, built with a clear upward trend, yearly seasonality, and random noise. To use real data, replace the data-generation block in `predictive_analytics.py` with `df = pd.read_csv("your_file.csv", parse_dates=["Date"])`, keeping a `Date` and value column (rename to `Sales` or adjust the code).

If You have Query Regarding This Then please Tag the details for understanding and tag your issue regarding this..
