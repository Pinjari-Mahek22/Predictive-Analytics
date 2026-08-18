# %% [markdown]
# # Predictive Analytics Using Historical Data
# Forecast future trends using regression models on time-based features.
# Author: Mahek Kamar Pinjari

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)
sns.set_style("whitegrid")

# %% [markdown]
# ## 1. Load / Generate Historical Dataset
# Monthly sales data with an upward trend + yearly seasonality + noise.
# If you have a real dataset (date + value columns), replace this block with:
# `df = pd.read_csv("your_file.csv", parse_dates=["Date"])`

# %%
n_months = 48  # 4 years of monthly history
dates = pd.date_range(start="2022-01-01", periods=n_months, freq="MS")

trend = np.linspace(200, 500, n_months)                       # gradual growth
seasonality = 40 * np.sin(2 * np.pi * np.arange(n_months) / 12)  # yearly cycle
noise = np.random.normal(0, 15, n_months)
sales = trend + seasonality + noise
sales = np.round(np.clip(sales, 50, None), 1)

df = pd.DataFrame({"Date": dates, "Sales": sales})
print(df.head())
print(df.describe())

# %% [markdown]
# ## 2. Exploratory Data Analysis

# %%
plt.figure(figsize=(11, 4))
plt.plot(df["Date"], df["Sales"], marker="o", color="#0E7C7B")
plt.title("Historical Monthly Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.tight_layout()
plt.savefig("historical_sales.png", dpi=150)
plt.close()

df["Month"] = df["Date"].dt.month
plt.figure(figsize=(8, 4))
sns.boxplot(x="Month", y="Sales", data=df, color="#D97706")
plt.title("Sales Distribution by Month (Seasonality Check)")
plt.tight_layout()
plt.savefig("seasonality_boxplot.png", dpi=150)
plt.close()

# %% [markdown]
# ## 3. Clean & Preprocess
# Check for missing values, create time-based features regression models can learn from:
# a numeric time index, seasonal encoding (sin/cos of month), and lag features.

# %%
print("Missing values:\n", df.isnull().sum())

df["time_idx"] = np.arange(len(df))
df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
df["lag_1"] = df["Sales"].shift(1)
df["lag_12"] = df["Sales"].shift(12)
df["rolling_mean_3"] = df["Sales"].shift(1).rolling(window=3).mean()

df_model = df.dropna().reset_index(drop=True)
print(f"Rows after adding lag features and dropping NaNs: {len(df_model)}")

# %% [markdown]
# ## 4. Train/Test Split (time-ordered — never shuffle time series!)

# %%
test_size = 6
train = df_model.iloc[:-test_size]
test = df_model.iloc[-test_size:]

features = ["time_idx", "month_sin", "month_cos", "lag_1", "lag_12", "rolling_mean_3"]
X_train, y_train = train[features], train["Sales"]
X_test, y_test = test[features], test["Sales"]

print(f"Train size: {len(train)}, Test size: {len(test)}")

# %% [markdown]
# ## 5. Train Models — Linear Regression & Random Forest Regressor

# %%
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

# %% [markdown]
# ## 6. Evaluate Model Accuracy

# %%
def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    print(f"{name} -> MAE: {mae:.2f} | RMSE: {rmse:.2f} | MAPE: {mape:.2f}% | R2: {r2:.3f}")
    return {"Model": name, "MAE": round(mae, 2), "RMSE": round(rmse, 2),
             "MAPE_%": round(mape, 2), "R2": round(r2, 3)}

results = [
    evaluate("Linear Regression", y_test.values, lr_preds),
    evaluate("Random Forest", y_test.values, rf_preds),
]
results_df = pd.DataFrame(results)
results_df.to_csv("model_evaluation.csv", index=False)
print(results_df)

# %% [markdown]
# ## 7. Visualize Actual vs Predicted

# %%
plt.figure(figsize=(11, 5))
plt.plot(df_model["Date"], df_model["Sales"], label="Actual", color="#1B2A4A", marker="o")
plt.plot(test["Date"], lr_preds, label="Linear Regression Prediction", color="#0E7C7B", marker="x", linestyle="--")
plt.plot(test["Date"], rf_preds, label="Random Forest Prediction", color="#D97706", marker="s", linestyle="--")
plt.axvline(test["Date"].iloc[0], color="grey", linestyle=":", label="Train/Test Split")
plt.title("Actual vs Predicted Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.close()

# %% [markdown]
# ## 8. Forecast Future Months (recursive forecasting with the better model)

# %%
best_model = rf if results_df.loc[1, "MAE"] < results_df.loc[0, "MAE"] else lr
best_name = "Random Forest" if best_model is rf else "Linear Regression"
print(f"Using {best_name} for future forecasting (lower MAE).")

future_steps = 6
history = df_model[["Date", "Sales"]].copy()
last_time_idx = df_model["time_idx"].iloc[-1]

future_rows = []
for step in range(1, future_steps + 1):
    next_date = history["Date"].iloc[-1] + pd.DateOffset(months=1)
    next_time_idx = last_time_idx + step
    month = next_date.month
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    lag_1 = history["Sales"].iloc[-1]
    lag_12 = history["Sales"].iloc[-12] if len(history) >= 12 else history["Sales"].mean()
    rolling_mean_3 = history["Sales"].iloc[-3:].mean()

    X_future = pd.DataFrame([{
        "time_idx": next_time_idx, "month_sin": month_sin, "month_cos": month_cos,
        "lag_1": lag_1, "lag_12": lag_12, "rolling_mean_3": rolling_mean_3,
    }])
    pred = best_model.predict(X_future)[0]

    future_rows.append({"Date": next_date, "Sales": pred})
    history = pd.concat([history, pd.DataFrame([{"Date": next_date, "Sales": pred}])], ignore_index=True)

future_df = pd.DataFrame(future_rows)
future_df.to_csv("future_forecast.csv", index=False)
print(future_df)

plt.figure(figsize=(11, 5))
plt.plot(df_model["Date"], df_model["Sales"], label="Historical", color="#1B2A4A", marker="o")
plt.plot(future_df["Date"], future_df["Sales"], label=f"Forecast ({best_name})",
          color="#D97706", marker="o", linestyle="--")
plt.axvline(df_model["Date"].iloc[-1], color="grey", linestyle=":", label="Forecast Start")
plt.title("6-Month Sales Forecast")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.savefig("future_forecast.png", dpi=150)
plt.close()

print("Saved: model_evaluation.csv, future_forecast.csv, and all plots (.png)")
