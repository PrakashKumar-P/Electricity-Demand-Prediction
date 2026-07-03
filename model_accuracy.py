import forecast_engine
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error

# Train once

forecast_engine.train_models()

monthly = forecast_engine.monthly
X_cols = forecast_engine.X_cols

lr = forecast_engine.lr
ridge = forecast_engine.ridge
lasso = forecast_engine.lasso
rf = forecast_engine.rf
lgbm = forecast_engine.lgbm
sarima = forecast_engine.sarima
hw = forecast_engine.hw
top5_df = forecast_engine.top5_df

# train-test split again 
train_size = int(len(monthly) * 0.85)
test = monthly.iloc[train_size:]

y_true = test["monthly_demand"].values

# Ensemble Prediction on test set
predictions = []

for i in range(len(test)):
    row = test.iloc[i].copy()
    next_df = pd.DataFrame([row])

    preds = {}
    preds["Linear"] = lr.predict(next_df[X_cols])[0]
    preds["Ridge"] = ridge.predict(next_df[X_cols])[0]
    preds["Lasso"] = lasso.predict(next_df[X_cols])[0]
    preds["RandomForest"] = rf.predict(next_df[X_cols])[0]
    preds["LightGBM"] = lgbm.predict(next_df[X_cols])[0]
    preds["SARIMA"] = sarima.forecast(1).values[0]
    preds["HoltWinters"] = hw.forecast(1).values[0]

    final = 0
    for _, r in top5_df.iterrows():
        final += r["weight"] * preds[r["Model"]]

    predictions.append(final)

y_pred = np.array(predictions)

# Metrics 
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

accuracy = 100 - mape

print("\n📊 MODEL PERFORMANCE")
print("RMSE :", round(rmse,2))
print("MAE  :", round(mae,2))
print("MAPE :", round(mape,2), "%")
print("ACCURACY :", round(accuracy,2), "%")