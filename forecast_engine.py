import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

monthly = None
X_cols = None

# trained models
lr = None
ridge = None
lasso = None
rf = None
lgbm = None
sarima = None
hw = None
top5_df = None


# ================= MODEL TRAINING =================

def train_models():

    global monthly, X_cols, lr, ridge, lasso, rf, lgbm, sarima, hw, top5_df

    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from lightgbm import LGBMRegressor
    from sklearn.metrics import mean_squared_error

    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    print("🔵 Training models...")

    try:
        # ---------- Load Data ----------
        df = pd.read_csv("electricity_data.csv")

        # Date parsing (DD/MM/YYYY format)
        df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["timestamp"])

        print("Last timestamp in raw dataset:", df["timestamp"].max())

        df = df.sort_values("timestamp")
        df.set_index("timestamp", inplace=True)

        df = df[df["electricity_demand"] > 0]
        df = df[df["temperature"].between(-10, 55)]

        # ---------- Monthly Aggregation ----------
        monthly = df.resample("ME").agg({
            "electricity_demand": "sum",
            "temperature": ["mean", "max", "min"],
            "is_weekend": "mean"
        })

        monthly.columns = [
            "monthly_demand",
            "avg_temp",
            "max_temp",
            "min_temp",
            "weekend_ratio"
        ]

        print("Last month after resampling:", monthly.index.max())

        # Holiday effect
        monthly["holiday_effect"] = (monthly["weekend_ratio"] > 0.30).astype(int)

        # ---------- Time Features ----------
        monthly["year"] = monthly.index.year
        monthly["month"] = monthly.index.month
        monthly["trend"] = np.arange(len(monthly))

        monthly["season"] = monthly["month"].map({
            12: "Winter", 1: "Winter", 2: "Winter",
            3: "Summer", 4: "Summer", 5: "Summer",
            6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
            10: "PostMonsoon", 11: "PostMonsoon"
        })

        monthly = pd.get_dummies(monthly, columns=["season"], drop_first=True)

        # ---------- Outlier Handling ----------
        Q1 = monthly["monthly_demand"].quantile(0.25)
        Q3 = monthly["monthly_demand"].quantile(0.75)
        IQR = Q3 - Q1
        monthly["monthly_demand"] = monthly["monthly_demand"].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)

        # ---------- Lag Features ----------
        for lag in [1, 3, 6, 12]:
            monthly[f"lag_{lag}"] = monthly["monthly_demand"].shift(lag)

        # ---------- Rolling Stats ----------
        for win in [3, 6, 12]:
            monthly[f"roll_mean_{win}"] = monthly["monthly_demand"].rolling(win).mean()
            monthly[f"roll_std_{win}"] = monthly["monthly_demand"].rolling(win).std()

        monthly = monthly.dropna()

        # ---------- Train/Test Split ----------
        train_size = int(len(monthly) * 0.85)
        train = monthly.iloc[:train_size]
        test = monthly.iloc[train_size:]

        X_cols = [c for c in monthly.columns if c != "monthly_demand"]

        X_train = train[X_cols]
        y_train = train["monthly_demand"]
        X_test = test[X_cols]
        y_test = test["monthly_demand"]

        model_preds = {}

        # Baselines
        model_preds["Naive"] = test["lag_1"].values
        model_preds["Seasonal_Naive"] = test["lag_12"].values

        # SARIMA
        try:
            sarima = SARIMAX(train["monthly_demand"], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
            model_preds["SARIMA"] = sarima.forecast(len(test)).values
            print("✅ SARIMA trained")
        except Exception as e:
            print(f"⚠️ SARIMA failed: {e}")
            model_preds["SARIMA"] = model_preds["Naive"]

        # Holt Winters
        try:
            hw = ExponentialSmoothing(train["monthly_demand"], trend="add", seasonal="add", seasonal_periods=12).fit()
            model_preds["HoltWinters"] = hw.forecast(len(test)).values
            print("✅ Holt-Winters trained")
        except Exception as e:
            print(f"⚠️ Holt-Winters failed: {e}")
            model_preds["HoltWinters"] = model_preds["Naive"]

        # ML Models
        lr = LinearRegression().fit(X_train, y_train)
        model_preds["Linear"] = lr.predict(X_test)
        print("✅ Linear Regression trained")

        ridge = Ridge(alpha=1.0).fit(X_train, y_train)
        model_preds["Ridge"] = ridge.predict(X_test)
        print("✅ Ridge trained")

        lasso = Lasso(alpha=0.001).fit(X_train, y_train)
        model_preds["Lasso"] = lasso.predict(X_test)
        print("✅ Lasso trained")

        rf = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1).fit(X_train, y_train)
        model_preds["RandomForest"] = rf.predict(X_test)
        print("✅ Random Forest trained")

        lgbm = LGBMRegressor(n_estimators=600, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1, verbose=-1).fit(X_train, y_train)
        model_preds["LightGBM"] = lgbm.predict(X_test)
        print("✅ LightGBM trained")

        # ---------- Ensemble Selection ----------
        model_scores = {}
        for name, pred in model_preds.items():
            model_scores[name] = rmse(y_test, pred)

        results_df = pd.DataFrame(model_scores.items(), columns=["Model", "RMSE"]).sort_values("RMSE")
        top5_df = results_df.head(5).copy()

        top5_df["weight"] = 1 / top5_df["RMSE"]
        top5_df["weight"] = top5_df["weight"] / top5_df["weight"].sum()

        print("\n📊 Top 5 Models:")
        print(top5_df.to_string(index=False))
        print("✅ Training completed and models stored in memory")
        
        return True

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ================= NORMAL FORECAST =================

def run_forecast():
    import pandas as pd
    global monthly, X_cols, lr, ridge, lasso, rf, lgbm, sarima, hw, top5_df

    if monthly is None or len(monthly) == 0:
        print("❌ Models not trained!")
        return 1000.0, "January 2025"

    try:
        # safer next month calculation
        last_month = monthly.index.max()
        next_month_index = last_month + pd.offsets.MonthEnd(1)
        
        print(f"Last month in data: {last_month}")
        print(f"Next month to predict: {next_month_index}")

        last_row = monthly.loc[last_month].copy()

        # Update time features
        last_row["trend"] += 1
        last_row["year"] = next_month_index.year
        last_row["month"] = next_month_index.month

        # Update season based on new month (consistent with training)
        next_month_num = next_month_index.month
        if next_month_num in [12, 1, 2]:
            new_season = "Winter"
        elif next_month_num in [3, 4, 5]:
            new_season = "Summer"
        elif next_month_num in [6, 7, 8, 9]:
            new_season = "Monsoon"
        else:
            new_season = "PostMonsoon"
        
        # Update season dummy variables
        season_columns = [col for col in X_cols if col.startswith("season_")]
        for col in season_columns:
            last_row[col] = 0
        season_col_name = f"season_{new_season}"
        if season_col_name in X_cols:
            last_row[season_col_name] = 1

        next_df = pd.DataFrame([last_row])
        next_df.index = [next_month_index]

        # Ensure all columns exist
        for col in X_cols:
            if col not in next_df.columns:
                next_df[col] = 0

        next_df = next_df[X_cols]

        # Get predictions from all models
        preds = {}
        preds["Linear"] = lr.predict(next_df)[0]
        preds["Ridge"] = ridge.predict(next_df)[0]
        preds["Lasso"] = lasso.predict(next_df)[0]
        preds["RandomForest"] = rf.predict(next_df)[0]
        preds["LightGBM"] = lgbm.predict(next_df)[0]
        
        if sarima is not None:
            try:
                preds["SARIMA"] = sarima.forecast(1).values[0]
            except:
                pass
        if hw is not None:
            try:
                preds["HoltWinters"] = hw.forecast(1).values[0]
            except:
                pass
        
        # Add baseline predictions if available
        if "lag_1" in next_df.columns:
            preds["Naive"] = next_df["lag_1"].values[0]
        if "lag_12" in next_df.columns:
            preds["Seasonal_Naive"] = next_df["lag_12"].values[0]

        # Weighted ensemble
        final = 0
        total_weight = 0
        for _, row in top5_df.iterrows():
            model_name = row["Model"]
            if model_name in preds:
                final += row["weight"] * preds[model_name]
                total_weight += row["weight"]
        
        if total_weight > 0:
            final = final / total_weight
        else:
            final = preds.get("Linear", 1000)

        return round(final, 2), next_month_index.strftime("%B %Y")
    
    except Exception as e:
        print(f"❌ Forecast error: {e}")
        import traceback
        traceback.print_exc()
        return 1250.0, "Next Month"

# ================= PREDICT SPECIFIC MONTH =================
# ================= PREDICT SPECIFIC MONTH =================

def predict_specific_month(months_ahead):
    """
    Predict demand for a specific month in the future
    
    Args:
        months_ahead: Number of months ahead to predict (1 = next month)
    
    Returns:
        Predicted demand value
    """
    global monthly, X_cols, lr, ridge, lasso, rf, lgbm, top5_df
    
    if monthly is None or len(monthly) == 0:
        print("❌ No monthly data available")
        return 400000
    
    try:
        print(f"📊 Predicting {months_ahead} month(s) ahead...")
        
        # Get the last row from historical data
        last_row = monthly.iloc[-1:].copy()
        current_demand = monthly["monthly_demand"].iloc[-1]
        
        # Store the last known date
        last_date = monthly.index[-1]
        print(f"Last known date: {last_date}, Demand: {current_demand}")
        
        for step in range(months_ahead):
            print(f"  Step {step + 1} of {months_ahead}")
            
            # Calculate next month and year
            current_month = last_row["month"].iloc[0]
            current_year = last_row["year"].iloc[0] if "year" in last_row.columns else last_date.year
            
            next_month = current_month + 1
            next_year = current_year
            if next_month > 12:
                next_month = 1
                next_year += 1
            
            # Update time features
            last_row["trend"] = last_row["trend"].iloc[0] + 1
            last_row["year"] = next_year
            last_row["month"] = next_month
            
            # Update season based on month (consistent with training data)
            # This mapping MUST match the one used in train_models()
            if next_month in [12, 1, 2]:
                new_season = "Winter"
            elif next_month in [3, 4, 5]:
                new_season = "Summer"
            elif next_month in [6, 7, 8, 9]:
                new_season = "Monsoon"
            else:  # 10, 11
                new_season = "PostMonsoon"
            
            print(f"    Month: {next_month}, Year: {next_year}, Season: {new_season}")
            
            # Update season dummy variables (these column names match your training)
            # Your training created columns like: season_Summer, season_Monsoon, season_PostMonsoon
            season_columns = [col for col in X_cols if col.startswith("season_")]
            for col in season_columns:
                last_row[col] = 0
            
            # Set the correct season column
            season_col_name = f"season_{new_season}"
            if season_col_name in X_cols:
                last_row[season_col_name] = 1
            else:
                print(f"    Warning: {season_col_name} not found in features")
            
            # Update lag features (shift previous demands)
            # For prediction, we use the last known values
            if 'lag_1' in last_row.columns:
                last_row['lag_1'] = current_demand
            if 'lag_3' in last_row.columns:
                # Use older historical data if available
                if len(monthly) >= 3:
                    last_row['lag_3'] = monthly["monthly_demand"].iloc[-3]
            if 'lag_6' in last_row.columns:
                if len(monthly) >= 6:
                    last_row['lag_6'] = monthly["monthly_demand"].iloc[-6]
            if 'lag_12' in last_row.columns:
                if len(monthly) >= 12:
                    last_row['lag_12'] = monthly["monthly_demand"].iloc[-12]
            
            # Update rolling statistics if they exist
            if 'roll_mean_3' in last_row.columns and len(monthly) >= 3:
                last_row['roll_mean_3'] = monthly["monthly_demand"].iloc[-3:].mean()
            if 'roll_mean_6' in last_row.columns and len(monthly) >= 6:
                last_row['roll_mean_6'] = monthly["monthly_demand"].iloc[-6:].mean()
            if 'roll_mean_12' in last_row.columns and len(monthly) >= 12:
                last_row['roll_mean_12'] = monthly["monthly_demand"].iloc[-12:].mean()
            
            if 'roll_std_3' in last_row.columns and len(monthly) >= 3:
                last_row['roll_std_3'] = monthly["monthly_demand"].iloc[-3:].std()
            if 'roll_std_6' in last_row.columns and len(monthly) >= 6:
                last_row['roll_std_6'] = monthly["monthly_demand"].iloc[-6:].std()
            if 'roll_std_12' in last_row.columns and len(monthly) >= 12:
                last_row['roll_std_12'] = monthly["monthly_demand"].iloc[-12:].std()
            
            # Holiday effect (based on month/season)
            # More holidays in certain months
            if next_month in [1, 10, 11, 12]:  # January, October, November, December
                last_row["holiday_effect"] = 1
            else:
                last_row["holiday_effect"] = 0
            
            # Weekend ratio (typical pattern)
            last_row["weekend_ratio"] = 0.27  # ~8 days per month
            
            # Temperature values (seasonal averages)
            seasonal_temps = {
                "Winter": {"avg": 22, "max": 28, "min": 16},
                "Summer": {"avg": 32, "max": 38, "min": 26},
                "Monsoon": {"avg": 30, "max": 34, "min": 24},
                "PostMonsoon": {"avg": 28, "max": 32, "min": 22}
            }
            temp = seasonal_temps.get(new_season, {"avg": 28, "max": 32, "min": 24})
            last_row["avg_temp"] = temp["avg"]
            last_row["max_temp"] = temp["max"]
            last_row["min_temp"] = temp["min"]
            
            # Ensure all required columns exist
            for col in X_cols:
                if col not in last_row.columns:
                    last_row[col] = 0
                    print(f"    Warning: Missing column '{col}', set to 0")
            
            # Prepare data for prediction
            next_df = last_row[X_cols]
            
            # Get predictions from all models
            preds = {}
            if lr is not None:
                preds["Linear"] = lr.predict(next_df)[0]
            if ridge is not None:
                preds["Ridge"] = ridge.predict(next_df)[0]
            if lasso is not None:
                preds["Lasso"] = lasso.predict(next_df)[0]
            if rf is not None:
                preds["RandomForest"] = rf.predict(next_df)[0]
            if lgbm is not None:
                preds["LightGBM"] = lgbm.predict(next_df)[0]
            
            # Add SARIMA and Holt-Winters if available
            if 'sarima' in globals() and sarima is not None:
                try:
                    preds["SARIMA"] = sarima.forecast(1).values[0]
                except:
                    pass
            if 'hw' in globals() and hw is not None:
                try:
                    preds["HoltWinters"] = hw.forecast(1).values[0]
                except:
                    pass
            
            # Weighted ensemble prediction
            final_prediction = 0
            total_weight = 0
            
            for _, row in top5_df.iterrows():
                model_name = row["Model"]
                if model_name in preds:
                    final_prediction += row["weight"] * preds[model_name]
                    total_weight += row["weight"]
            
            if total_weight > 0:
                current_demand = final_prediction / total_weight
            else:
                # Fallback to Linear Regression if ensemble fails
                current_demand = preds.get("Linear", current_demand)
                print(f"    Using fallback (Linear): {current_demand}")
            
            print(f"    Predicted demand for month {next_month}/{next_year}: {current_demand:.2f}")
            
            # Update current_demand for next iteration
            # Create a new row for the next iteration
            last_row = next_df.copy()
            # Convert back to DataFrame for next iteration
            last_row = pd.DataFrame([last_row.iloc[0].to_dict()])
        
        return float(current_demand)
        
    except Exception as e:
        print(f"❌ Predict specific month error: {e}")
        import traceback
        traceback.print_exc()
        # Return a fallback value using run_forecast
        try:
            base_pred, _ = run_forecast()
            return base_pred * (1 + (months_ahead * 0.02))
        except:
            return 400000

# ================= SCENARIO FORECAST =================

def run_scenario(temp_change, weekend_ratio, holiday_ratio):
    """
    Run scenario-based forecast with custom parameters
    
    Args:
        temp_change: Temperature change in Celsius (positive = warmer, increases demand)
        weekend_ratio: Ratio of weekend days (0-1)
        holiday_ratio: Ratio of holiday days (0-1)
    """
    
    global monthly, X_cols, lr, ridge, lasso, rf, lgbm, sarima, hw, top5_df

    if monthly is None or len(monthly) == 0:
        print("❌ Models not trained!")
        return 1200.0

    try:
        # Get last month data
        last_month = monthly.index.max()
        last_row = monthly.loc[last_month].copy()

        # Temperature increase = higher demand
        last_row["avg_temp"] += temp_change
        last_row["max_temp"] += temp_change
        last_row["min_temp"] += temp_change
        
        # Weekend effect (more weekends = higher demand)
        last_row["weekend_ratio"] = weekend_ratio
        
        # Holiday effect (more holidays = higher demand)
        last_row["holiday_effect"] = 1 if holiday_ratio > 0.1 else 0

        # Update time features
        last_row["trend"] += 1
        last_row["month"] = (last_row["month"] % 12) + 1
        
        # Update season based on new month
        month_map = {
            12: "Winter", 1: "Winter", 2: "Winter",
            3: "Summer", 4: "Summer", 5: "Summer",
            6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
            10: "PostMonsoon", 11: "PostMonsoon"
        }
        new_season = month_map.get(last_row["month"], "Summer")
        
        # Update season dummy variables
        for col in X_cols:
            if col.startswith("season_"):
                last_row[col] = 0
        last_row[f"season_{new_season}"] = 1

        # Prepare for prediction
        next_df = pd.DataFrame([last_row])
        
        # Ensure all columns exist
        for col in X_cols:
            if col not in next_df.columns:
                next_df[col] = 0

        next_df = next_df[X_cols]

        # Get predictions from all models
        preds = {}
        
        if lr is not None:
            preds["Linear"] = lr.predict(next_df)[0]
        if ridge is not None:
            preds["Ridge"] = ridge.predict(next_df)[0]
        if lasso is not None:
            preds["Lasso"] = lasso.predict(next_df)[0]
        if rf is not None:
            preds["RandomForest"] = rf.predict(next_df)[0]
        if lgbm is not None:
            preds["LightGBM"] = lgbm.predict(next_df)[0]

        # Weighted ensemble
        final_prediction = 0
        total_weight = 0
        
        for _, row in top5_df.iterrows():
            model_name = row["Model"]
            if model_name in preds:
                final_prediction += row["weight"] * preds[model_name]
                total_weight += row["weight"]
        
        if total_weight > 0:
            final_prediction = final_prediction / total_weight
        else:
            final_prediction = preds.get("Linear", 1200)

        return round(float(final_prediction), 2)

    except Exception as e:
        print(f"❌ Scenario error: {e}")
        import traceback
        traceback.print_exc()
        return 1250.0


# ================= ANOMALY DETECTION =================

def detect_anomalies():
    global monthly

    if monthly is None:
        return []

    data = monthly.copy()
    anomalies = []

    for i in range(len(data)):
        row = data.iloc[i]
        index = data.index[i]
        month_num = index.month

        # use previous data only (historical baseline)
        history = data.iloc[:i]
        history = history[history.index.month == month_num]

        if len(history) < 3:
            continue

        expected_avg = history["monthly_demand"].mean()
        lower_range = expected_avg * 0.92
        upper_range = expected_avg * 1.09
        actual_usage = row["monthly_demand"]

        if actual_usage > upper_range:
            extra_usage = actual_usage - upper_range
            anomalies.append({
                "date": index,
                "month": index.strftime("%B %Y"),
                "range_low": round(lower_range, 0),
                "range_high": round(upper_range, 0),
                "actual": round(actual_usage, 0),
                "extra": round(extra_usage, 0)
            })

    anomalies = sorted(anomalies, key=lambda x: x["date"], reverse=True)
    return anomalies


# ================= MODEL EVALUATION =================

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model():
    global monthly, X_cols, lr

    if monthly is None or len(monthly) == 0:
        return {}

    # split data
    split = int(len(monthly) * 0.8)
    train = monthly.iloc[:split]
    test = monthly.iloc[split:]

    X_train = train[X_cols]
    y_train = train["monthly_demand"]
    X_test = test[X_cols]
    y_test = test["monthly_demand"]

    # train model
    lr.fit(X_train, y_train)
    preds = lr.predict(X_test)

    # metrics
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
    r2 = r2_score(y_test, preds)

    return {
        "MAE": float(round(mae, 2)),
        "RMSE": float(round(rmse, 2)),
        "MAPE": float(round(mape, 2)),
        "R2": float(round(r2, 3))
    }


# ================= KPI METRICS =================

def get_kpi_metrics():
    global monthly
    
    if monthly is None or len(monthly) == 0:
        return {
            "avg_demand": 0,
            "peak_month": "N/A",
            "lowest_month": "N/A",
            "growth_rate": 0
        }
    
    try:
        avg_demand = round(monthly["monthly_demand"].mean(), 2)
        
        peak_idx = monthly["monthly_demand"].idxmax()
        peak_month = peak_idx.strftime("%B %Y")
        
        lowest_idx = monthly["monthly_demand"].idxmin()
        lowest_month = lowest_idx.strftime("%B %Y")
        
        if len(monthly) >= 24:
            recent = monthly["monthly_demand"].iloc[-12:].mean()
            previous = monthly["monthly_demand"].iloc[-24:-12].mean()
            growth_rate = round(((recent - previous) / previous) * 100, 2)
        else:
            growth_rate = 0
        
        return {
            "avg_demand": avg_demand,
            "peak_month": peak_month,
            "lowest_month": lowest_month,
            "growth_rate": growth_rate
        }
        
    except Exception as e:
        print(f"❌ KPI error: {e}")
        return {
            "avg_demand": 0,
            "peak_month": "N/A",
            "lowest_month": "N/A",
            "growth_rate": 0
        }


# ================= DEMAND ANALYSIS =================

def get_demand_analysis():
    global monthly
    
    if monthly is None or len(monthly) == 0:
        return {}, "No data available"
    
    try:
        graphs = {
            "monthly_trend": {
                "labels": [d.strftime("%b %Y") for d in monthly.index[-24:]],
                "values": list(monthly["monthly_demand"].iloc[-24:])
            }
        }
        
        summary = f"""
Demand Analysis Summary:
• Total Period: {len(monthly)} months
• Average Monthly Demand: {round(monthly['monthly_demand'].mean(), 2)} units
• Peak Demand: {round(monthly['monthly_demand'].max(), 2)} units
• Lowest Demand: {round(monthly['monthly_demand'].min(), 2)} units
• Standard Deviation: {round(monthly['monthly_demand'].std(), 2)} units
        """
        
        return graphs, summary.strip()
        
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        return {}, "Analysis unavailable"
    


    # Add this function to forecast_engine.py
def get_available_dates():
    """Get list of available dates in the dataset"""
    global monthly
    if monthly is not None:
        return [d.strftime("%B %Y") for d in monthly.index]
    return []


# ================= INITIALIZATION =================

print("=" * 50)
print("🔧 Initializing Forecast Engine...")
print("=" * 50)

try:
    train_models()
except Exception as e:
    print(f"❌ Initialization failed: {e}")
    print("Please ensure electricity_data.csv exists in the correct location")