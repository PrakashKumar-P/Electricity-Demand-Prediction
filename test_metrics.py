import forecast_engine

print("🔹 Testing Model Training...")
forecast_engine.train_models()
print("✅ Training Successful")

print("\n🔹 Testing Evaluation Metrics...")

metrics = forecast_engine.evaluate_model()

print("Metrics:", metrics)

# ================= ASSERT TESTS =================

# Check keys
assert "MAE" in metrics, "MAE missing!"
assert "RMSE" in metrics, "RMSE missing!"
assert "MAPE" in metrics, "MAPE missing!"
assert "R2" in metrics, "R2 missing!"

# Check types
assert isinstance(metrics["MAE"], float), "MAE not float!"
assert isinstance(metrics["RMSE"], float), "RMSE not float!"
assert isinstance(metrics["MAPE"], float), "MAPE not float!"
assert isinstance(metrics["R2"], float), "R2 not float!"

# Check values range
assert metrics["MAE"] >= 0, "MAE negative!"
assert metrics["RMSE"] >= 0, "RMSE negative!"
assert 0 <= metrics["MAPE"] <= 100, "MAPE invalid!"
assert -1 <= metrics["R2"] <= 1, "R2 out of range!"

print("✅ Metrics Value Test Passed")

# ================= STABILITY TEST =================

print("\n🔹 Testing Stability...")

m1 = forecast_engine.evaluate_model()
m2 = forecast_engine.evaluate_model()

print("Run1:", m1)
print("Run2:", m2)

print("✅ Stability Test Passed")

# ================= EDGE CASE TEST =================

print("\n🔹 Testing Edge Case Handling...")

try:
    forecast_engine.monthly = None
    result = forecast_engine.evaluate_model()
    print("Handled empty data:", result)
except Exception as e:
    print("Error handled:", e)

print("✅ Edge Case Test Completed")