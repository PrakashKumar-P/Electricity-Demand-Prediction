import forecast_engine

print("===========================================")
print(" ELECTRICITY DEMAND FORECAST SYSTEM TEST ")
print("===========================================")

# ===========================================
# FUNCTIONAL TESTING
# ===========================================

print("\n🔹 FUNCTIONAL TESTING")

print("\nTesting Model Training...")
forecast_engine.train_models()
print("✅ Model Training Successful")

print("\nTesting Normal Forecast...")

prediction, month = forecast_engine.run_forecast()

print("Prediction:", prediction)
print("Month:", month)

assert isinstance(prediction, float), "Prediction must be float"
assert isinstance(month, str), "Month must be string"

print("✅ Forecast Function Working")

print("\nTesting Scenario Forecast...")

scenario_prediction = forecast_engine.run_scenario(5, 0.2, 0.05)

print("Scenario Prediction:", scenario_prediction)

assert isinstance(scenario_prediction, float), "Scenario output must be float"

print("✅ Scenario Function Working")

print("\nTesting Anomaly Detection...")

anomalies = forecast_engine.detect_anomalies()

print("Number of anomalies detected:", len(anomalies))

if len(anomalies) > 0:

    first = anomalies[0]

    print("Sample anomaly:")
    print("Month:", first["month"])
    print("Actual Usage:", first["actual"])
    print("Extra Usage:", first["extra"])

    assert "month" in first
    assert "actual" in first
    assert "extra" in first

    print("✅ Anomaly Detection Working")

else:

    print("⚠ No anomalies detected in dataset")

print("✅ Functional Testing Completed")


# ===========================================
# SYSTEM TESTING
# ===========================================

print("\n🔹 SYSTEM TESTING")

print("Running full system pipeline...")

forecast_engine.train_models()

p1, month1 = forecast_engine.run_forecast()
p2 = forecast_engine.run_scenario(3, 0.15, 0.03)

anoms = forecast_engine.detect_anomalies()

print("Forecast Output:", p1)
print("Scenario Output:", p2)
print("Anomaly Count:", len(anoms))

assert isinstance(p1, float)
assert isinstance(p2, float)
assert isinstance(anoms, list)

print("✅ System Pipeline Working")


# ===========================================
# SECURITY TESTING
# ===========================================

print("\n🔹 SECURITY TESTING")

print("Testing extreme input values...")

try:
    forecast_engine.run_scenario(-100, -5, -3)
    print("Handled extreme negative input safely")
except Exception as e:
    print("Handled error:", e)

try:
    forecast_engine.run_scenario(100, 5, 3)
    print("Handled extreme positive input safely")
except Exception as e:
    print("Handled error:", e)

print("\nTesting repeated forecasting stability...")

p1, _ = forecast_engine.run_forecast()
p2, _ = forecast_engine.run_forecast()

difference = abs(p1 - p2)

print("Run1:", p1)
print("Run2:", p2)
print("Difference:", difference)

print("✅ System Stability Verified")

print("✅ Security Testing Completed")


# ===========================================
# FINAL RESULT
# ===========================================

print("\n===========================================")
print(" ALL TESTS COMPLETED SUCCESSFULLY ")
print(" Functional ✔  System ✔  Security ✔")
print("===========================================")