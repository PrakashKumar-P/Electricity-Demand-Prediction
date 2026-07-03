from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import sqlite3
import random
from flask_mail import Mail, Message
import forecast_engine
import pandas as pd
import datetime

app = Flask(__name__)
app.secret_key = "electric_secret"

# ================= MODEL LOADING =================
MODEL_READY = True
MODEL_ERROR = ""
try:
    forecast_engine.train_models()
except Exception as e:
    MODEL_READY = False
    MODEL_ERROR = str(e)

# ================= ADMIN EMAIL =================
ADMIN_EMAILS = [
    "roshankumar020205@gmail.com",
    "prakashalistor@gmail.com",
    "tamil04681@gmail.com"
]

# ================= MAIL CONFIG =================
app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 587
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USERNAME"] = "roshgtm007@gmail.com"
app.config["MAIL_PASSWORD"] = "iljs sxqw hmdr ydis"

mail = Mail(app)

# ================= DATABASE =================
def get_db():
    return sqlite3.connect("users.db")

def init_db():
    db = get_db()
    cursor = db.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            phone TEXT UNIQUE
        )
    """)
    db.commit()
    db.close()

init_db()

# ================= SEND OTP =================
def send_otp_to_admin_email(otp):
    msg = Message(
        subject="Admin Approval OTP",
        sender=app.config["MAIL_USERNAME"],
        recipients=ADMIN_EMAILS
    )
    msg.body = f"""
ADMIN APPROVAL REQUIRED

OTP: {otp}

Please share this OTP only if the user is verified.
"""
    mail.send(msg)

# ================= REGISTER =================
@app.route("/register", methods=["GET", "POST"])
def register():

    message = ""
    username_value = session.get("username", "")
    phone_value = session.get("phone", "")

    if request.method == "POST":
        action = request.form.get("action")

        if action == "send_otp":
            username = request.form.get("username").strip()
            password = request.form.get("password").strip()
            phone = request.form.get("phone").strip()

            db = get_db()
            cursor = db.cursor()

            cursor.execute(
                "SELECT * FROM users WHERE username=? OR phone=?",
                (username, phone)
            )
            existing = cursor.fetchone()
            db.close()

            if existing:
                message = "Username or Mobile number already exists"
                username_value = username
                phone_value = phone
            else:
                session["username"] = username
                session["password"] = password
                session["phone"] = phone

                otp = str(random.randint(100000, 999999))
                session["otp"] = otp

                send_otp_to_admin_email(otp)
                message = "OTP sent to Admin Email"

                username_value = username
                phone_value = phone

        elif action == "verify_otp":

            required_keys = ("username", "password", "phone", "otp")

            if not all(key in session for key in required_keys):
                message = "Session expired. Please request OTP again."

            elif request.form.get("otp") == session.get("otp"):

                db = get_db()
                cursor = db.cursor()

                cursor.execute(
                    "SELECT * FROM users WHERE username=? OR phone=?",
                    (session["username"], session["phone"])
                )
                existing = cursor.fetchone()

                if existing:
                    message = "Username or Mobile number already exists"
                else:
                    cursor.execute(
                        "INSERT INTO users (username, password, phone) VALUES (?, ?, ?)",
                        (session["username"], session["password"], session["phone"])
                    )
                    db.commit()
                    message = "Registration Successful! Please login"

                db.close()
                session.clear()

            else:
                message = "Invalid OTP"

    return render_template(
        "register.html",
        message=message,
        username_value=username_value,
        phone_value=phone_value
    )

# ================= LOGIN =================
@app.route("/", methods=["GET", "POST"])
def login():
    message = ""

    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        db = get_db()
        cursor = db.cursor()

        cursor.execute(
            "SELECT * FROM users WHERE username=? AND password=?",
            (username, password)
        )
        user = cursor.fetchone()
        db.close()

        if user:
            session["user"] = username
            return redirect(url_for("dashboard"))
        else:
            message = "Invalid Username or Password"

    return render_template("login.html", message=message)

# ================= DASHBOARD =================
@app.route("/dashboard")
def dashboard():

    if "user" not in session:
        return redirect(url_for("login"))

    # get last 12 months
    data = forecast_engine.monthly.tail(12)

    months = [d.strftime("%b %Y") for d in data.index]
    values = list(data["monthly_demand"])

    return render_template("dashboard.html", 
                      username=session["user"],
                      months=months,
                      values=values)


# ================= FORECAST =================
@app.route("/forecast")
def forecast():
    if "user" not in session:
        return redirect(url_for("login"))

    if not MODEL_READY:
        return render_template("forecast.html",
                               prediction=None,
                               month=None,
                               error=f"Forecast model not ready: {MODEL_ERROR}",
                               months=[],
                               values=[],
                               temp_value=None,
                               weekend_value=None,
                               holiday_value=None)

    try:
        prediction, month = forecast_engine.run_forecast()
        
        # Get historical data for the chart
        if forecast_engine.monthly is not None:
            historical_data = forecast_engine.monthly.tail(12)
            months = [d.strftime("%b %Y") for d in historical_data.index]
            values = list(historical_data["monthly_demand"])
            
            # Get the actual factor values used in prediction
            last_row = forecast_engine.monthly.iloc[-1]
            
            # Temperature values
            avg_temp = round(last_row.get("avg_temp", 25), 1)
            max_temp = round(last_row.get("max_temp", 32), 1)
            min_temp = round(last_row.get("min_temp", 18), 1)
            
            # Weekend days calculation (weekend_ratio * 30 days)
            weekend_ratio = last_row.get("weekend_ratio", 0.27)
            weekend_days = round(weekend_ratio * 30, 1)
            
            # Holiday effect
            holiday_effect = last_row.get("holiday_effect", 0)
            holiday_status = "Active" if holiday_effect == 1 else "Standard"
            holiday_impact = "+15%" if holiday_effect == 1 else "Normal"
            
            # Additional seasonal info
            season = last_row.get("season_Summer", 0)
            if season == 1:
                season_name = "Summer"
                season_impact = "High demand"
            elif last_row.get("season_Winter", 0) == 1:
                season_name = "Winter"
                season_impact = "Moderate demand"
            elif last_row.get("season_Monsoon", 0) == 1:
                season_name = "Monsoon"
                season_impact = "Lower demand"
            else:
                season_name = "Post-Monsoon"
                season_impact = "Recovery phase"
            
            # Get month name for seasonal context
            current_month = last_row.name.month if hasattr(last_row, 'name') else 1
            month_names = ["January", "February", "March", "April", "May", "June", 
                          "July", "August", "September", "October", "November", "December"]
            current_month_name = month_names[current_month - 1] if 1 <= current_month <= 12 else "Current"
            
        else:
            months = []
            values = []
            avg_temp = 25.0
            max_temp = 32.0
            min_temp = 18.0
            weekend_days = 8.0
            holiday_status = "Standard"
            holiday_impact = "Normal"
            season_name = "Normal"
            season_impact = "Regular"
            current_month_name = "Next Month"
        
        print("DEBUG →", prediction, month)
        
        return render_template("forecast.html", 
                             prediction=prediction, 
                             month=month,
                             months=months,
                             values=values,
                             avg_temp=avg_temp,
                             max_temp=max_temp,
                             min_temp=min_temp,
                             weekend_days=weekend_days,
                             holiday_status=holiday_status,
                             holiday_impact=holiday_impact,
                             season_name=season_name,
                             season_impact=season_impact,
                             current_month_name=current_month_name,
                             error=None)
                             
    except Exception as e:
        print(f"Forecast error: {e}")
        import traceback
        traceback.print_exc()
        return render_template("forecast.html",
                             prediction=None,
                             month=None,
                             error=str(e),
                             months=[],
                             values=[],
                             avg_temp=None,
                             max_temp=None,
                             min_temp=None,
                             weekend_days=None,
                             holiday_status=None,
                             holiday_impact=None,
                             season_name=None,
                             season_impact=None,
                             current_month_name=None)


# ================= HISTORICAL DEMAND API =================
@app.route('/get_historical_demand')
def get_historical_demand():
    """Get historical demand for a specific past month"""
    if "user" not in session:
        return jsonify({"success": False, "message": "Unauthorized"})
    
    month = request.args.get('month', type=int)
    year = request.args.get('year', type=int)
    
    if not month or not year:
        return jsonify({"success": False, "message": "Month and year required"})
    
    try:
        if forecast_engine.monthly is not None:
            # Create target date (first day of the month)
            target_date = pd.Timestamp(year=year, month=month, day=1)
            
            print(f"📅 Looking for historical date: {target_date}")
            print(f"📊 Available date range: {forecast_engine.monthly.index.min()} to {forecast_engine.monthly.index.max()}")
            
            # Try different matching methods
            demand = None
            matched_date = None
            
            # Method 1: Exact match
            if target_date in forecast_engine.monthly.index:
                matched_date = target_date
                demand = forecast_engine.monthly.loc[target_date, 'monthly_demand']
                print(f"✅ Exact match found for {target_date}")
            
            # Method 2: Find by year and month (more flexible)
            else:
                for idx in forecast_engine.monthly.index:
                    if idx.year == year and idx.month == month:
                        matched_date = idx
                        demand = forecast_engine.monthly.loc[idx, 'monthly_demand']
                        print(f"✅ Found by year/month: {idx}")
                        break
            
            # If found, return the data
            if demand is not None and matched_date is not None:
                row = forecast_engine.monthly.loc[matched_date]
                
                avg_temp = round(row.get('avg_temp', 25), 1)
                weekend_ratio = row.get('weekend_ratio', 0.27)
                weekend_days = round(weekend_ratio * 28, 1)
                weekend_days=int(min(max(weekend_days,8),9))
                holiday_effect = row.get('holiday_effect', 0)
                holiday_status = "Active" if holiday_effect == 1 else "Standard"
                
                print(f"📈 Historical demand for {matched_date.strftime('%B %Y')}: {demand:.2f} units")
                
                return jsonify({
                    "success": True,
                    "demand": round(demand, 2),
                    "factors": {
                        "temp": f"{avg_temp}°C",
                        "temp_status": "Historical (Actual)",
                        "weekend": f"{weekend_days} days",
                        "weekend_status": "Historical",
                        "holiday": holiday_status,
                        "holiday_status": "Historical"
                    }
                })
            else:
                # Show available date range for debugging
                min_date = forecast_engine.monthly.index.min()
                max_date = forecast_engine.monthly.index.max()
                
                print(f"❌ No data found for {month}/{year}")
                print(f"📅 Data available from {min_date.strftime('%B %Y')} to {max_date.strftime('%B %Y')}")
                
                return jsonify({
                    "success": False,
                    "message": f"No historical data available for {month}/{year}. Data available from {min_date.strftime('%B %Y')} to {max_date.strftime('%B %Y')}"
                })
        else:
            return jsonify({"success": False, "message": "Model not trained. Please restart the application."})
            
    except Exception as e:
        print(f"❌ Error fetching historical demand: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": str(e)})

# ================= FORECAST DEMAND API =================
# ================= FORECAST DEMAND API =================
# ================= FORECAST DEMAND API =================
@app.route('/get_forecast_demand')
def get_forecast_demand():
    """Get forecasted demand for a specific future month"""
    if "user" not in session:
        return jsonify({"success": False, "message": "Unauthorized"})
    
    month = request.args.get('month', type=int)
    year = request.args.get('year', type=int)
    
    if not month or not year:
        return jsonify({"success": False, "message": "Month and year required"})
    
    try:
        if forecast_engine.monthly is not None:
            # Get the last month from historical data
            last_month = forecast_engine.monthly.index.max()
            print(f"📅 Last month in data: {last_month}")
            
            # Calculate target date
            target_date = pd.Timestamp(year=year, month=month, day=1)
            print(f"🎯 Target date: {target_date}")
            
            # Calculate months difference
            months_ahead = (target_date.year - last_month.year) * 12 + (target_date.month - last_month.month)
            print(f"📊 Months ahead: {months_ahead}")
            
            # Check if it's a past or future month
            if months_ahead <= 0:
                print("⏪ Date is past or current, redirecting to historical API")
                return get_historical_demand()
            
            # Use the predict_specific_month function from forecast_engine
            if hasattr(forecast_engine, 'predict_specific_month'):
                print("🔮 Calling predict_specific_month...")
                prediction = forecast_engine.predict_specific_month(months_ahead)
                print(f"✅ Prediction for {months_ahead} month(s) ahead: {prediction:.2f}")
            else:
                # Fallback to regular forecast with adjustment
                print("⚠️ predict_specific_month not found, using fallback method")
                base_prediction, _ = forecast_engine.run_forecast()
                # Monthly growth factor (2% per month)
                prediction = base_prediction * (1 + (months_ahead * 0.02))
                print(f"📈 Base prediction: {base_prediction:.2f}, Adjusted: {prediction:.2f}")
            
            # Ensure prediction is reasonable (not negative or zero)
            if prediction <= 0:
                print("⚠️ Invalid prediction detected, using fallback")
                base_pred, _ = forecast_engine.run_forecast()
                prediction = base_pred * (1 + (months_ahead * 0.02))
            
            # Seasonal temperature ranges based on month (India context)
            seasonal_temps = {
                1: "18-22°C",   # January - Winter
                2: "20-24°C",   # February - Spring
                3: "24-28°C",   # March - Summer starts
                4: "28-32°C",   # April - Summer peak
                5: "30-35°C",   # May - Hottest
                6: "32-36°C",   # June - Monsoon begins
                7: "30-34°C",   # July - Monsoon
                8: "29-33°C",   # August - Monsoon
                9: "28-32°C",   # September - Post Monsoon
                10: "26-30°C",  # October - Autumn
                11: "22-26°C",  # November - Cool begins
                12: "18-22°C"   # December - Winter
            }
            
            # Holiday/Event impact based on month (India specific)
            seasonal_holidays = {
                1: "New Year/Pongal 🎊",
                2: "Standard",
                3: "Holi 🎨",
                4: "Summer Peak",
                5: "Summer Peak",
                6: "Monsoon Start",
                7: "Monsoon",
                8: "Monsoon",
                9: "Ganesh Chaturthi 🐘",
                10: "Navratri/Dussehra 👑",
                11: "Diwali 🪔",
                12: "Christmas 🎄"
            }
            
            # Weekend days calculation (based on month)
            # Using standard 8 weekend days per month (Saturday + Sunday)
            weekend_days = 8
            
            # Holiday effect multiplier
            holiday_multiplier = {
                1: 1.15,   # January - Pongal/New Year boost
                2: 1.0,
                3: 1.12,   # March - Holi boost
                4: 1.05,
                5: 1.08,   # May - Summer peak
                6: 1.0,
                7: 1.0,
                8: 1.0,
                9: 1.10,   # September - Ganesh Chaturthi
                10: 1.15,  # October - Navratri/Dussehra
                11: 1.20,  # November - Diwali (highest)
                12: 1.18   # December - Christmas
            }
            
            holiday_factor = holiday_multiplier.get(month, 1.0)
            
            # Adjust prediction with holiday factor if needed
            if holiday_factor > 1.0:
                adjusted_prediction = prediction * holiday_factor
                print(f"🎉 Holiday adjustment: {holiday_factor} -> {adjusted_prediction:.2f}")
                prediction = adjusted_prediction
            
            return jsonify({
                "success": True,
                "demand": round(prediction, 2),
                "factors": {
                    "temp": seasonal_temps.get(month, "24-28°C"),
                    "temp_status": "Forecasted (Seasonal average)",
                    "weekend": f"{weekend_days} days",
                    "weekend_status": "Projected (Standard pattern)",
                    "holiday": seasonal_holidays.get(month, "Standard"),
                    "holiday_status": f"Expected (Multiplier: {holiday_factor}x)"
                }
            })
        else:
            print("❌ Model not trained or monthly data is None")
            return jsonify({"success": False, "message": "Model not trained. Please restart the application."})
            
    except Exception as e:
        print(f"❌ Error fetching forecast demand: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback response with estimated demand
        try:
            # Try to get base forecast as fallback
            base_pred, _ = forecast_engine.run_forecast()
            fallback_demand = base_pred * (1 + (months_ahead * 0.02) if 'months_ahead' in locals() else 1)
            return jsonify({
                "success": True,
                "demand": round(fallback_demand, 2),
                "factors": {
                    "temp": "24-28°C",
                    "temp_status": "Estimated",
                    "weekend": "8 days",
                    "weekend_status": "Estimated",
                    "holiday": "Standard",
                    "holiday_status": "Estimated"
                },
                "warning": f"Using estimated values due to error: {str(e)}"
            })
        except:
            return jsonify({"success": False, "message": str(e)})

# ================= SCENARIO FORECAST =================
@app.route("/scenario", methods=["GET", "POST"])
def scenario():
    if "user" not in session:
        return redirect(url_for("login"))

    prediction = None
    base_prediction = None
    change_percent = None
    change_label = ""
    message = ""
    impact_description = ""

    temp_value = ""
    weekend_value = ""
    holiday_value = ""

    if request.method == "POST":

        temp_raw = request.form.get("temp_change", "").strip() 
        weekend_raw = request.form.get("weekend_days", "").strip()
        holiday_raw = request.form.get("holiday_days", "").strip()

        temp_value = temp_raw
        weekend_value = weekend_raw
        holiday_value = holiday_raw

        try:
            # temperature (positive = warmer, increases demand)
            temp_change = float(temp_raw) if temp_raw else 0.0

            # weekend days (convert to ratio)
            weekend_days = int(weekend_raw) if weekend_raw else 8
            if weekend_days < 0 or weekend_days > 31:
                message = "Weekend days must be between 0 and 31"
                return render_template("scenario.html",
                                       prediction=None,
                                       message=message,
                                       temp_value=temp_value,
                                       weekend_value=weekend_value,
                                       holiday_value=holiday_value)

            # holiday days (convert to ratio)
            holiday_days = int(holiday_raw) if holiday_raw else 0
            if holiday_days < 0 or holiday_days > 31:
                message = "Holiday days must be between 0 and 31"
                return render_template("scenario.html",
                                       prediction=None,
                                       message=message,
                                       temp_value=temp_value,
                                       weekend_value=weekend_value,
                                       holiday_value=holiday_value)

            days_in_month = 30

            weekend_ratio = weekend_days / days_in_month
            holiday_ratio = holiday_days / days_in_month

            # ✅ BASE (EXPECTED) DEMAND
            base_prediction, _ = forecast_engine.run_forecast()

            # ✅ SCENARIO DEMAND (Temperature increase = Higher demand)
            prediction = forecast_engine.run_scenario(
                temp_change,
                weekend_ratio,
                holiday_ratio
            )

            # ✅ CHANGE %
            if base_prediction and base_prediction != 0:
                change_percent = ((prediction - base_prediction) / base_prediction) * 100

                if change_percent > 0:
                    change_label = "Higher ⬆"
                    # Generate impact description
                    impacts = []
                    if temp_change > 0:
                        impacts.append(f"warmer by {temp_change}°C")
                    elif temp_change < 0:
                        impacts.append(f"cooler by {abs(temp_change)}°C")
                    
                    if weekend_days > 8:
                        impacts.append(f"{weekend_days - 8} more weekend days")
                    elif weekend_days < 8:
                        impacts.append(f"{8 - weekend_days} fewer weekend days")
                        
                    if holiday_days > 0:
                        impacts.append(f"{holiday_days} holiday days")
                    
                    if impacts:
                        impact_description = f"📈 Demand increased due to: {', '.join(impacts)}"
                    else:
                        impact_description = "📈 Demand increased based on scenario parameters"
                        
                elif change_percent < 0:
                    change_label = "Lower ⬇"
                    impacts = []
                    if temp_change < 0:
                        impacts.append(f"cooler by {abs(temp_change)}°C")
                    if weekend_days < 8:
                        impacts.append(f"{8 - weekend_days} fewer weekend days")
                    if impacts:
                        impact_description = f"📉 Demand decreased due to: {', '.join(impacts)}"
                    else:
                        impact_description = "📉 Demand decreased based on scenario parameters"
                else:
                    change_label = "No Change"
                    impact_description = "➡️ No significant change in demand"

        except ValueError:
            message = "Enter valid numeric values."

    return render_template(
        "scenario.html",
        prediction=round(prediction, 2) if prediction else None,
        base_prediction=round(base_prediction, 2) if base_prediction else None,
        change_percent=round(change_percent, 2) if change_percent is not None else None,
        change_label=change_label,
        message=message,
        temp_value=temp_value,
        weekend_value=weekend_value,
        holiday_value=holiday_value,
        impact_description=impact_description
    )


# ================= ANOMALY DETECTION =================
@app.route("/anomalies")
def anomalies():

    if "user" not in session:
        return redirect(url_for("login"))

    if not MODEL_READY:
        return render_template("anomalies.html",
                               anomalies=[],
                               error=f"Model not ready: {MODEL_ERROR}")

    try:
        anomaly_data = forecast_engine.detect_anomalies()
        
        # Get historical data for chart
        if forecast_engine.monthly is not None:
            historical_data = forecast_engine.monthly.tail(12)
            months = [d.strftime("%b %Y") for d in historical_data.index]
            values = list(historical_data["monthly_demand"])
        else:
            months = []
            values = []

        return render_template("anomalies.html",
                               anomalies=anomaly_data,
                               months=months,
                               values=values,
                               error=None)
    except Exception as e:
        return render_template("anomalies.html",
                               anomalies=[],
                               months=[],
                               values=[],
                               error=str(e))


# ================= ANALYSIS =================
@app.route("/analysis")
def analysis():

    if "user" not in session:
        return redirect(url_for("login"))

    try:
        graphs, summary = forecast_engine.get_demand_analysis()
        kpi = forecast_engine.get_kpi_metrics()

        return render_template("analysis.html",
                               graphs=graphs,
                               summary=summary,
                               kpi=kpi)
    except Exception as e:
        return render_template("analysis.html",
                               graphs={},
                               summary=f"Error loading analysis: {str(e)}",
                               kpi={})

# ================= LOGOUT =================
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)