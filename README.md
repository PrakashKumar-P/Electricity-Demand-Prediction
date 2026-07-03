  # Electricity Demand Prediction — Ensemble Learning Framework

An end-to-end machine learning system that predicts household electricity consumption, built with a Flask web app for real-time prediction and visualization.

## Overview

This project uses an ensemble learning approach to forecast electricity demand from historical usage data. It includes a full pipeline — from data preprocessing and feature engineering to model training, evaluation, and deployment behind a live web interface.

## Results

- **Prediction Accuracy:** 86.4%
- **MAPE (Mean Absolute Percentage Error):** 13.6%
- **MAE (Mean Absolute Error):** 55,181.38
- **RMSE (Root Mean Squared Error):** 55,612.94

Feature importance and model explainability were analyzed using **SHAP**, to understand which factors most influence demand predictions.

## Tech Stack

- **Language:** Python
- **ML/Data:** Scikit-learn, Pandas, NumPy, SHAP
- **Web Framework:** Flask
- **Frontend:** HTML/CSS (for the prediction dashboard)

## Features

- Data preprocessing and feature engineering pipeline for electricity usage data
- Ensemble model combining multiple algorithms for improved prediction accuracy
- SHAP-based feature importance visualization for model interpretability
- Flask web application for real-time prediction input and output
- Visualization of predicted vs. actual demand trends

## How It Works

1. Raw electricity usage data is cleaned and preprocessed
2. Features are engineered (e.g., time-based patterns, historical usage trends)
3. An ensemble model is trained and evaluated against baseline models
4. The trained model is served through a Flask API
5. Users interact with a simple web interface to get real-time demand predictions

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/electricity-demand-prediction.git
cd electricity-demand-prediction

# Install dependencies
pip install -r requirements.txt

# Run the Flask app
python app.py
```

Then open `http://localhost:5000` in your browser.

## Project Structure

```
electricity-demand-prediction/
├── app.py                 # Flask application entry point
├── model/                 # Trained model files
├── notebooks/              # Data exploration & model development
├── static/                 # CSS/JS for the web app
├── templates/               # HTML templates
├── requirements.txt
└── README.md
```

## Future Improvements

- Incorporate weather data as an additional feature
- Experiment with deep learning models (LSTM) for time-series forecasting
- Add model retraining pipeline for continuous improvement

## Author

**Prakash Kumar P**
[LinkedIn](https://linkedin.com/in/prakash-kumar-p-61a321413) · prakashkumar.20it@gmail.com
