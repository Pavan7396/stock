# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import lime
import lime.lime_tabular

# --------------------------
# Page Config
# --------------------------
st.set_page_config(page_title="Price Prediction App", layout="wide")

# --------------------------
# Custom CSS
# --------------------------
st.markdown("""
    <style>
    body {
        background-color: #f5f5f5;
    }
    .stApp {
        font-family: 'Arial', sans-serif;
    }
    h1 {
        color: #2E86AB;
        text-align: center;
    }
    .pred-box {
        background-color: #E8F6EF;
        padding: 15px;
        border-radius: 10px;
        margin-top: 10px;
        text-align: center;
        font-weight: bold;
        color: #117A65;
        font-size: 20px;
    }
    .lime-table th {
        background-color: #2E86AB;
        color: white;
        text-align: center;
    }
    .lime-table td {
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --------------------------
# Title
# --------------------------
st.markdown("<h1>Time Series Price Prediction with LIME</h1>", unsafe_allow_html=True)

# --------------------------
# 1. Generate synthetic data
# --------------------------
np.random.seed(42)
n_days = 100
noise = np.random.normal(0, 1, n_days)
prices = 100 + np.cumsum(noise)
dates = pd.date_range("2025-01-01", periods=n_days)
df = pd.DataFrame({"Date": dates, "Price": prices}).set_index("Date")

# --------------------------
# 2. Create lag features
# --------------------------
n_lags = 5
for lag in range(1, n_lags+1):
    df[f"lag_{lag}"] = df["Price"].shift(lag)
df = df.dropna().copy()
X = df[[f"lag_{i}" for i in range(1, n_lags+1)]]
y = df["Price"]

# --------------------------
# 3. Train Linear Regression
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = LinearRegression()
model.fit(X_train, y_train)

# --------------------------
# 4. LIME explainer
# --------------------------
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    mode="regression",
    discretize_continuous=False
)

def lime_exp_to_feat_weight(exp, feature_names):
    feats = {name: 0.0 for name in feature_names}
    for desc, weight in exp.as_list():
        matched = None
        for fname in feature_names:
            if fname in desc:
                matched = fname
                break
        if matched is None:
            key = desc.split()[0]
            matched = key if key in feature_names else None
        if matched:
            feats[matched] = weight
    return feats

# --------------------------
# Sidebar Inputs
# --------------------------
st.sidebar.header("Enter Last 5 Lag Values")
lag_inputs = []
for i in range(1, n_lags+1):
    val = st.sidebar.number_input(f"Lag {i}", value=float(X.iloc[-1][f"lag_{i}"]))
    lag_inputs.append(val)

predict_button = st.sidebar.button("Predict Next Day")
forecast_steps = st.sidebar.number_input("7-Day Forecast Steps", min_value=1, max_value=30, value=7)

# --------------------------
# Main Price Chart
# --------------------------
st.subheader("Price Time Series")
st.line_chart(df["Price"])

# --------------------------
# Prediction + LIME
# --------------------------
if predict_button:
    X_input = np.array(lag_inputs).reshape(1, -1)
    next_pred = model.predict(X_input)[0]
    st.markdown(f"<div class='pred-box'>Predicted Next-Day Price: {next_pred:.4f}</div>", unsafe_allow_html=True)
    
    exp_next = explainer.explain_instance(X_input.ravel(), model.predict, num_features=n_lags)
    next_weights = lime_exp_to_feat_weight(exp_next, X_train.columns.tolist())
    
    st.subheader("LIME Feature Contributions")
    df_weights = pd.DataFrame.from_dict(next_weights, orient='index', columns=['Weight'])
    st.markdown(df_weights.to_html(classes='lime-table'), unsafe_allow_html=True)
    
    # Bar chart
    fig, ax = plt.subplots()
    ax.bar(next_weights.keys(), next_weights.values(), color="#117A65")
    ax.set_ylabel("Contribution to Prediction")
    ax.set_title("LIME Feature Contribution")
    st.pyplot(fig)

# --------------------------
# 7-Day Iterative Forecast
# --------------------------
def iterative_forecast(model, last_known_series, n_steps=7, n_lags=5):
    preds = []
    current_series = list(last_known_series[-n_lags:])
    for s in range(n_steps):
        x = np.array(current_series[-n_lags:]).reshape(1, -1)
        yhat = model.predict(x)[0]
        preds.append(yhat)
        current_series.append(yhat)
    return preds

if st.sidebar.button("Run 7-Day Iterative Forecast"):
    last_known_prices = df["Price"].values
    future_preds = iterative_forecast(model, last_known_prices, n_steps=forecast_steps, n_lags=n_lags)
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_steps)
    forecast_df = pd.DataFrame({"Date": future_dates, "Predicted_Price": future_preds}).set_index("Date")
    
    st.subheader(f"{forecast_steps}-Day Iterative Forecast")
    st.line_chart(forecast_df)
    st.table(forecast_df)
