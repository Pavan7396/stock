# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import lime
import lime.lime_tabular

st.set_page_config(page_title="Stock Price Trend Predictor", layout="wide")

st.title("📈 Stock Price Trend Predictor with LIME")

# -----------------------------
# 1️⃣ Generate Synthetic Stock Data
# -----------------------------
np.random.seed(42)
days = 100
price = 100  # starting price
prices = [price]

for _ in range(days - 1):
    # Random walk: daily change between -1 and 1%
    change = prices[-1] * np.random.uniform(-0.01, 0.01)
    prices.append(prices[-1] + change)

dates = pd.date_range(start="2025-01-01", periods=days)
df = pd.DataFrame({"Date": dates, "Price": prices})
st.subheader("Synthetic Stock Prices (Random Walk)")
st.line_chart(df.set_index("Date")["Price"])

# -----------------------------
# 2️⃣ Create Lag Features
# -----------------------------
n_lags = st.slider("Select Number of Past Days to Use as Features (lags)", 1, 10, 5)
for lag in range(1, n_lags + 1):
    df[f"lag_{lag}"] = df["Price"].shift(lag)

df.dropna(inplace=True)

X = df[[f"lag_{lag}" for lag in range(1, n_lags + 1)]].values
y = df["Price"].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# -----------------------------
# 3️⃣ Train Linear Regression Model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# 4️⃣ User Input for Prediction
# -----------------------------
st.subheader("Predict Next Day Stock Price")
st.write("Input the latest prices for the past N days:")

user_input = []
for i in range(n_lags):
    val = st.number_input(f"Price {i+1} day(s) ago", value=float(df[f"lag_{i+1}"].iloc[-1]))
    user_input.append(val)

user_input_array = np.array(user_input).reshape(1, -1)
predicted_price = model.predict(user_input_array)[0]
st.success(f"Predicted Next Day Price: {predicted_price:.2f}")

# -----------------------------
# 5️⃣ LIME Explanation
# -----------------------------
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train,
    feature_names=[f"lag_{lag}" for lag in range(1, n_lags + 1)],
    mode="regression"
)

exp = explainer.explain_instance(
    data_row=user_input_array[0],
    predict_fn=model.predict,
    num_features=n_lags
)

# Convert to DataFrame for plotting
feat, weight = zip(*exp.as_list())
lime_df = pd.DataFrame({"Feature": feat, "Weight": weight})

st.subheader("LIME Feature Contributions")
st.dataframe(lime_df)

# Plot
fig, ax = plt.subplots()
ax.barh(lime_df['Feature'], lime_df['Weight'], color="skyblue")
ax.set_xlabel("Weight (Impact on Prediction)")
ax.set_title("LIME Explanation for Next Day Price Prediction")
st.pyplot(fig)
