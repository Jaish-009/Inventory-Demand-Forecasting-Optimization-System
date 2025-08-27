# Streamlit Interactive Dashboard
# Save as: inventory_dashboard.py
# Run with: streamlit run inventory_dashboard.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import streamlit as st

# -------------------------------
# Step 1: Title
st.title("Inventory Demand Forecasting & Optimization System")

# -------------------------------
# Step 2: Generate Sample Data
dates = pd.date_range(start='2024-01-01', periods=100)
products = ['Product A', 'Product B']
data = []

np.random.seed(42)
for product in products:
    sales = np.random.randint(20, 100, size=len(dates))
    for date, sale in zip(dates, sales):
        data.append([date, product, sale])

df = pd.DataFrame(data, columns=['Date', 'Product', 'Sales'])

# -------------------------------
# Step 3: Product Selection
product_choice = st.selectbox("Select Product:", products)
df_product = df[df['Product'] == product_choice][['Date', 'Sales']].copy()
df_product = df_product.rename(columns={'Date':'ds', 'Sales':'y'})

st.subheader("Historical Sales Data")
st.dataframe(df_product.tail(10))

# -------------------------------
# Step 4: Forecasting with Prophet
model = Prophet()
model.fit(df_product)
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

st.subheader("Forecasted Sales for Next 30 Days")
st.dataframe(forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(10))

# -------------------------------
# Step 5: Inventory Optimization
lead_time_days = st.number_input("Supplier Lead Time (days):", min_value=1, value=7)
service_level = st.number_input("Service Level Z-score (e.g., 1.65 for 95%):", min_value=0.0, value=1.65)

std_dev_daily = df_product['y'].std()
avg_daily_sales = df_product['y'].mean()
safety_stock = service_level * std_dev_daily * np.sqrt(lead_time_days)
reorder_point = avg_daily_sales * lead_time_days + safety_stock

st.subheader("Inventory Metrics")
st.write(f"Average Daily Sales: {avg_daily_sales:.2f}")
st.write(f"Safety Stock: {safety_stock:.2f}")
st.write(f"Reorder Point: {reorder_point:.2f}")

# -------------------------------
# Step 6: Visualization
st.subheader("Sales & Forecast Visualization")
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(df_product['ds'], df_product['y'], label='Actual Sales')
ax.plot(forecast['ds'], forecast['yhat'], label='Forecasted Sales', color='red')
ax.axhline(reorder_point, color='green', linestyle='--', label='Reorder Point')
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.legend()
st.pyplot(fig)

# -------------------------------
st.success("Interactive Inventory Forecasting Dashboard Ready!")
