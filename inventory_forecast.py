# Inventory Demand Forecasting & Optimization System
# Python 3.x

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

# Step 2: Generate Sample Sales Data
dates = pd.date_range(start='2024-01-01', periods=100)
products = ['Product A', 'Product B']
data = []

np.random.seed(42)
for product in products:
    sales = np.random.randint(20, 100, size=len(dates))
    for date, sale in zip(dates, sales):
        data.append([date, product, sale])

df = pd.DataFrame(data, columns=['Date', 'Product', 'Sales'])
df.to_csv('sales_data.csv', index=False)
print("Sample sales data created:\n", df.head())

# Step 3: Forecast Demand for Product A
df_product = df[df['Product'] == 'Product A'][['Date', 'Sales']].copy()
df_product = df_product.rename(columns={'Date':'ds', 'Sales':'y'})

model = Prophet()
model.fit(df_product)

# Forecast next 30 days
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Step 4: Inventory Optimization
lead_time_days = 7  # Supplier lead time
service_level = 1.65  # Z-score for 95% service level
std_dev_daily = df_product['y'].std()
avg_daily_sales = df_product['y'].mean()

safety_stock = service_level * std_dev_daily * np.sqrt(lead_time_days)
reorder_point = avg_daily_sales * lead_time_days + safety_stock

print(f"\nAverage Daily Sales: {avg_daily_sales:.2f}")
print(f"Safety Stock: {safety_stock:.2f}")
print(f"Reorder Point: {reorder_point:.2f}")

# Step 5: Visualization
plt.figure(figsize=(10,5))
plt.plot(df_product['ds'], df_product['y'], label='Actual Sales')
plt.plot(forecast['ds'], forecast['yhat'], label='Forecasted Sales', color='red')
plt.axhline(reorder_point, color='green', linestyle='--', label='Reorder Point')
plt.title("Inventory & Forecast Visualization")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.show()
