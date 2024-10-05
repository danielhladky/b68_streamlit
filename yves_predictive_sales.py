import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
import numpy as np

# Generate synthetic sales data
def generate_sales_data(start_date, end_date, products):
  data = []
  for date in pd.date_range(start_date, end_date):
    for product in products:
      quantity = np.random.randint(0, 10)  # Adjust quantity range as needed
      data.append([date, product, quantity])
  df = pd.DataFrame(data, columns=["ds", "Product", "y"])
  return df

# Define products and timeframe
products = ["Maui Jim Keha", "Maui Jim Pailolo"]
start_date = "2022-01-01"
end_date = "2024-12-31"

# Generate synthetic data
df = generate_sales_data(start_date, end_date, products)

# Create Prophet models for each product
models = {}
for product in products:
  product_df = df[df["Product"] == product]
  model = Prophet()
  model.fit(product_df)
  models[product] = model

# Predict future quantities for January 2025
future_dates = pd.date_range(start="2025-01-01", end="2025-01-31")
predictions = {}
for product, model in models.items():
  future = model.make_future_dataframe(periods=len(future_dates))
  forecast = model.predict(future)
  predictions[product] = forecast["yhat"].values

# Streamlit app
st.title("Sales Forecasting App")

# Product selection
selected_product = st.selectbox("Select Product:", products)

# Predict future quantities for January 2025
future_dates = pd.date_range(start="2025-01-01", end="2025-01-31")
predictions = {}
for product, model in models.items():
  future = model.make_future_dataframe(periods=len(future_dates))
  forecast = model.predict(future)
  predictions[product] = forecast["yhat"].values

# Create Plotly figure for the selected product
fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=future_dates,
        y=predictions[selected_product],
        name=selected_product,
    )
)
fig.update_layout(title=f"Predicted Sales for {selected_product} in January 2025")

# Display the figure in Streamlit
st.plotly_chart(fig)
