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

# Aggregate historical data by quarter
df_historical = df.copy()
df_historical["Quarter"] = df_historical["ds"].dt.to_period("Q")
df_historical = df_historical.groupby(["Quarter", "Product"])["y"].sum().reset_index()

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
st.title("Predictive Sales for Mauj Jim")

# Product selection
selected_product = st.selectbox("Select Product:", products)

# Predict future quantities for January 2025
future_dates = pd.date_range(start="2025-01-01", end="2025-01-31")
predictions = {}
for product, model in models.items():
  future = model.make_future_dataframe(periods=len(future_dates))
  forecast = model.predict(future)
  predictions[product] = forecast["yhat"].values

# Create Plotly figure for historical data
fig_historical = go.Figure()
for product in products:
    product_data = df_historical[df_historical["Product"] == product]
    fig_historical.add_trace(
        go.Bar(
            x=product_data["Quarter"].astype(str),  # Convert Period to string for x-axis
            y=product_data["y"],
            name=product,
        )
    )
fig_historical.update_layout(title="Historical Sales by Quarter (2022-2024)",
                            xaxis_title="Quarter",
                            yaxis_title="Total Sales")

# Display the historical chart in Streamlit
st.plotly_chart(fig_historical)

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
