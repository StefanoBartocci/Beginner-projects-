"""
Stock Price Visualization Dashboard
Built with Streamlit and FactSet data
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np

# Configure the page
st.set_page_config(
    page_title="📈 Stock Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title
st.title("📈 Stock Price Visualization Dashboard")
st.markdown("---")

# Sidebar for controls
st.sidebar.header("Dashboard Controls")

# For now, we'll create a simple demo with sample data
# Later we'll replace this with FactSet integration
st.sidebar.info("🚧 **Demo Mode**: Using sample data. FactSet integration coming next!")

# Sample stock selection
stocks = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
selected_stock = st.sidebar.selectbox("Select Stock", stocks)

# Date range selection
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
with col2:
    end_date = st.date_input("End Date", datetime.now())

# Chart type selection
chart_type = st.sidebar.radio("Chart Type", ["Line Chart", "Candlestick", "Area Chart"])

# Generate sample data for demonstration
@st.cache_data
def generate_sample_data(stock, start, end):
    """Generate sample stock data for demonstration"""
    date_range = pd.date_range(start=start, end=end, freq='D')
    
    # Different seed for each stock to create unique data patterns
    stock_seeds = {"AAPL": 42, "GOOGL": 123, "MSFT": 456, "TSLA": 789, "AMZN": 999}
    seed = stock_seeds.get(stock, 42)
    np.random.seed(seed)
    
    # Different initial prices and volatility for each stock
    stock_params = {
        "AAPL": {"initial_price": 150, "volatility": 0.02, "trend": 0.001},
        "GOOGL": {"initial_price": 2800, "volatility": 0.025, "trend": 0.0005},
        "MSFT": {"initial_price": 350, "volatility": 0.018, "trend": 0.0012},
        "TSLA": {"initial_price": 250, "volatility": 0.04, "trend": 0.002},
        "AMZN": {"initial_price": 3200, "volatility": 0.022, "trend": 0.0008}
    }
    
    params = stock_params.get(stock, {"initial_price": 100, "volatility": 0.02, "trend": 0.001})
    initial_price = params["initial_price"]
    volatility = params["volatility"]
    trend = params["trend"]
    
    # Generate realistic stock price movements
    returns = np.random.normal(trend, volatility, len(date_range))
    prices = [initial_price]
    
    for i in range(1, len(date_range)):
        price = prices[-1] * (1 + returns[i])
        prices.append(price)
    
    # Create OHLC data
    data = []
    for i, date in enumerate(date_range):
        open_price = prices[i] + np.random.normal(0, 0.5)
        close_price = prices[i]
        high_price = max(open_price, close_price) + abs(np.random.normal(0, 1))
        low_price = min(open_price, close_price) - abs(np.random.normal(0, 1))
        volume = np.random.randint(1000000, 10000000)
        
        data.append({
            'Date': date,
            'Open': open_price,
            'High': high_price,
            'Low': low_price,
            'Close': close_price,
            'Volume': volume,
            'Stock': stock
        })
    
    return pd.DataFrame(data)

# Get sample data first to calculate metrics
df = generate_sample_data(selected_stock, start_date, end_date)

# Calculate current metrics from the data
current_price = df['Close'].iloc[-1]
day_high = df['High'].iloc[-1]
day_low = df['Low'].iloc[-1]
volume = df['Volume'].iloc[-1]

# Calculate price change
if len(df) > 1:
    price_change = ((current_price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
else:
    price_change = 0

# Main content area
col1, col2, col3, col4 = st.columns(4)

# Dynamic metrics based on selected stock
with col1:
    st.metric("Current Price", f"${current_price:.2f}", f"{price_change:+.1f}%")
with col2:
    st.metric("Day High", f"${day_high:.2f}")
with col3:
    st.metric("Day Low", f"${day_low:.2f}")
with col4:
    st.metric("Volume", f"{volume/1000000:.1f}M")

# Data is already generated above for metrics

# Create the main chart
st.subheader(f"{selected_stock} Stock Price")

if chart_type == "Line Chart":
    fig = px.line(df, x='Date', y='Close', title=f'{selected_stock} Closing Price')
    fig.update_traces(line_color='#1f77b4', line_width=2)
    
elif chart_type == "Candlestick":
    fig = go.Figure(data=go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name=selected_stock
    ))
    fig.update_layout(title=f'{selected_stock} Candlestick Chart')
    
elif chart_type == "Area Chart":
    fig = px.area(df, x='Date', y='Close', title=f'{selected_stock} Price Area Chart')
    fig.update_traces(fill='tonexty')

# Customize chart appearance
fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Price ($)",
    hovermode='x unified',
    template='plotly_white',
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# Volume chart
st.subheader("Trading Volume")
vol_fig = px.bar(df, x='Date', y='Volume', title='Daily Trading Volume')
vol_fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Volume",
    template='plotly_white',
    height=300
)
st.plotly_chart(vol_fig, use_container_width=True)

# Data table
if st.sidebar.checkbox("Show Raw Data"):
    st.subheader("Raw Data")
    st.dataframe(df.tail(10), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**Next Steps:** Integrate with FactSet API for real stock data!")

# Instructions for next phase
with st.expander("📋 Setup Instructions"):
    st.markdown("""
    **To run this dashboard:**
    
    1. **Install Python packages:**
       ```bash
       pip install -r requirements.txt
       ```
    
    2. **Run the dashboard:**
       ```bash
       streamlit run main.py
       ```
    
    3. **Next Phase:** We'll integrate FactSet API to replace sample data with real stock information.
    """)
