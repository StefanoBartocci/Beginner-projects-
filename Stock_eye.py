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
import yfinance as yf

# Configure the page
st.set_page_config(
    page_title="Financial Markets Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title
st.title("Financial Markets Dashboard")
st.markdown("---")

# Sidebar for controls
st.sidebar.header("Dashboard Controls")

# Financial dashboard controls

# Financial instruments selection
instruments = {
    "Technology Stocks": ["AAPL", "GOOGL", "MSFT", "NVDA", "META", "NFLX", "AMD", "CRM", "ORCL", "ADBE"],
    "Popular Stocks": ["TSLA", "AMZN", "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA", "DIS"],
    "Commodities": ["GLD", "SLV", "USO", "UNG", "DBA", "CORN", "WEAT", "SOYB", "PDBC", "PALL"],
    "Bonds & Fixed Income": ["TLT", "IEF", "SHY", "TIPS", "LQD", "HYG", "AGG", "BND", "VGIT", "VCIT"],
    "Crypto ETFs": ["BITO", "ETHE", "GBTC", "COIN", "MSTR", "RIOT", "MARA"],
    "Currency ETFs": ["UUP", "FXE", "FXY", "FXB", "EWJ", "EWZ", "EWU", "EWC"]
}

# Create dropdown for category selection
category = st.sidebar.selectbox("Select Category", list(instruments.keys()))
selected_instrument = st.sidebar.selectbox("Select Instrument", instruments[category])

# Add search functionality
st.sidebar.markdown("---")
st.sidebar.markdown("**Or search for any symbol:**")
custom_symbol = st.sidebar.text_input("Enter Symbol (e.g., BTC-USD, EURUSD=X)", "").upper()

# Use custom symbol if provided, otherwise use selected instrument
if custom_symbol:
    selected_stock = custom_symbol
    st.sidebar.info(f"Using custom symbol: {custom_symbol}")
else:
    selected_stock = selected_instrument

# Date range selection
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
with col2:
    end_date = st.date_input("End Date", datetime.now())

# Chart type selection
chart_type = st.sidebar.radio("Chart Type", ["Line Chart", "Candlestick", "Area Chart"])

# Display options
st.sidebar.markdown("---")
st.sidebar.markdown("**Display Options:**")
show_raw_data = st.sidebar.checkbox("Show Raw Data", value=False)
show_market_news = st.sidebar.checkbox("Show Market News Tab", value=True)

# Fetch real stock data from Yahoo Finance
@st.cache_data
def get_stock_data(symbol, start_date, end_date):
    """Fetch real stock data from Yahoo Finance"""
    try:
        # Create ticker object
        ticker = yf.Ticker(symbol)
        
        # Get historical data
        df = ticker.history(start=start_date, end=end_date)
        
        # Reset index to get Date as a column
        df = df.reset_index()
        
        # Rename columns to match our format
        df = df.rename(columns={
            'Date': 'Date',
            'Open': 'Open', 
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume'
        })
        
        # Add stock symbol column
        df['Stock'] = symbol
        
        # Remove timezone info if present
        if df['Date'].dt.tz is not None:
            df['Date'] = df['Date'].dt.tz_localize(None)
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return pd.DataFrame()  # Return empty dataframe on error

# Fetch company info and financial ratios
@st.cache_data
def get_company_info(symbol):
    """Fetch company information and financial ratios"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Extract key financial metrics
        company_data = {
            'company_name': info.get('longName', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 'N/A'),
            'forward_pe': info.get('forwardPE', 'N/A'),
            'peg_ratio': info.get('pegRatio', 'N/A'),
            'price_to_book': info.get('priceToBook', 'N/A'),
            'dividend_yield': info.get('dividendYield', 'N/A'),
            'beta': info.get('beta', 'N/A'),
            'eps': info.get('trailingEps', 'N/A'),
            'revenue': info.get('totalRevenue', 0),
            'profit_margin': info.get('profitMargins', 'N/A'),
            'debt_to_equity': info.get('debtToEquity', 'N/A'),
            'return_on_equity': info.get('returnOnEquity', 'N/A'),
            'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 'N/A'),
            'fifty_two_week_low': info.get('fiftyTwoWeekLow', 'N/A')
        }
        
        return company_data
        
    except Exception as e:
        st.error(f"Error fetching company info for {symbol}: {str(e)}")
        return {}

# Fetch general financial market news with fallback
@st.cache_data
def get_general_market_news():
    """Fetch general financial market news from major indices and market symbols"""
    all_news = []
    
    # Major market symbols to get broad financial news (prioritized order)
    market_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'NVDA', 'SPY', 'QQQ']
    
    for symbol in market_symbols:
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            if news:  # Check if news exists
                for article in news[:3]:  # Get fewer from each to have variety
                    title = article.get('title', '')
                    summary = article.get('summary', '')
                    publisher = article.get('publisher', '')
                    link = article.get('link', '')
                    
                    # Only include articles with actual content
                    if title and title != 'No title' and len(title) > 10:
                        formatted_article = {
                            'title': title,
                            'publisher': publisher if publisher else 'Financial News',
                            'link': link if link else '#',
                            'publish_time': article.get('providerPublishTime', 0),
                            'summary': summary[:250] + '...' if len(summary) > 250 else summary if summary else f"Latest update on {symbol}",
                            'source_symbol': symbol
                        }
                        all_news.append(formatted_article)
        except Exception as e:
            continue
    
    # If we got some news, process it
    if all_news:
        # Remove duplicates based on title and sort by publish time
        seen_titles = set()
        unique_news = []
        for article in all_news:
            if article['title'] not in seen_titles:
                seen_titles.add(article['title'])
                unique_news.append(article)
        
        # Sort by publish time (newest first)
        unique_news.sort(key=lambda x: x['publish_time'], reverse=True)
        return unique_news[:15]  # Return top 15 unique articles
    
    # Fallback: return some sample market insights if no news available
    return get_fallback_market_content()

def get_fallback_market_content():
    """Provide fallback market content when news API is unavailable"""
    fallback_content = [
        {
            'title': 'Market Analysis: Focus on Your Selected Instruments',
            'publisher': 'Dashboard Insights',
            'link': '#',
            'publish_time': int(datetime.now().timestamp()),
            'summary': 'Use the Analysis tab to examine price trends, financial ratios, and technical indicators for your selected instruments. The dashboard provides comprehensive data for stocks, bonds, commodities, and ETFs.',
            'source_symbol': 'INFO'
        },
        {
            'title': 'Investment Tip: Diversification Across Asset Classes',
            'publisher': 'Financial Education',
            'link': '#',
            'publish_time': int(datetime.now().timestamp()) - 3600,
            'summary': 'Consider exploring different categories available in this dashboard: Technology stocks for growth, bonds for stability, commodities for inflation protection, and international ETFs for geographic diversification.',
            'source_symbol': 'TIP'
        },
        {
            'title': 'Dashboard Feature: Compare Multiple Instruments',
            'publisher': 'User Guide',
            'link': '#',
            'publish_time': int(datetime.now().timestamp()) - 7200,
            'summary': 'Switch between different financial instruments to compare their performance. Use the date range controls to analyze historical trends and the chart types to visualize data in different ways.',
            'source_symbol': 'GUIDE'
        }
    ]
    return fallback_content

# Get real stock data and company info from Yahoo Finance
with st.spinner(f'Loading {selected_stock} data...'):
    df = get_stock_data(selected_stock, start_date, end_date)
    company_info = get_company_info(selected_stock)

# Check if data was successfully loaded
if df.empty:
    st.error("Failed to load stock data. Please try again.")
    st.stop()

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

# Company header
if company_info:
    st.subheader(f"{company_info.get('company_name', selected_stock)} ({selected_stock})")
    if company_info.get('sector') != 'N/A':
        st.caption(f"**Sector:** {company_info.get('sector')} | **Industry:** {company_info.get('industry')}")

# Price metrics
st.markdown("### Price Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Current Price", f"${current_price:.2f}", f"{price_change:+.1f}%")
with col2:
    st.metric("Day High", f"${day_high:.2f}")
with col3:
    st.metric("Day Low", f"${day_low:.2f}")
with col4:
    st.metric("Volume", f"{volume/1000000:.1f}M")

# Financial ratios section
if company_info:
    st.markdown("### Financial Ratios")
    
    # Format market cap
    market_cap = company_info.get('market_cap', 0)
    if market_cap > 1e12:
        market_cap_str = f"${market_cap/1e12:.2f}T"
    elif market_cap > 1e9:
        market_cap_str = f"${market_cap/1e9:.2f}B"
    elif market_cap > 1e6:
        market_cap_str = f"${market_cap/1e6:.2f}M"
    else:
        market_cap_str = f"${market_cap:,.0f}" if market_cap > 0 else "N/A"
    
    # Format revenue
    revenue = company_info.get('revenue', 0)
    if revenue > 1e12:
        revenue_str = f"${revenue/1e12:.2f}T"
    elif revenue > 1e9:
        revenue_str = f"${revenue/1e9:.2f}B"
    elif revenue > 1e6:
        revenue_str = f"${revenue/1e6:.2f}M"
    else:
        revenue_str = f"${revenue:,.0f}" if revenue > 0 else "N/A"
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        pe_ratio = company_info.get('pe_ratio', 'N/A')
        pe_display = f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else "N/A"
        st.metric("P/E Ratio", pe_display)
    
    with col2:
        market_cap_display = market_cap_str
        st.metric("Market Cap", market_cap_display)
    
    with col3:
        beta = company_info.get('beta', 'N/A')
        beta_display = f"{beta:.2f}" if isinstance(beta, (int, float)) else "N/A"
        st.metric("Beta", beta_display)
    
    with col4:
        eps = company_info.get('eps', 'N/A')
        eps_display = f"${eps:.2f}" if isinstance(eps, (int, float)) else "N/A"
        st.metric("EPS", eps_display)
    
    with col5:
        dividend_yield = company_info.get('dividend_yield', 'N/A')
        if isinstance(dividend_yield, (int, float)):
            div_display = f"{dividend_yield*100:.2f}%"
        else:
            div_display = "N/A"
        st.metric("Dividend Yield", div_display)
    
    # Additional ratios row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        pb_ratio = company_info.get('price_to_book', 'N/A')
        pb_display = f"{pb_ratio:.2f}" if isinstance(pb_ratio, (int, float)) else "N/A"
        st.metric("P/B Ratio", pb_display)
    
    with col2:
        st.metric("Revenue (TTM)", revenue_str)
    
    with col3:
        profit_margin = company_info.get('profit_margin', 'N/A')
        if isinstance(profit_margin, (int, float)):
            margin_display = f"{profit_margin*100:.2f}%"
        else:
            margin_display = "N/A"
        st.metric("Profit Margin", margin_display)
    
    with col4:
        high_52w = company_info.get('fifty_two_week_high', 'N/A')
        high_display = f"${high_52w:.2f}" if isinstance(high_52w, (int, float)) else "N/A"
        st.metric("52W High", high_display)
    
    with col5:
        low_52w = company_info.get('fifty_two_week_low', 'N/A')
        low_display = f"${low_52w:.2f}" if isinstance(low_52w, (int, float)) else "N/A"
        st.metric("52W Low", low_display)

# Data is already generated above for metrics

# Create the main chart
st.markdown("### Price Chart")

if chart_type == "Line Chart":
    fig = px.line(df, x='Date', y='Close', title=f'{selected_stock} Price History')
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
    fig.update_layout(title=f'{selected_stock} OHLC Chart')
    
elif chart_type == "Area Chart":
    fig = px.area(df, x='Date', y='Close', title=f'{selected_stock} Price Trend')
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

# Create tabs based on user selection
if show_market_news:
    tab1, tab2 = st.tabs(["📊 Analysis", "📰 Market News"])
else:
    tab1 = st.container()

with tab1:
    # Data table (moved here from sidebar)
    if show_raw_data:
        st.subheader("Raw Data")
        st.dataframe(df.tail(10), use_container_width=True)

if show_market_news:
    with tab2:
        # General market news section
        st.markdown("### Financial Market News")
        
        # Load general market news
        with st.spinner('Loading market news...'):
            market_news = get_general_market_news()
        
        if market_news and len(market_news) > 0:
            # Show news count and refresh option
            col1, col2 = st.columns([3, 1])
            with col1:
                st.caption(f"Showing {len(market_news)} latest articles from major financial sources")
            with col2:
                if st.button("🔄 Refresh News"):
                    st.cache_data.clear()
                    st.rerun()
            
            st.markdown("---")
            
            # Create two columns for news layout
            col1, col2 = st.columns(2)
            
            for i, article in enumerate(market_news):
                # Alternate between columns
                with col1 if i % 2 == 0 else col2:
                    with st.container():
                        st.markdown(f"**[{article['title']}]({article['link']})**")
                        publish_date = datetime.fromtimestamp(article['publish_time']).strftime('%Y-%m-%d %H:%M') if article['publish_time'] else 'Recent'
                        st.caption(f"**{article['publisher']}** | {publish_date}")
                        st.markdown(f"{article['summary']}")
                        st.markdown("---")
        else:
            st.warning("📰 Unable to load live market news at the moment.")
            st.info("This can happen due to API rate limits or connectivity issues. Try refreshing the page or check back later.")

# Footer
st.markdown("---")
st.markdown("**Financial Markets Dashboard** - Real-time data for Stocks, Bonds, Commodities & More!")

# Instructions for next phase
with st.expander("Setup Instructions"):
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
    
    3. **Live Data:** Now using real Yahoo Finance data with live stock prices!
    """)
