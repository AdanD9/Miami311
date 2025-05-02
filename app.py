import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import catboost
import requests
import os
from datetime import datetime, timedelta

# Set page config for the app
st.set_page_config(
    page_title="Miami 311 Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS to improve the app appearance
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e8df5;
        color: white;
    }
    h1, h2, h3 {
        padding-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the cleaned Miami 311 data"""
    file_path = "miami311_clean.parquet"
    
    # Check if file exists locally
    if not os.path.exists(file_path):
        # This is your Google Drive file ID - just the ID part from the URL
        file_id = "1J6XGdlEc2P3xacZmNmlKo5oDWSTvuJl8"
        
        try:
            with st.spinner("Downloading data file (this may take a while)..."):
                import gdown
                url = f"https://drive.google.com/uc?id={file_id}"
                gdown.download(url, file_path, quiet=False)
                st.success("Download completed!")
        except Exception as e:
            st.error(f"Error downloading file: {e}")
            st.stop()
    
    # Load based on file extension
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    else:
        return pd.read_parquet(file_path)

@st.cache_resource
def load_model():
    """Load the CatBoost forecasting model"""
    file_path = "miami311_catboost_model.cbm"
    
    # Check if file exists locally
    if not os.path.exists(file_path):
        # Your Google Drive file ID - you need to replace this with the actual ID after uploading
        file_id = "1S8GwHo4Ks3pNMCKiIsNYvA0PCzAddxSo"
        
        try:
            with st.spinner("Downloading model file..."):
                import gdown
                url = f"https://drive.google.com/uc?id={file_id}"
                gdown.download(url, file_path, quiet=False)
                st.success("Model download completed!")
        except Exception as e:
            st.error(f"Error downloading model file: {e}")
            st.stop()
    
    try:
        model = catboost.CatBoostRegressor(task_type="CPU", devices="-1")
        model.load_model(file_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Create a function to prepare monthly panel data
@st.cache_data
def create_monthly_panel(df):
    """Create a monthly panel of service request volumes by zip code and issue type"""
    panel = (
        df
        .assign(month=df["ticket_created_date_time"].dt.to_period("M").dt.to_timestamp())
        .groupby(["zip_code", "issue_type", "month"], observed=True)
        .size()
        .rename("volume")
        .reset_index()
        .sort_values(["zip_code", "issue_type", "month"])
    )
    
    # Add lags (previous 1, 3, 6, 12 months)
    for k in (1, 3, 6, 12):
        panel[f'vol_lag_{k}'] = panel.groupby(["zip_code", "issue_type"])["volume"].shift(k)
        
    # Add rolling mean
    panel['vol_roll_mean_3'] = (
        panel.groupby(["zip_code", "issue_type"])["volume"]
        .transform(lambda s: s.rolling(3, min_periods=1).mean())
    )
    
    # Add percentage changes
    panel['vol_pct_change_1'] = (panel['volume'] - panel['vol_lag_1']) / panel['vol_lag_1']
    panel['vol_pct_change_12'] = (panel['volume'] - panel['vol_lag_12']) / panel['vol_lag_12']
    
    # Add month-related features
    panel['month_num'] = panel['month'].dt.month
    panel['sin_month'] = np.sin(2 * np.pi * panel['month_num'] / 12)
    panel['cos_month'] = np.cos(2 * np.pi * panel['month_num'] / 12)
    
    return panel

monthly_panel = create_monthly_panel(load_data())
data = monthly_panel.copy()
model = load_model()  

raw_df = load_data()
monthly_panel = create_monthly_panel(raw_df)
data = raw_df

def last_row(panel, z, it):
    row = (panel[(panel.zip_code==z) & (panel.issue_type==it)]
                  .sort_values("month")
                  .tail(1))
    return row.iloc[0] if not row.empty else None  

# Create tabs for the app
tab1, tab2 = st.tabs(["📊 Dashboard", "🔮 Forecasting"])

# Tab 1: Dashboard
with tab1:
    st.title('Miami-Dade County 311 Service Requests Dashboard')
    
    # Display basic statistics
    st.header('Overview')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Requests", f"{len(data):,}")
    with col2:
        st.metric("Date Range", f"{data['ticket_created_date_time'].min().strftime('%b %Y')} - {data['ticket_created_date_time'].max().strftime('%b %Y')}")
    with col3:
        st.metric("Unique Issue Types", f"{data['issue_type'].nunique():,}")
    
    # Show requests over time
    st.header('Service Requests Over Time')
    
    # Monthly trend
    monthly_counts = data.resample('M', on='ticket_created_date_time').size()
    fig = px.line(monthly_counts, labels={'value': 'Count', 'index': 'Date'})
    fig.update_layout(title='Monthly Service Requests', height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Two columns for more insights
    col1, col2 = st.columns(2)
    
    with col1:
        # Top issue types
        st.subheader('Top 10 Request Types')
        top_issues = data['issue_type'].value_counts().head(10)
        fig = px.bar(top_issues, labels={'value': 'Count', 'index': 'Issue Type'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Requests by district
        st.subheader('Requests by District')
        district_counts = data['neighborhood_district'].value_counts()
        fig = px.pie(district_counts, values=district_counts.values, names=district_counts.index, hole=0.4)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Map of requests
    st.header('Geographic Distribution')
    
    # Create a sample for the map to avoid overloading
    map_sample = data.sample(min(10000, len(data))).copy()
    
    fig = px.scatter_mapbox(
        map_sample, 
        lat='latitude', 
        lon='longitude',
        color='issue_type',
        zoom=9,
        mapbox_style='carto-positron',
        opacity=0.6,
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Resolution time analysis
    st.header('Resolution Time Analysis')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Average resolution time by issue type
        avg_resolution = data.groupby('issue_type')['actual_completed_days'].mean().sort_values(ascending=False).head(10)
        fig = px.bar(avg_resolution, labels={'value': 'Avg. Days to Close', 'index': 'Issue Type'})
        fig.update_layout(height=400, title='Top 10 Issue Types by Avg. Resolution Time')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # SLA breach rate by issue type
        breach_rate = data.groupby('issue_type')['sla_breached'].mean().sort_values(ascending=False).head(10) * 100
        fig = px.bar(breach_rate, labels={'value': 'SLA Breach Rate (%)', 'index': 'Issue Type'})
        fig.update_layout(height=400, title='Top 10 Issue Types by SLA Breach Rate')
        st.plotly_chart(fig, use_container_width=True)

# Tab 2: Forecasting
with tab2:
    st.title('311 Service Request Volume Forecasting')
    
    st.info("""
    This tool predicts the number of 311 service requests expected for a given zip code, issue type, and month.
    The model is based on historical patterns and trend analysis.
    """)
    
    # Create the selection inputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Get unique zip codes sorted
        zip_codes = sorted(monthly_panel['zip_code'].unique())
        selected_zip = st.selectbox('Select ZIP Code', zip_codes)
    
    with col2:
        # Get issue types for the selected zip code
        issue_types = sorted(monthly_panel[monthly_panel['zip_code'] == selected_zip]['issue_type'].unique())
        selected_issue = st.selectbox('Select Issue Type', issue_types)
    
    with col3:
        # Date selector (starting from Jan 2024 for 48 months)
        start_month = pd.Timestamp("2024-01-01")
        next_months = [(start_month + pd.DateOffset(months=i)).strftime("%Y-%m") for i in range(13)]
        selected_month_str = st.selectbox("Select Month to Predict", next_months)
        selected_month = pd.Timestamp(selected_month_str + "-01")
    
    # Show historical data for the selected combination
    st.subheader('Historical Trend')
    
    historical_data = monthly_panel[
        (monthly_panel['zip_code'] == selected_zip) & 
        (monthly_panel['issue_type'] == selected_issue)
    ].sort_values('month')
    
    if not historical_data.empty:
        fig = px.line(
            historical_data, 
            x='month', 
            y='volume',
            title=f'Historical Volume for {selected_issue} in ZIP {selected_zip}'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No historical data available for this combination.")
    
    # Make the prediction
    if model is not None and st.button('Predict Volume'):
        st.subheader('Prediction Result')
        
        # Check if we have enough historical data
        if len(historical_data) < 12:
            st.warning("Not enough historical data for a reliable prediction. At least 12 months of data is recommended.")
        
        # Prepare the input data for prediction
        # Get the most recent data point from historical_data
        if not historical_data.empty:
            latest_data = historical_data.sort_values('month').iloc[-1].copy()
            
            # Update the month and create features
            input_data = pd.DataFrame([latest_data])
            input_data['month'] = selected_month
            input_data['month_num'] = selected_month.month
            input_data['sin_month'] = np.sin(2 * np.pi * input_data['month_num'] / 12)
            input_data['cos_month'] = np.cos(2 * np.pi * input_data['month_num'] / 12)
            
            # Select only the needed columns (same as used in training)
            feature_cols = ['zip_code', 'issue_type', 'vol_lag_1', 'vol_lag_3', 'vol_lag_6', 'vol_lag_12', 
                            'vol_roll_mean_3', 'vol_pct_change_1', 'vol_pct_change_12', 'sin_month', 'cos_month']
            
            # Handle missing values (in case some lags are not available)
            input_data = input_data[feature_cols].fillna(0)
            
            # Make the prediction
            prediction = model.predict(input_data)[0]
            prediction = max(0, round(prediction))  # Ensure prediction is non-negative and rounded
            
            # Display the prediction
            col1, col2 = st.columns(2)
            
            with col1:
                # Show the prediction as a metric
                st.metric("Predicted Volume", f"{prediction}")
                
                # Show average and max volumes as context
                avg_volume = historical_data['volume'].mean()
                max_volume = historical_data['volume'].max()
                st.metric("Historical Avg. Volume", f"{avg_volume:.1f}")
                st.metric("Historical Max. Volume", f"{max_volume:.0f}")
            
            with col2:
                # Show a visual comparison with historical data
                compare_data = historical_data.copy()
                
                # Add the prediction to the plot
                prediction_point = pd.DataFrame({
                    'month': [selected_month],
                    'volume': [prediction],
                    'type': ['Predicted']
                })
                
                compare_data['type'] = 'Historical'
                plot_data = pd.concat([
                    compare_data[['month', 'volume', 'type']],
                    prediction_point
                ])
                
                fig = px.line(
                    plot_data, 
                    x='month', 
                    y='volume', 
                    color='type',
                    markers=True,
                    title='Historical vs Predicted Volume'
                )
                st.plotly_chart(fig, use_container_width=True)
                
            # Provide some explanation
            st.subheader("Prediction Explanation")
            st.markdown(f"""
            The model predicts **{prediction}** service requests for **{selected_issue}** in ZIP code **{selected_zip}** for **{selected_month.strftime('%B %Y')}**.
            
            This prediction is based on:
            - Previous months' volumes
            - Seasonal patterns in the data
            - The 3-month rolling average ({input_data['vol_roll_mean_3'].values[0]:.1f})
            - Recent trend (past month change: {input_data['vol_pct_change_1'].values[0]:.1%})
            - Annual trend (past year change: {input_data['vol_pct_change_12'].values[0]:.1%})
            """)
        else:
            st.error("Cannot make prediction: No historical data available for this combination.") 
