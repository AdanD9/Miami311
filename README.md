# Miami 311 Analytics and Prediction Platform

This Streamlit application analyzes historical 311 service requests in Miami-Dade County and provides a forecasting model to predict future service request volumes by zip code, issue type, and month.

## Features

### Dashboard Tab
- Overview of 311 service requests with key metrics
- Time series analysis of service request volumes
- Top request types and distribution by district
- Geographic visualization of request locations
- Resolution time analysis by issue type
- SLA breach rate analysis

### Forecasting Tab
- Interactive forecasting tool for predicting future service request volumes
- Selection of zip code, issue type, and target month for prediction
- Historical trend visualization for selected parameters
- Detailed prediction results with explanations
- Comparison of predictions with historical data

## Setup Instructions

1. Clone this repository
2. Install the required packages:
```
pip install -r requirements.txt
```
3. Ensure you have the dataset file `miami311_clean.parquet` in the same directory as the app
4. Ensure you have the model file `cat_vol_v1.cbm` in the same directory as the app
5. Run the Streamlit app:
```
streamlit run app.py
```

## Data Source

The data used in this application comes from the Miami-Dade County's official GIS Open Data portal, containing 311 service requests from 2013 to 2023. The data has been cleaned and processed for analysis.

## Model Details

The forecasting model is built using CatBoost, a gradient boosting library developed by Yandex. The model was trained on historical 311 service request data with the following features:
- Previous months' request volumes (lags)
- Rolling averages
- Percentage changes
- Seasonal components
- Spatial and categorical features

The model achieves a Mean Absolute Percentage Error (MAPE) of approximately 7% on validation data, making it reasonably accurate for forecasting purposes. 