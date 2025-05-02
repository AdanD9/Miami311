"""
Save CatBoost model for use with the Streamlit app.
This script loads the panel data, trains a CatBoost model, and saves it.
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool

print("Loading data...")
# Load cleaned data
data = pd.read_parquet('miami311_clean.parquet')

print("Creating monthly panel...")
# Create monthly panel data
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

panel = create_monthly_panel(data)

# Prepare training data
print("Preparing training data...")
target = "volume"
date_col = "month"

panel.sort_values(date_col, inplace=True)
train_mask = panel[date_col] < panel[date_col].max() - pd.DateOffset(months=12)
train, valid = panel[train_mask], panel[~train_mask]

cat_cols = ["zip_code", "issue_type"]
num_cols = [c for c in panel.columns if c not in cat_cols + [target, date_col]]

X_train = train[cat_cols + num_cols]
y_train = train[target]
X_valid = valid[cat_cols + num_cols]
y_valid = valid[target]

pool_tr = Pool(X_train, y_train, cat_features=cat_cols)
pool_va = Pool(X_valid, y_valid, cat_features=cat_cols)

# Train the model
print("Training the model...")
model = CatBoostRegressor(
    iterations=3000,
    learning_rate=0.08,  # Using parameters from the optimized model in the notebook
    depth=10,
    l2_leaf_reg=8.0,
    bagging_temperature=2.7,
    loss_function="MAE",
    eval_metric="MAPE",
    random_seed=42,
    early_stopping_rounds=200,
    verbose=200
)

model.fit(pool_tr, eval_set=pool_va)

# Save the model
print("Saving the model...")
model.save_model('cat_vol_v1.cbm')

print("Done. Model saved to cat_vol_v1.cbm") 