import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def generate_predictions(train_df, predict_df, cumulative_mode=False):
    """
    Trains models on historical data and predicts the 2026 class success.
    Uses cumulative_mode to toggle between total wAV and Avg_wAV_Per_Season.
    """
    # 1. Vectorized Split

    # 2. Define Features and Target
    features = ['Ht_Inches', 'Wt', '40yd', 'Vertical', 'Broad Jump']
    
    if (cumulative_mode):
        target = 'wAV'
    else:
        target = 'Avg_wAV_Per_Season'

    X_train = train_df[features]
    y_train = train_df[target]
    X_predict = predict_df[features]

    # 3. Train Models
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)

    # 4. Generate Predictions
    predict_df['Predicted_Success_LR'] = lr_model.predict(X_predict)
    predict_df['Predicted_Success_RF'] = rf_model.predict(X_predict)

    # 5. Calculate Hybrid Score
    hybrid_scores = []
    for index, row in predict_df.iterrows():
        hybrid = (row['Predicted_Success_LR'] + row['Predicted_Success_RF']) / 2.0
        hybrid_scores.append(hybrid)

    predict_df['Hybrid_Success_Score'] = hybrid_scores
    
    return predict_df

# ==========================================
# LOCAL TESTING BLOCK
# ==========================================
if __name__ == "__main__":
    from data_cleaner import clean_and_merge_data
    
    print("1. Running cleaner...")
    train_df, predict_df = clean_and_merge_data()
    
    print("2. Running predictor (Normalized Mode)...")
    scored_df = generate_predictions(train_df, predict_df, cumulative_mode=False)
    
    scored_df = scored_df.sort_values(by='Hybrid_Success_Score', ascending=False).reset_index(drop=True)
    
    print("\nSUCCESS!")
    print(f"Scored {len(scored_df)} prospects from the 2026 class.")
    print("\nTop 5 Scored Prospects:")
    print(scored_df[['Player', 'Pos', 'Hybrid_Success_Score']].head(5))