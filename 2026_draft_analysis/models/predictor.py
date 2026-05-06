import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

def generate_predictions(train_df, predict_df, cumulative_mode=False):
    if (cumulative_mode):
        target_col = 'wAV'
    else:
        target_col = 'Avg_wAV_Per_Season'
        
    # 1. Strict Whitelist: Only grab the exact base traits we know are numbers
    feature_cols = ['Ht_Inches', 'Wt', '40yd', 'Vertical', 'Broad Jump']
    
    # Grab the One-Hot Encoded positional columns
    for col in train_df.columns:
        if (str(col).startswith('Pos_')):
            feature_cols.append(col)
            
    # 2. Isolate the data
    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].copy()
    X_predict = predict_df[feature_cols].copy()
    
    # 3. Create the Safety net
    # Force every single column to be a numeric float. If a string like 'CB' makes it in, it gets turned into a NaN, which we then fill with 0.
    for col in feature_cols:
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0)
        X_predict[col] = pd.to_numeric(X_predict[col], errors='coerce').fillna(0)
        
    # 4. Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_predict_scaled = scaler.transform(X_predict)
    
    # 5. Model 1: Support Vector Machine (Baseline carryover from cmse202)
    svm_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    svm_model.fit(X_train_scaled, y_train)
    
    predict_df['Predicted_Success_SVM'] = svm_model.predict(X_predict_scaled)
    
    # 6. Model 2: Random Forest (Advanced carryover from cmse202)
    rf_model = RandomForestRegressor(n_estimators=150, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    predict_df['Predicted_Success_RF'] = rf_model.predict(X_predict_scaled)
    
    # 7. Hybrid Score
    hybrid_scores = []
    for index, row in predict_df.iterrows():
        hybrid = (row['Predicted_Success_SVM'] + row['Predicted_Success_RF']) / 2.0
        hybrid_scores.append(hybrid)
        
    predict_df['Hybrid_Success_Score'] = hybrid_scores
    
    return predict_df

if __name__ == "__main__":
    from data_cleaner import clean_and_merge_data
    
    print("1. Running cleaner...")
    train_df, predict_df = clean_and_merge_data()
    
    if (len(predict_df) > 0):
        print("2. Running predictor (Normalized Mode)...")
        scored_df = generate_predictions(train_df, predict_df, cumulative_mode=False)
        scored_df = scored_df.sort_values(by='Hybrid_Success_Score', ascending=False).reset_index(drop=True)
        
        print("\nSUCCESS!")
        print(f"Scored {len(scored_df)} prospects from the 2026 class.")
        print("\nTop 5 Scored Prospects:")
        print(scored_df[['Player', 'Hybrid_Success_Score']].head(5))
    else:
        print("No prediction data available to score.")