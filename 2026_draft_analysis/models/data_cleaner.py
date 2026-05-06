import pandas as pd
import numpy as np
import os

def convert_to_inches(measurement):
    """Translates heights and broad jumps into pure inches."""
    if (pd.isna(measurement)):
        return np.nan
    
    measurement = str(measurement).strip()

    if (measurement.replace('.', '', 1).isdigit()):
        return float(measurement)

    if ("'" in measurement):
        clean_str = measurement.replace('"', '').replace('”', '')
        parts = clean_str.split("'")
        if (len(parts) >= 2):
            feet = parts[0].strip()
            inches_str = parts[1].strip()
            inches = inches_str.split(" ")[0] if inches_str else "0"
            
            if (feet.isdigit() and inches.isdigit()):
                return (float(feet) * 12) + float(inches)

    if ("-" in measurement):
        parts = measurement.split("-")
        
        if (len(parts) == 2):
            left = parts[0].strip()
            right = parts[1].strip()

            if (left.isdigit() and right.isdigit()):
                return (float(left) * 12) + float(right)

    return np.nan

def clean_and_merge_data():
    """
    Ingests Combine and Stats data, handles missing values, 
    and returns TWO DataFrames: one for training, one for prediction.
    """
    data_dir = '../data/historical_training_data/'

    # --- 1. Load Historical Data (2021-2025) ---
    historical_combine = []
    historical_stats = []

    print("\n--- DIAGNOSTIC RADAR: SEARCHING FOR FILES ---")
    for year in range(2021, 2026):
        combine_file_path = f'{data_dir}{year}Combine.csv'
        print(f"Looking for: {combine_file_path}  -> Found: {os.path.exists(combine_file_path)}")
        
        if (os.path.exists(combine_file_path)):
            df = pd.read_csv(combine_file_path)
            df.rename(columns=lambda x: str(x).strip('\ufeff').strip(), inplace=True)
            df['Draft_Year'] = year
            historical_combine.append(df)
            
        stats_file_path = f'{data_dir}{year}Stats.csv'
        if (os.path.exists(stats_file_path)):
            df = pd.read_csv(stats_file_path, header=1)
            df.rename(columns=lambda x: str(x).strip('\ufeff').strip(), inplace=True)
            df['Draft_Year'] = year
            historical_stats.append(df)
    print("---------------------------------------------\n")

    if not historical_combine:
        raise FileNotFoundError(f"CRITICAL ERROR: Could not find ANY historical combine files in '{data_dir}'. Please check your folder names and file names in VS Code!")

    hist_combine_df = pd.concat(historical_combine, ignore_index=True)
    hist_stats_df = pd.concat(historical_stats, ignore_index=True)

    # Clean Names
    hist_combine_df['Player'] = hist_combine_df['Player'].str.replace(r'[^a-zA-Z\s.-]', '', regex=True).str.strip()
    hist_stats_df['Player'] = hist_stats_df['Player'].str.replace(r'[^a-zA-Z\s.-]', '', regex=True).str.strip()
    hist_combine_df['Pos'] = hist_combine_df['Pos'].astype(str).str.strip()

    # THE FIX IS HERE: Added suffixes=('', '_stats')
    # Merge Historical
    train_df = pd.merge(hist_combine_df, hist_stats_df, on=['Player', 'Draft_Year'], how='left', suffixes=('', '_stats'))

    # --- 2. Load Prediction Data (2026) ---
    predict_df = pd.DataFrame()
    path_2026 = '../data/historical_training_data/2026Combine.csv' 
    print(f"Looking for 2026 File: {path_2026} -> Found: {os.path.exists(path_2026)}")
    
    if (os.path.exists(path_2026)):
        predict_df = pd.read_csv(path_2026)
        predict_df.rename(columns=lambda x: str(x).strip('\ufeff').strip(), inplace=True)
        predict_df['Draft_Year'] = 2026
        predict_df['Player'] = predict_df['Player'].str.replace(r'[^a-zA-Z\s.-]', '', regex=True).str.strip()
        predict_df['Pos'] = predict_df['Pos'].astype(str).str.strip()

    if predict_df.empty:
        print("\nWARNING: 2026 Combine data was not found. Predictor will have no rookies to score.")

    # --- 3. Process Both DataFrames ---
    
    # Target Columns (Only relevant for train_df)
    train_df['wAV'] = pd.to_numeric(train_df.get('wAV', np.nan), errors='coerce').fillna(0)
    train_df['Years_In_League'] = 2026 - train_df['Draft_Year']
    
    avg_wav_list = []
    for index, row in train_df.iterrows():
        years = row['Years_In_League']
        wav = row['wAV']
        if (years > 0):
            avg_wav_list.append(wav / years)
        else:
            avg_wav_list.append(0.0)
    train_df['Avg_wAV_Per_Season'] = avg_wav_list

    # Feature Processing for BOTH
    features = ['Ht', 'Wt', '40yd', 'Vertical', 'Broad Jump']
    
    for df in [train_df, predict_df]:
        if not df.empty:
            df['Ht_Inches'] = df['Ht'].apply(convert_to_inches)
            df['Broad Jump'] = df['Broad Jump'].apply(convert_to_inches)
            
            for col in ['Wt', '40yd', 'Vertical']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Filling missing combine drills with positional averages
            traits_to_save = ['40yd', 'Vertical', 'Broad Jump', 'Wt', 'Ht_Inches']
            unique_positions = df['Pos'].unique()
            for trait in traits_to_save:
                global_avg = df[trait].mean()
                for pos in unique_positions:
                    pos_mask = (df['Pos'] == pos)
                    pos_avg = df.loc[pos_mask, trait].mean()
                    for index, row in df.iterrows():
                        if (row['Pos'] == pos):
                            if (pd.isna(row[trait])):
                                if (pd.isna(pos_avg)):
                                    df.at[index, trait] = global_avg
                                else:
                                    df.at[index, trait] = pos_avg

            # Drop unnecessary columns and K/P
            kicker_punter_indices = df[df['Pos'].isin(['K', 'P'])].index
            df.drop(kicker_punter_indices, inplace=True)
            df.drop(columns=['Bench', '3Cone', 'Shuttle', 'Ht'], errors='ignore', inplace=True)
            
            # Crucially: Drop rows ONLY based on feature columns
            feature_cols_for_drop = ['Ht_Inches', 'Wt', '40yd', 'Vertical', 'Broad Jump']
            df.dropna(subset=feature_cols_for_drop, inplace=True)
            df.reset_index(drop=True, inplace=True)

    # Ensure train_df drops rows missing the target variable before returning
    train_df.dropna(subset=['Avg_wAV_Per_Season', 'wAV'], inplace=True)

    return train_df, predict_df

if __name__ == "__main__":
    print("Booting up the data cleaner engine...")
    try:
        train_df, predict_df = clean_and_merge_data()
        print("\nSUCCESS!")
        print(f"Historical Training Players Processed: {len(train_df)}")
        print(f"2026 Prediction Players Survived: {len(predict_df)}")
        
    except Exception as e:
        print(f"\nERROR: The cleaner crashed.")
        print(f"Details: {e}")