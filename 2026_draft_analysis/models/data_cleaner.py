import pandas as pd
import numpy as np
import os

def convert_to_inches(measurement):
    """Translates heights and broad jumps into inches for standardized variable."""
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
            
            if (inches_str == ""):
                inches_str = "0"
            else:
                inches_str = inches_str.split(" ")[0]
                
            if (feet.isdigit() and inches_str.isdigit()):
                return (float(feet) * 12) + float(inches_str)
                
    if ("-" in measurement):
        parts = measurement.split("-")
        
        if (len(parts) == 2):
            left = parts[0].strip()
            right = parts[1].strip()
            
            if (left.isdigit() and right.isdigit()):
                return (float(left) * 12) + float(right)
                
    return np.nan

def clean_and_merge_data():
    """Grabs and cleans the historical combine and stats data, merges them together, and prepares the 2026 combine data for prediction."""
    data_dir = '../data/historical_training_data/'
    
    historical_combine = []
    historical_stats = []
    
    # 1. Load Historical Data
    for year in range(2021, 2026):
        combine_path = f'{data_dir}{year}Combine.csv'
        if (os.path.exists(combine_path)):
            df = pd.read_csv(combine_path)
            
            stripped_columns = []
            for col in df.columns:
                stripped_columns.append(str(col).strip('\ufeff').strip())
            df.columns = stripped_columns
            
            df['Draft_Year'] = year
            historical_combine.append(df)
            
        stats_path = f'{data_dir}{year}Stats.csv'
        if (os.path.exists(stats_path)):
            df = pd.read_csv(stats_path, header=1)
            
            stripped_columns = []
            for col in df.columns:
                stripped_columns.append(str(col).strip('\ufeff').strip())
            df.columns = stripped_columns
            
            df['Draft_Year'] = year
            historical_stats.append(df)
            
    if (len(historical_combine) == 0):
        raise FileNotFoundError(f"Could not find historical combine files in '{data_dir}'.")
        
    hist_combine_df = pd.concat(historical_combine, ignore_index=True)
    hist_stats_df = pd.concat(historical_stats, ignore_index=True)
    
    hist_combine_df['Player'] = hist_combine_df['Player'].str.replace(r'[^a-zA-Z\s.-]', '', regex=True).str.strip()
    hist_stats_df['Player'] = hist_stats_df['Player'].str.replace(r'[^a-zA-Z\s.-]', '', regex=True).str.strip()
    hist_combine_df['Pos'] = hist_combine_df['Pos'].astype(str).str.strip()
    
    train_df = pd.merge(hist_combine_df, hist_stats_df, on=['Player', 'Draft_Year'], how='left', suffixes=('', '_stats'))
    
    # 2. Load Prediction Data
    predict_df = pd.DataFrame()
    path_2026 = '../data/historical_training_data/2026Combine.csv'
    
    if (os.path.exists(path_2026)):
        predict_df = pd.read_csv(path_2026)
        
        stripped_columns = []
        for col in predict_df.columns:
            stripped_columns.append(str(col).strip('\ufeff').strip())
        predict_df.columns = stripped_columns
        
        predict_df['Draft_Year'] = 2026
        predict_df['Player'] = predict_df['Player'].str.replace(r'[^a-zA-Z\s.-]', '', regex=True).str.strip()
        predict_df['Pos'] = predict_df['Pos'].astype(str).str.strip()
        
    # 3. Base Processing
    if ('wAV' in train_df.columns):
        train_df['wAV'] = pd.to_numeric(train_df['wAV'], errors='coerce').fillna(0)
    else:
        train_df['wAV'] = 0.0
        
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
    
    dataframes_to_process = [train_df, predict_df]
    
    for df in dataframes_to_process:
        if (len(df) > 0):
            df['Ht_Inches'] = df['Ht'].apply(convert_to_inches)
            df['Broad Jump'] = df['Broad Jump'].apply(convert_to_inches)
            
            numeric_columns = ['Wt', '40yd', 'Vertical']
            for col in numeric_columns:
                if (col in df.columns):
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
            traits_to_save = ['40yd', 'Vertical', 'Broad Jump', 'Wt', 'Ht_Inches']
            unique_positions = df['Pos'].unique()
            
            for trait in traits_to_save:
                if (trait in df.columns):
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
                                        
            kicker_punter_indices = []
            for index, row in df.iterrows():
                if (row['Pos'] == 'K' or row['Pos'] == 'P'):
                    kicker_punter_indices.append(index)
                    
            df.drop(kicker_punter_indices, inplace=True)
            
            columns_to_drop = ['Bench', '3Cone', 'Shuttle', 'Ht']
            for col in columns_to_drop:
                if (col in df.columns):
                    df.drop(columns=[col], inplace=True)
                    
            feature_cols_for_drop = ['Ht_Inches', 'Wt', '40yd', 'Vertical', 'Broad Jump']
            df.dropna(subset=feature_cols_for_drop, inplace=True)
            df.reset_index(drop=True, inplace=True)
            
    train_df.dropna(subset=['Avg_wAV_Per_Season', 'wAV'], inplace=True)
    train_df.reset_index(drop=True, inplace=True)
    
    # 4. One-Hot Encoding
    if (len(predict_df) > 0):
        combined_df = pd.concat([train_df, predict_df], keys=[0, 1])
        combined_df = pd.get_dummies(combined_df, columns=['Pos'], prefix='Pos', dtype=int)
        
        train_df_ohe = combined_df.xs(0).reset_index(drop=True)
        predict_df_ohe = combined_df.xs(1).reset_index(drop=True)
    else:
        train_df_ohe = pd.get_dummies(train_df, columns=['Pos'], prefix='Pos', dtype=int)
        predict_df_ohe = pd.DataFrame(columns=train_df_ohe.columns)
        
    return train_df_ohe, predict_df_ohe

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