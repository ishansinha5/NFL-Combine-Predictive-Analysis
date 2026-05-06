import pandas as pd
import numpy as np
import os

def fix_height(ht):
    """Translates standard height and Excel date corruptions into total inches."""
    if (pd.isna(ht)):
        return np.nan
    
    ht = str(ht).strip()

    if ("-" in ht):
        parts = ht.split("-")
        
        if (len(parts) == 2):
            left = parts[0]
            right = parts[1]

            # Case 1: Standard numeric format (e.g., "6-2")
            if (left.isdigit() and right.isdigit()):
                return (int(left) * 12) + int(right)

            month_map = {
                "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
                "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
                "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
            }

            left_str = left[:3].title()
            right_str = right[:3].title()

            # Case 2: Excel corrupted Day-Month (e.g., "11-May")
            if (left.isdigit() and right_str in month_map):
                inches = int(left)
                feet = month_map[right_str]
                return (feet * 12) + inches

            # Case 3: Excel corrupted Month-Day (e.g., "May-11")
            if (left_str in month_map and right.isdigit()):
                feet = month_map[left_str]
                inches = int(right)
                return (feet * 12) + inches

    return np.nan

def clean_and_merge_data():
    """
    Ingests raw Combine and Stats data, handles missing values, 
    removes non-predictive positions, and returns a pristine DataFrame.
    """
    data_dir = '../data/historical_training_data/'

    all_combine_data = []
    all_stats_data = []

    # 1. Load Data
    for year in range(2021, 2027):
        if (year == 2026):
            combine_file_path = '../data/2026Combine.csv'
        else:
            combine_file_path = f'{data_dir}{year}Combine.csv'
            
        if (os.path.exists(combine_file_path)):
            combine_df = pd.read_csv(combine_file_path)
            combine_df['Draft_Year'] = year
            all_combine_data.append(combine_df)
            
        stats_file_path = f'{data_dir}{year}Stats.csv'
        if (os.path.exists(stats_file_path)):
            stats_df = pd.read_csv(stats_file_path, header=1)
            stats_df['Draft_Year'] = year
            all_stats_data.append(stats_df)

    master_combine = pd.concat(all_combine_data, ignore_index=True)
    master_stats = pd.concat(all_stats_data, ignore_index=True)

    # 2. Regex Scrubbing
    master_combine['Player'] = master_combine['Player'].str.replace(r'[^a-zA-Z\s.-]', '', regex=True).str.strip()
    master_stats['Player'] = master_stats['Player'].str.replace(r'[^a-zA-Z\s.-]', '', regex=True).str.strip()

    # 3. LEFT Join
    merged_df = pd.merge(
        master_combine, 
        master_stats, 
        on=['Player', 'Draft_Year'], 
        how='left',
        suffixes=('', '_stats')
    )

    feature_columns = [
        'Player', 'Pos', 'Draft_Year', 'Ht', 'Wt', '40yd', 
        'Vertical', 'Bench', 'Broad Jump', '3Cone', 'Shuttle'
    ]
    target_columns = ['G', 'wAV']

    for col in target_columns:
        if (col not in merged_df.columns):
            merged_df[col] = np.nan

    all_columns = feature_columns + target_columns
    clean_df = merged_df[all_columns].copy()

    # 4. Numerics & Metrics
    clean_df['Ht_Inches'] = clean_df['Ht'].apply(fix_height)
    clean_df = clean_df.drop(columns=['Ht'])

    clean_df['Years_In_League'] = 2026 - clean_df['Draft_Year']
    clean_df['wAV'] = pd.to_numeric(clean_df['wAV'], errors='coerce').fillna(0)
    clean_df['G'] = pd.to_numeric(clean_df['G'], errors='coerce').fillna(0)

    avg_wav_list = []
    for index, row in clean_df.iterrows():
        years = row['Years_In_League']
        wav = row['wAV']
        
        if (years > 0):
            avg_wav_list.append(wav / years)
        else:
            avg_wav_list.append(np.nan)

    clean_df['Avg_wAV_Per_Season'] = avg_wav_list

    # 5. Omit Kickers and Punters
    kicker_punter_indices = []
    for index, row in clean_df.iterrows():
        if (row["Pos"] == "K" or row["Pos"] == "P"):
            kicker_punter_indices.append(index)

    clean_df = clean_df.drop(kicker_punter_indices).reset_index(drop=True)

    # 6. Drop Heavily Missing Columns
    clean_df = clean_df.drop(columns=['Bench', '3Cone', 'Shuttle'])

    # 7. Filling in the blanks using position averages
    traits_to_save = ['40yd', 'Vertical', 'Broad Jump', 'Wt']
    unique_positions = clean_df['Pos'].unique()

    for trait in traits_to_save:
        for pos in unique_positions:
            pos_mask = (clean_df['Pos'] == pos)
            pos_avg = clean_df.loc[pos_mask, trait].mean()
            
            for index, row in clean_df.iterrows():
                if (row['Pos'] == pos):
                    if (pd.isna(row[trait])):
                        clean_df.at[index, trait] = pos_avg

    # 8. Final Drop & Return
    clean_df = clean_df.dropna().reset_index(drop=True)
    
    return clean_df

# ==========================================
# LOCAL TESTING BLOCK
# ==========================================
# This will only run if you execute this file directly.
if __name__ == "__main__":
    print("Spinning up the data cleaner engine...")
    
    try:
        # Run your master function
        df = clean_and_merge_data()
        
        # Spit out the diagnostics
        print("\nSUCCESS!")
        print(f"Total Players Processed: {len(df)}")
        print(f"Final Data Shape: {df.shape}")
        
        print("\nSneak peek at the first 3 rows:")
        print(df[['Player', 'Pos', 'Draft_Year', 'Avg_wAV_Per_Season']].head(3))
        
    except Exception as e:
        print(f"\nERROR: The cleaner crashed.")
        print(f"Details: {e}")