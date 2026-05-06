import pandas as pd
import numpy as np
import os

def fix_height(ht):
    """Translates standard height and Excel date corruptions into total inches."""
    if (pd.isna(ht)):
        return np.nan
    
    ht = str(ht).strip()

    # SAFEGUARD 1: If it's already purely inches (e.g., "74")
    if (ht.isdigit()):
        return int(ht)

    if ("-" in ht):
        parts = ht.split("-")
        
        if (len(parts) == 2):
            left = parts[0]
            right = parts[1]

            if (left.isdigit() and right.isdigit()):
                return (int(left) * 12) + int(right)

            month_map = {
                "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
                "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
                "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
            }

            left_str = left[:3].title()
            right_str = right[:3].title()

            if (left.isdigit() and right_str in month_map):
                inches = int(left)
                feet = month_map[right_str]
                return (feet * 12) + inches

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

    master_combine['Player'] = master_combine['Player'].str.replace(r'[^a-zA-Z\s.-]', '', regex=True).str.strip()
    master_stats['Player'] = master_stats['Player'].str.replace(r'[^a-zA-Z\s.-]', '', regex=True).str.strip()
    
    # SAFEGUARD 2: Strip trailing spaces from positions
    master_combine['Pos'] = master_combine['Pos'].astype(str).str.strip()

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
            # SAFEGUARD 3: explicitly assign 0.0 to rookies
            avg_wav_list.append(0.0) 

    clean_df['Avg_wAV_Per_Season'] = avg_wav_list

    kicker_punter_indices = []
    for index, row in clean_df.iterrows():
        if (row["Pos"] == "K" or row["Pos"] == "P"):
            kicker_punter_indices.append(index)

    clean_df = clean_df.drop(kicker_punter_indices).reset_index(drop=True)
    clean_df = clean_df.drop(columns=['Bench', '3Cone', 'Shuttle'])

    # SAFEGUARD 4: Added Ht_Inches to the traits to save
    traits_to_save = ['40yd', 'Vertical', 'Broad Jump', 'Wt', 'Ht_Inches']
    unique_positions = clean_df['Pos'].unique()

    for trait in traits_to_save:
        for pos in unique_positions:
            pos_mask = (clean_df['Pos'] == pos)
            pos_avg = clean_df.loc[pos_mask, trait].mean()
            
            for index, row in clean_df.iterrows():
                if (row['Pos'] == pos):
                    if (pd.isna(row[trait])):
                        clean_df.at[index, trait] = pos_avg

    clean_df = clean_df.dropna().reset_index(drop=True)
    
    return clean_df

if __name__ == "__main__":
    print("Spinning up the data cleaner engine...")
    try:
        df = clean_and_merge_data()
        print("\n✅ SUCCESS!")
        print(f"Total Players Processed: {len(df)}")
        print(f"Final Data Shape: {df.shape}")
        
        # Adding a quick check to prove 2026 survived
        rookies = df[df['Draft_Year'] == 2026]
        print(f"2026 Rookies Survived: {len(rookies)}")
        
    except Exception as e:
        print(f"\n❌ ERROR: The cleaner crashed.")
        print(f"Details: {e}")