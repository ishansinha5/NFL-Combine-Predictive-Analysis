import sys
import pandas as pd
import matplotlib.pyplot as plt
import os

from data_cleaner import clean_and_merge_data
from predictor import generate_predictions

def generate_position_visualizations(scored_df, mode):
    save_dir = "../visualizations/"
    if (mode == "normalized"):
        save_dir = "../visualizations/normalized/positions/"
    if (mode == "cumulative"):
        save_dir = "../visualizations/cumulative/positions/"
        
    if (os.path.exists(save_dir) == False):
        os.makedirs(save_dir)

    # 1. Load team mapping to add abbreviations
    picks_path = '../data/raw_2026_picks.csv'
    picks_df = pd.read_csv(picks_path)
    
    team_map = {
        'Chicago Bears': '(CB)',
        'Minnesota Vikings': '(MV)',
        'Detroit Lions': '(DL)',
        'Green Bay Packers': '(GBP)',
        'Baltimore Ravens': '(BR)',
        'Indianapolis Colts': '(IC)',
        'Buffalo Bills': '(BB)'
    }
    
    # 2. Reconstruct Pos and Update Names with Abbreviations
    pos_list = []
    labeled_names = []
    
    for index, row in scored_df.iterrows():
        # Find Position
        player_pos = "UNK"
        for col in row.index:
            if (str(col).startswith('Pos_')):
                if (row[col] == 1):
                    player_pos = str(col).replace('Pos_', '')
                    break
        pos_list.append(player_pos)
        
        # Find Team Abbreviation
        current_player = row['Player']
        found_team = "(N/A)"
        
        for p_index, p_row in picks_df.iterrows():
            if (p_row['Player'] == current_player):
                team_full_name = p_row['Team']
                if (team_full_name in team_map):
                    found_team = team_map[team_full_name]
                break
        
        labeled_names.append(current_player + " " + found_team)
        
    scored_df['Pos'] = pos_list
    scored_df['Labeled_Player'] = labeled_names
    
    # 3. Iterate through positions
    unique_positions = scored_df['Pos'].unique()
    for position in unique_positions:
        if (position == "UNK"):
            continue
            
        pos_df = scored_df[scored_df['Pos'] == position]
        if (len(pos_df) > 0):
            pos_df = pos_df.sort_values(by='Hybrid_Success_Score', ascending=False)
            if (len(pos_df) > 15):
                pos_df = pos_df.head(15)
                
            plot_df = pos_df[['Labeled_Player', 'Predicted_Success_SVM', 'Predicted_Success_RF']]
            plot_df = plot_df.set_index('Labeled_Player')
            
            plot_df.plot(kind='bar', figsize=(14, 7), color=['darkorange', 'purple'])
            plt.title('Top 2026 Prospects: ' + position + ' (SVM vs RF)')
            plt.ylabel('Predicted Score')
            plt.xlabel('Player (Team)')
            plt.xticks(rotation=45)
            plt.legend(['SVM Score', 'Random Forest Score'])
            plt.tight_layout()
            
            plt.savefig(save_dir + position + "_rankings.png")
            plt.close()
            print("Generated positional chart for: " + position)

if (__name__ == "__main__"):
    print("1. Spinning up the ML Engine...")
    train_df, predict_df = clean_and_merge_data()
    
    mode = "normalized"
    cum_mode = False
    if (len(sys.argv) > 1):
        if (sys.argv[1] == "cumulative"):
            mode = "cumulative"
            cum_mode = True
            
    scored_df = generate_predictions(train_df, predict_df, cumulative_mode=cum_mode)
    generate_position_visualizations(scored_df, mode)