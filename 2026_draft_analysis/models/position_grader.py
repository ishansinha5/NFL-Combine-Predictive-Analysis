import sys
import pandas as pd
import matplotlib.pyplot as plt
import os

from data_cleaner import clean_and_merge_data
from predictor import generate_predictions

def generate_position_visualizations(scored_df, mode):
    # 1. Establish the routing logic for the toggle
    save_dir = "../visualizations/"
    
    if (mode == "normalized"):
        save_dir = "../visualizations/normalized/positions/"
        
    if (mode == "cumulative"):
        save_dir = "../visualizations/cumulative/positions/"
        
    if (os.path.exists(save_dir) == False):
        os.makedirs(save_dir)

    # 2. Reconstruct the base 'Pos' column from OHE columns without lambdas
    pos_list = []
    
    for index, row in scored_df.iterrows():
        player_pos = "UNK"
        for col in row.index:
            if (str(col).startswith('Pos_')):
                if (row[col] == 1):
                    player_pos = str(col).replace('Pos_', '')
                    break
        pos_list.append(player_pos)
        
    # Append the un-encoded list back into the dataframe safely
    scored_df['Pos'] = pos_list
    
    # 3. Iterate through unique positions
    unique_positions = scored_df['Pos'].unique()
    
    for position in unique_positions:
        if (position == "UNK"):
            continue
            
        pos_df = scored_df[scored_df['Pos'] == position]
        
        if (len(pos_df) > 0):
            pos_df = pos_df.sort_values(by='Hybrid_Success_Score', ascending=False)
            
            # Limit to top 15 for visual clarity on crowded charts
            if (len(pos_df) > 15):
                pos_df = pos_df.head(15)
                
            plot_df = pos_df[['Player', 'Predicted_Success_SVM', 'Predicted_Success_RF']]
            plot_df = plot_df.set_index('Player')
            
            plot_df.plot(kind='bar', figsize=(14, 7), color=['darkorange', 'purple'])
            
            plt.title('Top 2026 Prospects: ' + position + ' (SVM vs RF)')
            plt.ylabel('Predicted Score')
            plt.xlabel('Player')
            plt.xticks(rotation=45)
            plt.legend(['SVM Score', 'Random Forest Score'])
            plt.tight_layout()
            
            filename = position + "_rankings.png"
            plt.savefig(save_dir + filename)
            plt.close()
            
            print("Generated positional chart for: " + position)

if (__name__ == "__main__"):
    print("1. Booting up the ML Engine (OHE Enabled)...")
    train_df, predict_df = clean_and_merge_data()
    
    mode = "normalized"
    cum_mode = False
    
    if (len(sys.argv) > 1):
        if (sys.argv[1] == "cumulative"):
            mode = "cumulative"
            cum_mode = True
            
    print("2. Running SVM/RF A/B Test & Ensemble Generation for " + mode + " mode...")
    scored_df = generate_predictions(train_df, predict_df, cumulative_mode=cum_mode)
    
    print("3. Generating Positional Visualizations...")
    generate_position_visualizations(scored_df, mode)
    print("Positional visualizations complete.")