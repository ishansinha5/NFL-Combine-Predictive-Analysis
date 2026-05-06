import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from data_cleaner import clean_and_merge_data
from predictor import generate_predictions

def generate_team_visualizations(merged_df, mode):
    # 1. Establish the routing logic for the toggle
    save_dir = "../visualizations/"
    
    if (mode == "normalized"):
        save_dir = "../visualizations/normalized/"
        
    if (mode == "cumulative"):
        save_dir = "../visualizations/cumulative/"
        
    if (os.path.exists(save_dir) == False):
        os.makedirs(save_dir)

    # 2. OVERALL DRAFT CLASS GRADES
    team_averages = merged_df.groupby('Team')['Hybrid_Success_Score'].mean().reset_index()
    team_averages = team_averages.sort_values(by='Hybrid_Success_Score', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Hybrid_Success_Score', y='Team', data=team_averages, palette='viridis', hue='Team', legend=False)
    plt.title('Overall 2026 Draft Class Average Hybrid Score (' + mode.upper() + ')')
    plt.xlabel('Avg Hybrid Success Score')
    plt.ylabel('NFL Franchise')
    plt.tight_layout()
    plt.savefig(save_dir + "Overall_Draft_Grades.png")
    plt.close()

    # 3. INDIVIDUAL TEAM GROUPED BAR CHARTS
    unique_teams = merged_df['Team'].unique()
    
    for team_name in unique_teams:
        team_df = merged_df[merged_df['Team'] == team_name]
        
        if (len(team_df) > 0):
            team_df = team_df.sort_values(by='Hybrid_Success_Score', ascending=False)
            
            plot_df = team_df[['Player', 'Predicted_Success_SVM', 'Predicted_Success_RF']]
            plot_df = plot_df.set_index('Player')
            
            plot_df.plot(kind='bar', figsize=(12, 6), color=['blue', 'green'])
            
            plt.title(team_name + ' - 2026 Draft Picks (SVM vs RF)')
            plt.ylabel('Predicted Score')
            plt.xlabel('Drafted Player')
            plt.xticks(rotation=45)
            plt.legend(['SVM Score', 'Random Forest Score'])
            plt.tight_layout()
            
            filename = team_name + "_picks.png"
            plt.savefig(save_dir + filename)
            plt.close()
            
            print("Generated team chart for: " + team_name)

def grade_teams(run_mode):
    print("1. Spinning up the ML Engine (OHE Enabled)...")
    train_df, predict_df = clean_and_merge_data()
    
    print("2. Running SVM/RF A/B Test & Ensemble Generation for " + run_mode + " mode...")
    cum_mode = False
    if (run_mode == "cumulative"):
        cum_mode = True
        
    scored_df = generate_predictions(train_df, predict_df, cumulative_mode=cum_mode)
    
    print("3. Loading Draft Picks...")
    picks_path = '../data/raw_2026_picks.csv'
    
    if (os.path.exists(picks_path) == False):
        raise FileNotFoundError("Could not find picks dataset.")
        
    picks_df = pd.read_csv(picks_path)
    
    picks_df['Player'] = picks_df['Player'].str.strip()
    picks_df['Team'] = picks_df['Team'].str.strip()
    
    print("4. Merging Predictions with Team Data...")
    team_drafts = pd.merge(picks_df, scored_df, on='Player', how='inner')
    
    # ADDED THE COLTS TO THE TARGET ARRAY
    target_teams = [
        'Chicago Bears', 
        'Detroit Lions', 
        'Green Bay Packers', 
        'Minnesota Vikings', 
        'Baltimore Ravens',
        'Indianapolis Colts'
    ]
    merged_df = team_drafts[team_drafts['Team'].isin(target_teams)].copy()
    
    print("\n=================================")
    print("       2026 DRAFT GRADES         ")
    print("=================================")
    
    for team in target_teams:
        team_data = merged_df[merged_df['Team'] == team]
        
        if (len(team_data) > 0):
            avg_score = team_data['Hybrid_Success_Score'].mean()
            print("\n" + team.upper() + " (Drafted " + str(len(team_data)) + " modeled players)")
            print("Class Grade (Avg Hybrid Score): " + str(round(avg_score, 3)))
            
            sorted_team = team_data.sort_values(by='Hybrid_Success_Score', ascending=False)
            
            for index, row in sorted_team.iterrows():
                player_pos = "UNK"
                for col in row.index:
                    if (str(col).startswith('Pos_')):
                        if (row[col] == 1):
                            player_pos = str(col).replace('Pos_', '')
                            break
                            
                print("  - " + row['Player'] + " (" + player_pos + ") | Score: " + str(round(row['Hybrid_Success_Score'], 3)) + " (SVM: " + str(round(row['Predicted_Success_SVM'], 3)) + ", RF: " + str(round(row['Predicted_Success_RF'], 3)) + ")")
                
    print("\n5. Generating Visualizations...")
    generate_team_visualizations(merged_df, run_mode)

if (__name__ == "__main__"):
    mode = "normalized"
    
    if (len(sys.argv) > 1):
        if (sys.argv[1] == "cumulative"):
            mode = "cumulative"
            
    grade_teams(mode)