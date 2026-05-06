import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from data_cleaner import clean_and_merge_data
from predictor import generate_predictions

def grade_teams():
    print("1. Spinning up the ML Engine (OHE Enabled)...")
    train_df, predict_df = clean_and_merge_data()
    
    print("2. Running SVM/RF A/B Test & Ensemble Generation...")
    scored_df = generate_predictions(train_df, predict_df, cumulative_mode=False)
    
    print("3. Loading Draft Picks...")
    picks_path = '../data/raw_2026_picks.csv'
    
    if (not os.path.exists(picks_path)):
        raise FileNotFoundError(f"Could not find {picks_path}.")
        
    picks_df = pd.read_csv(picks_path)
    
    picks_df['Player'] = picks_df['Player'].str.strip()
    picks_df['Team'] = picks_df['Team'].str.strip()
    
    print("4. Merging Predictions with Team Data...")
    team_drafts = pd.merge(picks_df, scored_df, on='Player', how='inner')
    
    target_teams = [
        'Chicago Bears', 
        'Detroit Lions', 
        'Green Bay Packers', 
        'Minnesota Vikings', 
        'Baltimore Ravens'
    ]
    
    target_df = team_drafts[team_drafts['Team'].isin(target_teams)].copy()
    
    print("\n=================================")
    print("       2026 DRAFT GRADES         ")
    print("=================================")
    
    team_scores = []
    
    for team in target_teams:
        team_data = target_df[target_df['Team'] == team]
        
        if (len(team_data) > 0):
            avg_score = team_data['Hybrid_Success_Score'].mean()
            
            team_dict = {
                'Team': team, 
                'Average_Score': avg_score, 
                'Players_Drafted': len(team_data)
            }
            team_scores.append(team_dict)
            
            print(f"\n{team.upper()} (Drafted {len(team_data)} modeled players)")
            print(f"Class Grade (Avg Hybrid Score): {avg_score:.3f}")
            
            sorted_team = team_data.sort_values(by='Hybrid_Success_Score', ascending=False)
            
            for index, row in sorted_team.iterrows():
                player_pos = "UNK"
                for col in row.index:
                    if (col.startswith('Pos_') and row[col] == 1):
                        player_pos = col.replace('Pos_', '')
                        break
                        
                print(f"  - {row['Player']} ({player_pos}) | Score: {row['Hybrid_Success_Score']:.3f} (SVM: {row['Predicted_Success_SVM']:.3f}, RF: {row['Predicted_Success_RF']:.3f})")
                
    print("\n5. Generating Visualization...")
    if (len(team_scores) == 0):
        print("WARNING: No matching players found. Check raw_2026_picks.csv formatting.")
        return
        
    score_df = pd.DataFrame(team_scores).sort_values(by='Average_Score', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Average_Score', y='Team', data=score_df, palette='viridis', hue='Team', legend=False)
    
    plt.title('2026 Target Draft Class Strength (SVM + RF Hybrid)', fontsize=14, fontweight='bold')
    plt.xlabel('Average Predicted wAV Per Season', fontsize=12)
    plt.ylabel('NFL Franchise', fontsize=12)
    
    max_score = score_df['Average_Score'].max()
    plt.xlim(0, max_score * 1.15)
    
    for index, value in enumerate(score_df['Average_Score']):
        plt.text(value + 0.05, index, f'{value:.3f}', va='center', fontsize=11)
        
    plt.tight_layout()
    
    vis_path = '../visualizations/2026_draft_grades.png'
    os.makedirs(os.path.dirname(vis_path), exist_ok=True)
    plt.savefig(vis_path)
    
    print(f"Chart saved successfully to: {vis_path}")
    plt.show()

if __name__ == "__main__":
    grade_teams()