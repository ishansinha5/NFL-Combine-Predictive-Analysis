# 2026 NFL Draft Class Analyzer: Machine Learning Pipeline

Welcome to my NFL Draft analysis project. This codebase is a massive architectural evolution of a final project I originally completed for CMSE 202 at Michigan State University. In that original project, my group built fragmented Jupyter notebooks to predict player success, and I specifically handled the machine learning models for the Tight End position. 

I wanted to take those early concepts and upgrade them into a proper, object-oriented enterprise machine learning pipeline. 

This program ingests historical NFL Combine data (2021–2025), trains an ensemble of models on raw athletic traits, and scores the 2026 NFL Draft class. I specifically targeted the NFC North (since I am a Chicago Bears fan), the Baltimore Ravens, and the Indianapolis Colts (the latter two are the favorite teams of friends of mine).

## The Tech Stack & Architecture

Moving from Jupyter notebooks to a modular Python pipeline meant making some big architectural shifts. Here is the logic behind the code:

* **One-Hot Encoding (The Master Model):** In the original CMSE 202 project, we built separate "micro-models" for every single position. That was an overfitting nightmare and computationally expensive. By using One-Hot Encoding (OHE) on the `Pos` column, I was able to build a single master model that evaluates over 1,700 players simultaneously while still dynamically adjusting its mathematical expectations based on what position a prospect plays.
* **The A/B Ensemble Test (SVM vs. Random Forest):** To honor the legacy of the original project, I kept the Support Vector Machine (SVM) as a baseline since it requires perfectly scaled data and handles linear boundaries well. I paired it head-to-head with a Random Forest Regressor, which is a much more advanced engine for handling complex, non-linear trait intersections (like size-adjusted speed). The final grade for a player is a blended "Hybrid Success Score" of both models.
* **Strict Feature Whitelisting:** The models are mathematically locked down to only look at pre-draft athletic traits (Height, Weight, 40-yard dash, Jumps) and the OHE positional columns. This prevents "target leakage"—ensuring the model can't cheat by accidentally looking at post-draft NFL stats or college names during training.

## Conclusions & Takeaways

So, who won the draft? Mathematically, the **Minnesota Vikings** walked away with the best overall draft class among the teams evaluated, scoring an average Hybrid Score of 1.960 per drafted player, heavily anchored by their defensive line picks. 

Beyond the football insights, this project was a massive learning experience for me:
1.  **Modular Python:** I learned how to properly separate concerns. Moving data cleaning, model training, and orchestration into their own isolated scripts (`data_cleaner.py`, `predictor.py`, `team_grader.py`) makes the code infinitely easier to debug and scale.
2.  **Handling Missing Data:** I built strict drop logic. If a player wasn't invited to the Combine or didn't run the required drills, the pipeline refuses to hallucinate dummy stats to score them. This maintains mathematical integrity (and is why a few late-round picks or undrafted free agents are intentionally missing from the final outputs).
3.  **Advanced ML Techniques:** Upgrading from basic linear regressions to an ensemble method utilizing Random Forests and OHE drastically improved the model's ability to evaluate complex, real-world data.

## How to Use & Repurpose This Repo

Want to see how your favorite team's draft class scored? You can easily clone this repo and run your own teams through the pipeline. 

**Step-by-Step Instructions:**
1.  **Clone the repository:** `git clone https://github.com/yourusername/2026_draft_analysis.git`
2.  **Navigate to the data folder:** Open `/data/raw_2026_picks.csv`.
3.  **Add your team:** Manually add the players your team drafted using the exact format: `Team Name,Player Name,Position,College`. *(Note: Ensure the player's name is spelled exactly as it appears in standard NFL Combine databases, or the `inner` join will drop them)*
4.  **Update the Orchestrator:** Open `/models/team_grader.py`, find the `target_teams` list variable, and add your exact team string to the array.
5.  **Run the pipeline:** Open your terminal and navigate to the `/models/` directory.

To generate the Franchise evaluations, run:
`python team_grader.py`

To generate the Positional evaluations, run:
`python position_grader.py`

By default, the pipeline runs in **Normalized Mode**. If you want to evaluate players based on raw career longevity rather than per-season impact, simply append `cumulative` to the commands:
`python team_grader.py cumulative`
`python position_grader.py cumulative`

To understand the exact difference between these two mathematical approaches, check out the `README.md` inside the `/visualizations/` folder.
