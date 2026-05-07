# The Predictive Architecture Core

This folder contains the machine learning orchestration engine, processing scripts, and training configurations that handle everything from raw string cleanup to running complex ensemble evaluations.

## Script Architecture & File Purposes

### 1. `Model_Visualization_Sandbox.ipynb` (The Prototyping Environment)
Despite the name, this Jupyter Notebook served as the critical data engineering sandbox. This is where the complex positional imputation logic, missing value diagnostics, and structural data dropping rules were rigorously prototyped and verified before being permanently translated into the modular data_cleaner.py script.

### 2. `data_cleaner.py` (The Extraction Pipeline)
The front-end script responsible for parsing, cleaning, and formatting datasets. It runs explicit conditional parsing to split and convert string measurements into numeric values. It maps and calculates the positional averages for data imputation and handles the final step of concatenating the historical and target classes to build identical One-Hot Encoded (OHE) structural representations.

### 3. `predictor.py` (The Machine Learning Engine)
The core mathematical processing unit. This script sets up a strict StandardScaler to normalize the physical metrics before training. It then runs an A/B ensemble head-to-head test:
* **Support Vector Regressor (SVR):** Sets a stable baseline using a radial basis function (RBF) kernel to parse linear and close boundaries.
* **Random Forest Regressor:** A robust multi-decision-tree structure capable of tracking complex, non-linear trait intersections (such as size-adjusted speed ratios).
The script automatically blends their separate outputs to generate a final, highly balanced Hybrid_Success_Score.

### 4. `team_grader.py` (Franchise Class Orchestrator)
The primary pipeline evaluator for assessing overall draft classes. It reads individual team rosters from ../data/raw_2026_picks.csv and computes collective success ratings across our target teams. It generates the final console output breakdowns and dynamically exports the franchise-wide bar chart visualizations straight to the ../visualizations/ directory, properly routing them based on the active normalization mode.

### 5. `position_grader.py` (Positional Analytics & Competitive Intelligence)
The script used to isolate and score the entire 2026 draft class sorted exclusively by field position. It generates comprehensive grouped bar charts ranking the top 15 athletic prospects per position to provide deep athletic comparisons, tagging players with their drafted team abbreviations.

## Command Line Execution Options

By default, scripts execute in **Normalized Mode**. To alter how the model evaluates long-term durability versus immediate seasonal efficiency, you can toggle between execution modes via the terminal:

### Normalized Mode (Default Efficiency Metric)
Evaluates prospects against Avg_wAV_Per_Season. This isolates a player's true per-season value, establishing a level playing field between high-impact younger prospects and older veterans. Output visualizations route to ../visualizations/normalized/.
`python team_grader.py`
`python position_grader.py`

### Cumulative Mode (Absolute Longevity Metric)
Forces the engine to lock onto absolute total career wAV. This shifts the weights toward predicting raw physical durability and career length, though it naturally favors longer-tenured players in the training data. Output visualizations route to ../visualizations/cumulative/.
`python team_grader.py cumulative`
`python position_grader.py cumulative`