# Data Engineering, Imputation Logic & Storage

This directory manages the core datasets that power the machine learning pipeline, separating raw historical draft data from the target 2026 prospect class.

## Dataset Inventory

* **Historical Combine Data (2021Combine.csv – 2025Combine.csv):** Ingests raw, verified athletic testing metrics straight from the NFL Combine over a five-year window.
* **Historical Career Production (2021Stats.csv – 2025Stats.csv):** Houses actual NFL career production numbers for matching players, tracking total games played (G) and career Weighted Approximate Value (wAV).
* **Target Roster Evaluation Class (raw_2026_picks.csv):** Defines the exact 2026 draft selections made by our tracked franchises: the Chicago Bears, Detroit Lions, Green Bay Packers, Minnesota Vikings, Baltimore Ravens, Buffalo Bills, and Indianapolis Colts.

## Data Pipelines (Atomic Breakdowns)

### 1. String Parsing & Cleaning
The extraction pipeline applies rigorous string formatting. Raw height strings (e.g., "6-4") are split and converted into pure numerical inches (76.0). Broad jump data is similarly parsed into total inches to ensure mathematical uniformity for the StandardScaler used in the predictive core.

### 2. Location-Based Positional Imputation
To preserve data integrity, missing values are never filled using a crude global mean. If an offensive tackle skips the vertical jump or a wide receiver lacks a 40-yard dash time, the missing value is calculated based entirely on the mathematical average of that specific prospect's position group. This ensures speed expectations are weighed realistically according to real-world positional archetypes rather than penalizing heavy linemen or skewing standard defensive back metrics.

### 3. Rigid Whitelisting & Structural Pruning
The model isolates five exact numeric testing traits: Height (inches), Weight (lbs), 40-Yard Dash (seconds), Vertical Jump (inches), and Broad Jump (inches). This strict structural constraint physically prevents target leakage during training by blocking post-draft variables from reaching the mathematical core. If a draft prospect lacks baseline measurable data entirely, the pipeline enforces strict drop-logic, purging the blank row rather than inventing false stats. This zero-hallucination constraint guarantees that every single success grade is backed completely by empirical physical traits.