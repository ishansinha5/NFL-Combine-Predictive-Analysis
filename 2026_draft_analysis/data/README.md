# Data Engineering, Imputation Logic & Storage

This directory manages the core datasets that power the machine learning pipeline, separating raw historical draft data from the target 2026 prospect class.

## Dataset Inventory

* **Historical Combine Data (2021Combine.csv – 2025Combine.csv):** Ingests raw, verified athletic testing metrics straight from the NFL Combine over a five-year window.
* **Historical Career Production (2021Stats.csv – 2025Stats.csv):** Houses actual NFL career production numbers for matching players, tracking total games played (G) and career Weighted Approximate Value (wAV).
* **Target Roster Evaluation Class (raw_2026_picks.csv):** Defines the exact 2026 draft selections made by our tracked franchises: the Chicago Bears, Detroit Lions, Green Bay Packers, Minnesota Vikings, Baltimore Ravens, Buffalo Bills, and Indianapolis Colts.

## Data Pipelines (Atomic Breakdowns)

### 1. Rigid Whitelisting & Feature Lock
The extraction pipeline applies a strict mathematical filter across all historical datasets. The model isolates five exact numeric testing traits: Height (converted entirely to total inches), Weight (lbs), 40-Yard Dash (seconds), Vertical Jump (inches), and Broad Jump (converted from feet-inches to total inches). This strict structural constraint physically prevents target leakage during training by blocking post-draft variables from reaching the mathematical core.

### 2. Location-Based Positional Imputation
To preserve data integrity, missing values are never filled using a crude global mean. If an offensive tackle skips the vertical jump or a wide receiver lacks a 40-yard dash time, the missing value is calculated based entirely on the mathematical average of that specific prospect's position group. This ensures speed expectations are weighed realistically according to real-world positional archetypes rather than penalizing heavy linemen or skewing standard defensive back metrics.

### 3. Structural Pruning
If a draft prospect did not participate in the Combine drills or lacked baseline measurable data entirely, the pipeline enforces strict drop-logic, purging the blank row rather than inventing false stats. This zero-hallucination constraint guarantees that every single success grade is backed completely by empirical physical traits.