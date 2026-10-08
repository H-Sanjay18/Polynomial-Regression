# Polynomial Regression & Regularization

This repository contains the code for an academic machine learning assignment focused on applying polynomial regression (OLS, Ridge, Lasso, and ElasticNet) to two distinct multivariate datasets to predict continuous variables. 

## Author
**H Sanjay** (Roll Number: BT2024192) - IIIT Bangalore

## Project Structure
* `main.py`: The core script that runs 5-fold cross-validation, dynamically selects the optimal polynomial degree using the 1-SE rule, and outputs test predictions.
* `plot.py`: Reads the cross-validation logs and generates diagnostic visualization plots for model evaluation.
* `requirements.txt`: The required Python dependencies.
* `BT2024192_train_var1.csv` / `BT2024192_test_var1.csv`: Datasets for Phase 1 (Steam Turbine Optimization).
* `BT2024192_train_var2.csv` / `BT2024192_test_var2.csv`: Datasets for Phase 2 (Thermal Reservoir Mapping).

## Setup & Execution

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train models and generate predictions:**
   ```bash
   python main.py
   ```
   *(This will generate `cv_results_var1.csv`, `cv_results_var2.csv`, and the final prediction CSV files).*

3. **Generate evaluation plots:**
   ```bash
   python plot.py
   ```
   *(This will output `phase1_cv_analysis.png` and `phase2_cv_analysis.png`).*
