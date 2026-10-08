import pandas as pd
import numpy as np
import math
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
import warnings

# ignore convergence warnings from lasso/ridge
warnings.filterwarnings("ignore")

def load_data(roll_no, phase):
    # setup file names based on my roll number and phase
    train_file = f"{roll_no}_train_var{phase}.csv"
    test_file = f"{roll_no}_test_var{phase}.csv"
    
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    # split target and features
    X_train = train_df.drop(columns=['y'])
    y_train = train_df['y']
    
    X_test = test_df.drop(columns=['y']) if 'y' in test_df.columns else test_df
    
    return X_train, y_train, X_test

def find_best_model(X_train, y_train, max_degree, phase):
    degree_space = list(range(1, max_degree + 1)) 
    
    # define pipelines with polynomial features and scaling for each model type
    pipelines = {
        'OLS': Pipeline([
            ('poly', PolynomialFeatures(include_bias=False)),
            ('scaler', StandardScaler()),
            ('model', LinearRegression())
        ]),
        'Ridge': Pipeline([
            ('poly', PolynomialFeatures(include_bias=False)),
            ('scaler', StandardScaler()),
            ('model', Ridge(max_iter=10000))
        ]),
        'Lasso': Pipeline([
            ('poly', PolynomialFeatures(include_bias=False)),
            ('scaler', StandardScaler()),
            ('model', Lasso(max_iter=10000, tol=1e-3))
        ]),
        'ElasticNet': Pipeline([
            ('poly', PolynomialFeatures(include_bias=False)),
            ('scaler', StandardScaler()),
            ('model', ElasticNet(max_iter=10000, tol=1e-3))
        ])
    }

    # param grids to test out
    param_grids = {
        'OLS': {
            'poly__degree': degree_space
        },
        'Ridge': {
            'poly__degree': degree_space,
            'model__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
        },
        'Lasso': {
            'poly__degree': degree_space,
            'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
        },
        'ElasticNet': {
            'poly__degree': degree_space,
            'model__alpha': [0.001, 0.01, 0.1, 1.0],
            'model__l1_ratio': [0.2, 0.5, 0.8]
        }
    }

    n_splits = 5
    cv_strategy = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    all_results = []

    print(f"--- Starting Cross-Validation (Evaluating all degrees 1 through {max_degree}) ---")
    print(f"{'Model':<12} | {'Degree':<6} | {'CV MSE':<10} | {'Std Error':<10} | {'Parameters'}")
    print("-" * 100)
    
    # loop through models and find best params
    for name, pipeline in pipelines.items():
        grid_search = GridSearchCV(
            pipeline, 
            param_grids[name], 
            cv=cv_strategy, 
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        # log all the results
        cv_res = grid_search.cv_results_
        for i in range(len(cv_res['params'])):
            mean_mse = -cv_res['mean_test_score'][i]
            std_mse = cv_res['std_test_score'][i]
            se_mse = std_mse / math.sqrt(n_splits)
            params = cv_res['params'][i]
            degree = params['poly__degree']
            
            clean_params = {k.replace('model__', '').replace('poly__', ''): v for k, v in params.items()}
            
            all_results.append({
                'Model': name,
                'Degree': degree,
                'Params': params,
                'Mean_MSE': mean_mse,
                'SE_MSE': se_mse
            })
            
            print(f"{name:<12} | {degree:<6} | {mean_mse:<10.4f} | {se_mse:<10.4f} | {clean_params}")

    df_results = pd.DataFrame(all_results)
    
    # dump to csv so i can plot the learning curves later
    csv_log_name = f"cv_results_var{phase}.csv"
    df_results.to_csv(csv_log_name, index=False)
    print(f"\n[Info] Full CV logs saved to {csv_log_name} for graphing purposes.")

    # find the absolute best model
    min_idx = df_results['Mean_MSE'].idxmin()
    abs_min_row = df_results.loc[min_idx]
    abs_min_mse = abs_min_row['Mean_MSE']
    abs_min_se = abs_min_row['SE_MSE']
    
    # apply the 1-SE rule to avoid overfitting
    threshold = abs_min_mse + abs_min_se
    candidates = df_results[df_results['Mean_MSE'] <= threshold].copy()
    
    # sort by lowest degree first to pick the simplest model
    candidates.sort_values(by=['Degree', 'Mean_MSE'], ascending=[True, True], inplace=True)
    
    winner_row = candidates.iloc[0]
    winner_name = winner_row['Model']
    winner_params = winner_row['Params']
    winner_degree = winner_row['Degree']
    winner_mse = winner_row['Mean_MSE']
    
    print("\n--- 1-SE RULE SELECTION SUMMARY ---")
    print(f"Absolute Min MSE Found: {abs_min_mse:.4f} (SE: {abs_min_se:.4f}) by {abs_min_row['Model']} (Degree {abs_min_row['Degree']})")
    print(f"1-SE Threshold: {abs_min_mse:.4f} + {abs_min_se:.4f} = {threshold:.4f}")
    print(f"Simplest model within threshold: {winner_name} (Degree {winner_degree}) with MSE {winner_mse:.4f}")
    
    # retrain the winning model on the full training dataset
    final_pipeline = pipelines[winner_name]
    final_pipeline.set_params(**winner_params)
    final_pipeline.fit(X_train, y_train)
    
    return final_pipeline, winner_name

def process_phase(roll_no, phase, max_target_degree):
    print(f"\n{'='*70}\nExecuting Phase {phase} (var{phase})\n{'='*70}")
    
    X_train, y_train, X_test = load_data(roll_no, phase)
    print(f"Loaded Phase {phase} Data: X_train shape = {X_train.shape}, X_test shape = {X_test.shape}\n")
    
    best_model, model_name = find_best_model(X_train, y_train, max_degree=max_target_degree, phase=phase)
    
    train_preds = best_model.predict(X_train)
    train_r2 = r2_score(y_train, train_preds)
    print(f"\n=> Final Selected Model Full Train R2 Score: {train_r2:.4f}")

    # export final predictions
    print("Generating predictions for test data...")
    test_predictions = best_model.predict(X_test)
    
    output_filename = f"{roll_no}_pred_var{phase}.csv"
    submission_df = pd.DataFrame({'Predicted_y': test_predictions})
    submission_df.to_csv(output_filename, index=False)
    print(f"Predictions saved to {output_filename}\n")


if __name__ == "__main__":
    # my roll number
    ROLL_NUMBER = "BT2024192"
    
    # run both phases with the requested max degrees
    process_phase(roll_no=ROLL_NUMBER, phase=1, max_target_degree=10)
    process_phase(roll_no=ROLL_NUMBER, phase=2, max_target_degree=20)