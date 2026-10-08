import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 1.8,
    'lines.markersize': 6,
    'figure.autolayout': True
})

def plot_phase_diagnostics(csv_file, phase_num, winning_model_name, output_png):
    df = pd.read_csv(csv_file)
    
    min_idx = df['Mean_MSE'].idxmin()
    abs_min_row = df.loc[min_idx]
    abs_min_mse = abs_min_row['Mean_MSE']
    abs_min_se = abs_min_row['SE_MSE']
    threshold = abs_min_mse + abs_min_se
    
    candidates = df[df['Mean_MSE'] <= threshold].copy()
    candidates.sort_values(by=['Degree', 'Mean_MSE'], ascending=[True, True], inplace=True)
    winner_row = candidates.iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    ax1 = axes[0]
    models = ['OLS', 'Ridge', 'Lasso', 'ElasticNet']
    colors = {'OLS': '#d62728', 'Ridge': '#1f77b4', 'Lasso': '#2ca02c', 'ElasticNet': '#ff7f0e'}
    markers = {'OLS': 'x', 'Ridge': 'o', 'Lasso': 's', 'ElasticNet': '^'}

    for model in models:
        df_model = df[df['Model'] == model]
        best_per_degree = df_model.groupby('Degree')['Mean_MSE'].min().reset_index()
        ax1.plot(
            best_per_degree['Degree'], 
            best_per_degree['Mean_MSE'], 
            label=model, 
            color=colors[model], 
            marker=markers[model], 
            linestyle='-'
        )
    
    ax1.set_yscale('log')
    ax1.set_xlabel('Polynomial Degree')
    ax1.set_ylabel('Mean CV MSE (Log Scale)')
    ax1.set_title(f'Phase {phase_num}: Model Comparison Across Degrees')
    ax1.set_xticks(sorted(df['Degree'].unique()))
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend()

    ax2 = axes[1]
    df_winner = df[df['Model'] == winning_model_name]
    
    best_winner_curve = df_winner.loc[df_winner.groupby('Degree')['Mean_MSE'].idxmin()].sort_values('Degree')

    ax2.errorbar(
        best_winner_curve['Degree'], 
        best_winner_curve['Mean_MSE'], 
        yerr=best_winner_curve['SE_MSE'],
        fmt='-o', 
        color=colors[winning_model_name], 
        ecolor='gray', 
        elinewidth=1.2, 
        capsize=3, 
        label=f'{winning_model_name} Curve ($\pm 1$ SE)'
    )

    ax2.axhline(
        threshold, 
        color='black', 
        linestyle='--', 
        alpha=0.7, 
        label=f'1-SE Cutoff ({threshold:.4f})'
    )
    
    ax2.scatter(
        abs_min_row['Degree'], 
        abs_min_row['Mean_MSE'], 
        color='blue', 
        s=120, 
        zorder=5, 
        label=f'Absolute Min (Deg {abs_min_row["Degree"]}, MSE {abs_min_mse:.4f})'
    )
    
    ax2.scatter(
        winner_row['Degree'], 
        winner_row['Mean_MSE'], 
        color='gold', 
        edgecolors='black', 
        s=180, 
        marker='*', 
        zorder=6, 
        label=f'Selected 1-SE Model (Deg {winner_row["Degree"]}, MSE {winner_row["Mean_MSE"]:.4f})'
    )

    ax2.set_xlabel('Polynomial Degree')
    ax2.set_ylabel('Mean CV MSE')
    ax2.set_title(f'Phase {phase_num}: 1-SE Selection Diagnostic ({winning_model_name})')
    ax2.set_xticks(sorted(df['Degree'].unique())[::2] if len(df['Degree'].unique()) > 10 else sorted(df['Degree'].unique()))
    ax2.grid(True, ls="--", alpha=0.5)
    ax2.legend()

    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"Plot successfully saved: {output_png}")

if __name__ == "__main__":
    plot_phase_diagnostics(
        csv_file='cv_results_var1.csv',
        phase_num=1,
        winning_model_name='Lasso',
        output_png='phase1_cv_analysis.png'
    )

    plot_phase_diagnostics(
        csv_file='cv_results_var2.csv',
        phase_num=2,
        winning_model_name='Ridge',
        output_png='phase2_cv_analysis.png'
    )