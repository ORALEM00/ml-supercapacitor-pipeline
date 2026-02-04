import os
import shap
import numpy as np
import matplotlib.pyplot as plt

from src.leakproof_ml.utils import  load_results_from_json
from src.leakproof_ml.plots import plot_predictions, histogram_errors, plot_interpretability_bar 
from src.leakproof_ml.plots import interpretability_comparison_plot

# Models to obtain results for plotting
models_name = ['XGBRegressor', 'Ridge', 'CatBoostRegressor', 'VotingRegressor_0']

results ={}
for model in models_name:
    # Create output directory   
    output_path = f"resulting_plots/{model}/"
    os.makedirs(os.path.dirname(output_path), exist_ok = True)

    # Metric plots
    for method in ["trainTest","trainTest_removed","randomCV","randomCV_removed","groupedCV", 
           "groupedCV_removed"]: 
        # Load results per methdology
        results = load_results_from_json(f"raw_results/{model}/tuned/{method}.json")

        # Create metric directory
        metric_path = os.path.join(output_path, "metrics")
        os.makedirs(metric_path, exist_ok=True)

        # Store plots per methodology
        plot_predictions(results['y_true'], results['y_predict'], 
                         filename=f"{metric_path}/{method}_predictions.png")
        histogram_errors(results['y_true'], results['y_predict'], 
                         filename=f"{metric_path}/{method}_histogram_errors.png")

    # Interpretability plots
    for method in ["pi", "shap"]:
        for key in ['trainTest', 'trainTest_removed', 'randomCV', 'randomCV_removed', 'groupedCV', 'groupedCV_removed']:
            res = load_results_from_json(f"raw_interpretability_results/{model}/{method}/{method}_{key}.json")
            results[f"{method}_{key}"] = res

            # Create method directory
            method_path = os.path.join(output_path, method)
            os.makedirs(method_path, exist_ok=True)

            # PLot individual bars
            plot_interpretability_bar(results[f"{method}_{key}"], 
                           title = f"{method} Feature Importance for {model}", 
                           filename=f"{method_path}/{method}_{key}.png",
                           method=method)
            
            if method == 'shap':
                plt.figure()
                # Also plot SHAP summary plot
                shap.summary_plot(
                    np.array(results[f"{method}_{key}"]["shap_values"]), 
                    np.array(results[f"{method}_{key}"]['X_test']), 
                    results[f"{method}_{key}"]['features'],
                    show=False
                )
                plt.tight_layout()
                plt.savefig(f"{method_path}/shap_summary_{key}.png", dpi=300)
                plt.close()
    
    # Plot comprehensive comparison plots shap
    interpretability_comparison_plot(
        results['shap_groupedCV_removed'], results['shap_groupedCV'],
        results['shap_randomCV_removed'], results['shap_randomCV'],
        results['shap_trainTest_removed'], results['shap_trainTest'], 
        method='shap', title=f'Comprehensive SHAP Importance Comparison for {model}', 
        filename=f"{output_path}/final_shap_comparison_6bar.png"
        )

    # Plot comprehensive comparison plots permutation
    interpretability_comparison_plot(
        results['pi_groupedCV_removed'], results['pi_groupedCV'],
        results['pi_randomCV_removed'], results['pi_randomCV'],
        results['pi_trainTest_removed'], results['pi_trainTest'], 
        filename=f"{output_path}/final_pi_comparison_6bar.png",
        method='perm', title=f'Comprehensive Permutation Importance Comparison for {model}'
        )