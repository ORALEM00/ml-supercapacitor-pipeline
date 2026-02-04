import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns




def residual_errors(y, y_predictions, filename = None):
    """
    Generate a scatter plot of residuals against actual values.

    Parameters
    ----------
    y : pandas.Series or array-like
        The true target values.
    y_predictions : pandas.Series or array-like
        The values predicted by the model.
    filename : str, optional
        The system path and name to save the plot (e.g., 'results/residuals.png').
        If None, the plot is displayed interactively.

    Returns
    -------
    None
    """
    # Calculate residuals
    residuals = y - y_predictions

    # Plot residuals
    plt.scatter(y, residuals, alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Actual Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    if filename:
        plt.savefig(filename, dpi=300)
        plt.close()
    else:
        plt.show()
    



def histogram_errors(y, y_predictions, filename = None):
    """
    Generate a histogram of residual errors.

    Parameters
    ----------
    y : pandas.Series or array-like
        The true target values.
    y_predictions : pandas.Series or array-like
        The values predicted by the model.
    filename : str, optional
        Path to save the resulting plot image.

    Returns
    -------
    None
    """
    # Calculate residuals
    residuals = y - y_predictions
    # Plot histogram of residuals
    plt.hist(residuals, bins=30, color='blue', edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--')
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title("Histogram of Residual Errors")
    if filename:
        plt.savefig(filename, dpi=300)
        plt.close()
    else:
        plt.show()




def plot_predictions(y, y_predictions, filename = None):
    """
    Create a scatter plot comparing actual values against model predictions.

    Parameters
    ----------
    y : pandas.Series or array-like
        The true target values.
    y_predictions : pandas.Series or array-like
        The values predicted by the model.
    filename : str, optional
        Path to save the resulting plot image.

    Returns
    -------
    None
    """
    # Scatter plot of predicted vs actual values
    plt.scatter(y, y_predictions, edgecolor='k', alpha=0.7)
    plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual Values')
    if filename:
        plt.savefig(filename, dpi=300)
        plt.close()
    else:
        plt.show()




def plot_metric_scores(data, title, filename = None):
    """
    Compare performance metrics across different methodologies and data states.

    Generates a grouped bar plot with error bars (standard deviation). This 
    visualization specifically highlights the difference in mean scores and 
    stability when outliers are removed versus when they are kept, across 
    various cross-validation or splitting strategies.

    Parameters
    ----------
    data : dict
        A dictionary structured with the following keys:
        - 'labels': list of str, names of the methodologies (e.g., ['ShuffleSplit', 'GroupKFold']).
        - 'metric': str, the name of the metric being plotted (e.g., 'R2', 'MSE').
        - 'with_outliers_means': list of float, mean scores for the original data.
        - 'with_outliers_stds': list of float, standard deviation for the original data.
        - 'without_outliers_means': list of float, mean scores after outlier removal.
        - 'without_outliers_stds': list of float, standard deviation after outlier removal.
    title : str
        The main title for the plot.
    filename : str, optional
        The system path and filename to save the resulting image. 
        If None, the plot is displayed but not saved.

    Returns
    -------
    None
    """
    # Create the figure and axes
    plt.style.use('seaborn-v0_8-whitegrid') # A clean, professional style
    data_size = len(data['labels'])
    fig, ax = plt.subplots(figsize=(6 * data_size/7, 6))
    
    color1_dark, color1_light = '#1f77b4', '#aec7e8'  # Azul

    # Define bar width and positions
    bar_width = 0.1
    x = np.arange(len(data['labels'])) * 0.4

    # With outliers bar
    rects1 = ax.bar(x - bar_width*0.7, data['with_outliers_means'], bar_width,
           yerr=data['with_outliers_stds'], capsize=5,
           label='With Outliers', color=color1_light, edgecolor='black')

    # Withot outliers bar
    rects2 = ax.bar(x + bar_width*0.7, data['without_outliers_means'], bar_width,
           yerr=data['without_outliers_stds'], capsize=5,
           label='Without Outliers', color=color1_dark, edgecolor='black')

    # Plot details
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel(f"Mean {data['metric']} Score", fontsize=12)
    #plt.yticks(np.arange(0, 1, 0.1)) 
    ax.set_xticks(x)
    ax.set_xticklabels(data['labels'], rotation=45, ha='right', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    #ax.set_ylim(0) # R-squared typically ranges from 0 to 1


    # Customize the grid and spines
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.grid(axis='x', visible=False) # Remove vertical grid lines
    
    # Add std text labels
    for i in range(len(data['labels'])):
        # With Outliers
        
        y_pos_with = data['with_outliers_means'][i] + data['with_outliers_stds'][i] + 0.01
        ax.text(x[i] - bar_width*0.65, y_pos_with,
                f"{data['with_outliers_means'][i]:.2f}",
                ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Without outliers
        y_pos_without = data['without_outliers_means'][i] + data['without_outliers_stds'][i] + 0.01
        ax.text(x[i] + bar_width*0.65, y_pos_without,
                f"{data['without_outliers_means'][i]:.2f}",
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Ensure a tight layout and save the figure
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300)
        plt.close()
    else:
        plt.show()




# Feature frequency plot
def feature_frequency(
        X, y, 
        features, 
        filename = None
        ): 
    """
    Analyze and visualize the selection frequency of features across CV folds.

    In a nested cross-validation or multiple-run setting, different features 
    may be selected in different folds. This function counts those occurrences 
    to identify "stable" features—those that meet a specific selection 
    threshold.

    Parameters
    ----------
    X : pandas.DataFrame
        The design matrix used to extract the full list of potential features.
    y : pandas.Series or array-like
        The target vector (included for API consistency).
    features : list of lists
        A list containing the feature names selected in each fold 
        (e.g., [[feat1, feat2], [feat1, feat3], ...]).
    filename : str, optional
        Path to save the generated bar plot.

    Returns
    -------
    stable_features : list of str (if return_stable=True)
        List of features meeting the selection threshold.
    unique, counts : ndarray, ndarray (if return_stable=False)
        All unique features and their corresponding selection counts, 
        sorted by frequency.
    """
    # Flatten the list of features from all folds
    selected_features = np.array([f for sublist in features for f in sublist])
    # Identify and count unique features
    unique, counts = np.unique(selected_features, return_counts=True)
        
    # Add columns with zero counts (not selected in any fold)
    for col in X.columns: 
        if col not in unique: 
            unique = np.append(unique, col)
            counts = np.append(counts, 0)
            
    # Sort by frequency (descending)
    sort_idx = np.argsort(-counts)  # descending order
    unique = unique[sort_idx]
    counts = counts[sort_idx]
    
    # Plot - horizontal bar
    plt.figure(figsize=(8, max(4, len(unique) * 0.3)))
    sns.barplot(x=counts, y=unique, hue=unique, palette='viridis', 
                legend=False)
    # Label axis
    plt.xlabel("Frequency", fontsize=14)
    plt.ylabel("Features", fontsize=14)
    # plt.title("Feature Selection Frequency", fontsize=16, weight="bold", pad=15)
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    # Add explicit count labels to the end of each bar for clarity
    for i, v in enumerate(counts):
        plt.text(v + 0.1, i, str(v), va='center', fontsize=10)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300)
        plt.close()
    else:
        plt.show()

    return unique, counts