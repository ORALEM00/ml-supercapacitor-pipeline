import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

def plot_metric_scores(data, title, filename):
    """
    Generates a bar plot comparing the results from the three methodologies (TrainTest split,
    RandomCV, Grouped CV) both with and without outliers.

    Parameters:
        data (dict): A dictionary containing the scores.
            Expected format:
            {
                'labels': list of strings,
                'metric': str
                'with_outliers_means': list of floats,
                'with_outliers_stds': list of floats,
                'without_outliers_means': list of floats,
                'without_outliers_stds': list of floats
            }
        title (str): The title of the plot.
        filename (str): The name of the file to save the plot.
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
    plt.savefig(filename, dpi=300)
    plt.show()


# Feature frequency
""" def feature_frequency(df, features, select_stable = False, threshold = 0.5,
                      group = "Electrode_ID", target = "Specific_Capacitance"): 
    # Flatten list
    selected_features = np.array([f for sublist in features for f in sublist])
    # Count and group per feature
    unique, counts = np.unique(selected_features, return_counts=True)

    # Threshold for selecting stable features
    threshold = threshold 
    n_folds = len(features) 
    # Minimum appearances to be considered stable
    min_appearances = int(n_folds * threshold)

    # Select stable features
    stable_features = []
    for col, count in zip(unique,counts): 
        if count >= min_appearances:
            stable_features.append(col)
            
    if select_stable == True:
        return stable_features

    X = df.copy()
    if group is not None: 
        X = X.drop([group, target], axis = 1)
    else: 
        X = X.drop([target], axis = 1)
        
    # Add columns that do not appear in count
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
    sns.barplot(x=counts, y=unique, palette='viridis')
    plt.xlabel("Frequency", fontsize=14)
    plt.ylabel("Features", fontsize=14)
    #plt.title("Feature Selection Frequency", fontsize=16, weight="bold", pad=15)
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    # Add value labels
    for i, v in enumerate(counts):
        plt.text(v + 0.1, i, str(v), va='center', fontsize=10)

    # Improve style
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

    return unique, counts """


def feature_frequency(X, y, features, return_stable = False, threshold = 0.5): 
    # Flatten list
    selected_features = np.array([f for sublist in features for f in sublist])
    # Count and group per feature
    unique, counts = np.unique(selected_features, return_counts=True)

    # Threshold for selecting stable features
    threshold = threshold 
    n_folds = len(features) 
    # Minimum appearances to be considered stable
    min_appearances = int(n_folds * threshold)

    # Select stable features
    stable_features = []
    for col, count in zip(unique,counts): 
        if count >= min_appearances:
            stable_features.append(str(col))
            
    if return_stable == True:
        return stable_features
        
    # Add columns that do not appear in count
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
    sns.barplot(x=counts, y=unique, palette='viridis')
    plt.xlabel("Frequency", fontsize=14)
    plt.ylabel("Features", fontsize=14)
    #plt.title("Feature Selection Frequency", fontsize=16, weight="bold", pad=15)
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    # Add value labels
    for i, v in enumerate(counts):
        plt.text(v + 0.1, i, str(v), va='center', fontsize=10)

    # Improve style
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

    return unique, counts