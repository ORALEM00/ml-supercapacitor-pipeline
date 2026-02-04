import numpy as np



def drop_outliers(df, target_column, group_id_colum  = None, iqr_factor = 1.5):
    """
    Remove outliers from a DataFrame based on the Interquartile Range (IQR) method.

    This function identifies outliers in the target column and filters them out. 
    If a group ID is provided, the function treats outliers at the group level: 
    if a single sample in a group is an outlier, the entire group is removed 
    to prevent biased or incomplete time-series/sensor analysis.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataset containing the target values and optional group IDs.
    target_column : str
        The name of the column used to calculate outlier thresholds.
    group_id_colum : str, optional
        The column name identifying related samples (e.g., 'electrode_id', 
        'subject_id'). If provided, any group containing at least one outlier 
        will be dropped entirely.
    iqr_factor : float, default=1.5
        The multiplier for the IQR to determine the outlier bounds. 
        A factor of 1.5 is standard; 3.0 is typically used for "extreme" outliers.

    Returns
    -------
    df_cleaned : pandas.DataFrame
        A copy of the original DataFrame with outliers (and their associated 
        groups, if applicable) removed.
    """
    # Calculate quartiles
    Q1 = df[target_column].quantile(0.25)
    Q3 = df[target_column].quantile(0.75)
    # Inter-quartile range
    IQR = Q3 - Q1

    # Thresholds for outliers detection
    lower_bound = Q1 - IQR * iqr_factor
    upper_bound = Q3 + IQR * iqr_factor

    # Handle Group-Level Removal: drop all samples belonging to an outlier group
    if group_id_colum is not None:
        lower_groups = df[df[target_column] < lower_bound][group_id_colum].unique()
        upper_groups = df[df[target_column] > upper_bound][group_id_colum].unique()

        # Consolidate all unique IDs identified as outliers
        outlier_groups = np.concatenate([lower_groups, upper_groups])
        
        # Filter the DataFrame to exclude all rows associated with these group IDs
        df_cleaned = df[~df[group_id_colum].isin(outlier_groups)].copy()

    # Handle Row-Level Removal: drop only individual rows identified as outliers
    else:
        df_cleaned = df[(df[target_column] >= lower_bound) & (df[target_column] <= upper_bound)] 
    return df_cleaned