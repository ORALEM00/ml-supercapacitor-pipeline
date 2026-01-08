import numpy as np
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import BaseCrossValidator


################### CHECCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCKKKKKKKKKKKKKK ########

# Drop outliers based on complete electrodes or row based
def drop_outliers(df, target_column, group_id_colum  = None, iqr_factor = 1.5):
    # Calculate quartiles
    Q1 = df[target_column].quantile(0.25)
    Q3 = df[target_column].quantile(0.75)
    # Inter-quartile range
    IQR = Q3 - Q1

    # Thresholds
    lower_bound = Q1 - IQR * iqr_factor
    upper_bound = Q3 + IQR * iqr_factor

    # If given an group id column
    if group_id_colum is not None:
        # Identify electrodes with outside the region
        lower_groups = df[df[target_column] < lower_bound][group_id_colum].unique()
        upper_groups = df[df[target_column] > upper_bound][group_id_colum].unique()
        # Join list
        outlier_groups = np.concatenate([lower_groups, upper_groups])
        
        # Drop rows with that group id
        df_cleaned = df[~df[group_id_colum].isin(outlier_groups)].copy()

    # If there is no group id just drop individual rows 
    else:
        df_cleaned = df[(df[target_column] >= lower_bound) & (df[target_column] <= upper_bound)] 
    return df_cleaned


class CustomGroupKFold(BaseCrossValidator):
    """
    Custom implementation of Group K-Fold cross-validation that preserves 
    the original order of groups while allowing optional random shuffling.

    This splitter ensures that the same group is not represented in both
    training and testing sets. It works similar to scikit-learn's 
    `GroupKFold`, but  it preserves their original order or applies 
    a reproducible random shuffle if a random seed is provided. Compatible
    with scikit-learn. 

    Arguments: 
        n_splits : int
            Number of folds (at least 2)
        random_state: int
            Random seed used to shuffle the unique groups before splitting.

    Methods: 
        split(X, y, groups):
            Generates indices to split data into training and test sets. 
        get_n_splits(X=None, y=None, groups=None):
            Returns the number of splitting iterations. 

    """
    def __init__(self, n_splits=10, random_state=None):
        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, X, y, groups):
        if groups is None: 
            raise ValueError("The 'groups' parameter must be specified for CustomGroupKFold.")
            
        # Get the unique groups in the order provided, without re-sorting
        unique_groups = np.array(list(dict.fromkeys(groups)))  # preserves order
        if self.random_state is not None:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(unique_groups)
        # Now, split these unique groups into n_splits parts
        folds = np.array_split(unique_groups, self.n_splits)
        for fold in folds:
            test_mask = groups.isin(fold)
            train_idx = np.where(~test_mask)[0]
            test_idx = np.where(test_mask)[0]
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

# Dictionary to suppress verbosity of models 
VERBOSITY_KWARGS = {
    'XGBRegressor': {"verbosity": 0, "silent": True},
    'LGBMRegressor': {"verbose": -1, "verbosity": -1},
    'CatBoostRegressor': {"verbose": False},
    'MLPRegressor': {"verbose": False},
    # Add others if ever needed
}


def _get_model_instance(model_class, params=None, random_state = 42):
    """
    Instantiates a scikit-learn model, handling models that do not
    accept the random_state parameter.

    Parameters:
        model_class (class): The scikit-learn model class to be used.
        params (dict): Optional dictionary of hyperparameters for the model.
        random_state (int): The random seed for reproducibility.

    Returns:
        model: An instantiated model object.
    """
    model_name = model_class.__name__
    # Add verbosity suppression if applicable
    silence_params = VERBOSITY_KWARGS.get(model_name, {})

    if params:
        # To ensure dictionary
        if not isinstance(params, dict):
            raise TypeError(
                    f"Param must be a dictionary, but got instead {type(params).__name__}."
                )
        
        try:
            model = model_class(random_state=random_state, **params, **silence_params)
        except TypeError:
            model = model_class(**params, **silence_params)
    else:
        try:
            model = model_class(random_state=random_state, **silence_params)
        except TypeError:
            model = model_class(**silence_params)
    return model

# Creates instance of voting regressor from the lists of models class passed
def _get_voting_regressor_instance(models_class_list, params_list = None, weights = None, random_state = 42):
    # List of models and of params must be same length (if parameters are given)
    # Each model will have its paramater dictionary on the list.
    if params_list is not None and len(models_class_list) != len(params_list):
        raise ValueError(
            f"Length mismatch: models_class_list has {len(models_class_list)} elements, "
            f"but params_list has {len(params_list)} elements."
        ) 
    
    model_tupple = []
    for i in range(len(models_class_list)):
        # Parameters must be in same order as model classes in list
        if params_list is not None:  
            model_instance = _get_model_instance(models_class_list[i], params = params_list[i], random_state = random_state)
        else:
            model_instance = _get_model_instance(models_class_list[i], random_state = random_state)

        model = (f"model_{i}", model_instance)
        # Add to list of model tupples
        model_tupple.append(model)
    # Use model_tupple to instance the voting regressor
    model_ensemble = VotingRegressor(estimators = model_tupple, weights = weights)
    return model_ensemble


# Function to return dictionary
def _create_cv_results_dict(R2_score, MAE_score, RMSE_score, y_predict, y_true, features = None, params = None):
    """
    Helper function to create a standardized dictionary of results.
    """
    results = {
        'R2_score': R2_score,
        'MAE_score': MAE_score,
        'RMSE_score': RMSE_score,
        'y_predict': y_predict,
        'y_true': y_true,
    }
    if features is not None:
        results['features'] = features
    if params is not None:
        results['params'] = params

    return results

# Function to validate parameters of cv
def _validate_cv_params(i,model_class, params):
    if params is not None:
            # More than one model (Voting Regressor)
            if isinstance(model_class, list): 
                if np.ndim(params) == 2:
                    current_params = [p[i] for p in params]
                else:
                    # List of dictionaries for each model
                    current_params = params 
            else:
                if (isinstance(params, list)):
                    current_params = params[i]
                else:
                    current_params = params
    else: 
        current_params = None

    return current_params

# Validation function to raise errors 
def _validate_cv_inputs(model_class, cv_splitter, params):
    """
    Ensures that inputs are correctly formated, and if not
    Raise Errors with the specific information of error source
    """
    n_folds = getattr(cv_splitter, "n_splits", None)

    # Case 1: No parameters, so no error
    if params is None:
        return
    
    # Case 2: Model list for Voting Regressor
    if isinstance(model_class, list):
        n_models = len(model_class) # Number of models
        params_dim = np.ndim(params)

        # Fixed parameters (1D list)
        if params_dim == 1:
            if not all(isinstance(p, dict) for p in params):
                raise TypeError(
                    "For a VotingRegressor, when passing a single list of parameters, "
                    "each element must be a dictionary corresponding to a model."
                    "An element passed is not a dictionary."
                )
            # Lenght error is already handled inside the _get_voting_regressor_instance
        
        # Parameters change per fold (2D list)
        if params_dim == 2:
            # Not given enough parameters for all models
            if n_models != len(params):
                raise ValueError(
                    f"Params expected {n_models} lists of parameter dictionaries (one per model), "
                    f"but got {len(params)}."
                )
            
            # List of parameter contain less parameters than number of folds
            for fold_idx, fold_params in enumerate(params):
                if len(fold_params) != n_folds:
                    raise ValueError(
                        f"Params expected {n_models} lists of parameter dictionaries (one per model), "
                        f"each list containing {n_folds} dictionaries (one per fold), "
                        f"but instead got {len(fold_params)} in fold {fold_idx}."
                    )
        
        if params_dim > 2:
            raise ValueError(
                f"Invalid params shape for VotingRegressor: expected 1D or 2D list, "
                f"got {params_dim}D."
            )
        
    # Case 3: Single model

    # params must be either just one dictionary
    # or n depending the number of n_folds
    else:
        # List of parameters
        if isinstance(params,list):
            # Lenght mismatch 
            if len(params) != n_folds:
                raise ValueError(
                    f"Length mismatch: Params expected a single dictionary or a list of {n_folds} dictionaries"
                    f"(one per fold), but instead got {len(params)}."
                ) 
            # All must be dictionaries
            elif not all(isinstance(p, dict) for p in params):
                raise TypeError(
                    f"Params expected a single dictionary or a list of {n_folds} dictionaries (one per fold),"
                    "an element in the list passed is not a dictionary."
                )
        # Single dictionary case
        elif not isinstance(params, dict):
            raise TypeError(
                "Params expected a single dictionary or a list of dictionaries (one per fold),"
                f"but got instead {type(params).__name__}."
            )
            

    