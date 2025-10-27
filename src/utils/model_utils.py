import numpy as np
from sklearn.model_selection import BaseCrossValidator


################### CHECCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCKKKKKKKKKKKKKK ########
# It must always go before removing the specific capacitance and Electrode_ID
def drop_outliers(X, max_value = 800):
    values_drop = X[X["Specific_Capacitance"] > max_value].Electrode_ID
    for id_e in values_drop: 
        X = X.drop(X[X["Electrode_ID"] == id_e].index)
    return X   


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
    if params:
        try:
            model = model_class(random_state=random_state, **params)
        except TypeError:
            model = model_class(**params)
    else:
        try:
            model = model_class(random_state=random_state)
        except TypeError:
            model = model_class()
    return model


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