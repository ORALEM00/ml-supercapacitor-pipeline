import numpy as np
from sklearn.model_selection import BaseCrossValidator

class ShuffledGroupKFold(BaseCrossValidator):
    """
    Custom implementation of Group K-Fold cross-validation that allows 
    optional shuffling of groups before splitting.

    This splitter ensures that the same group is not represented in both
    training and testing sets. It works similar to scikit-learn's 
    `GroupKFold`, but  adds the ability to shuffle groups prior to splitting,
    which can help reduce bias when the order of groups is not random.

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
        """
        Generate indices to split data into training and test sets.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), optional
            The target variable for supervised learning problems.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into 
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        # Ensure groups are provided
        if groups is None: 
            raise ValueError("The 'groups' parameter must be specified for ShuffledGroupKFold.")
            
        # Extract unique groups 
        unique_groups = np.array(list(dict.fromkeys(groups)))
        # Shuffle groups if random_state is set
        if self.random_state is not None:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(unique_groups)

        # Split  unique groups into n_splits 
        folds = np.array_split(unique_groups, self.n_splits)

        # Index Generation: Iterate through folds to create train/test masks
        for fold in folds:
            test_mask = groups.isin(fold)
            # Convert mask to integer indices for scikit-learn compatibility
            train_idx = np.where(~test_mask)[0]
            test_idx = np.where(test_mask)[0]
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits