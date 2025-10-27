import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


def model_pipeline(df_X, model):
    """
    Create a model pipeline with preprocessing for continuous variables. 

    Implements scikit-learn Pipeline that standarize continuous variables (float64) columns 
    using StandardScaler while leaving the other columns unchanged

    Parameters: 
        df_X (pd.Dataframe): Dataset without target column
        model (object): An instantiated scikit-learn compatible model
    """
     #Copy to avoid changes on original 
    X = df_X.copy()
    
    #Only scale continuos variables 
    columns_scale  = [col for col in X.columns if X[col].dtypes == "float64"]
    #Create Column Transformer
    preprocessor = ColumnTransformer(
        transformers = [
            ('continuous', StandardScaler(with_mean = False), columns_scale),
        ],
    remainder='passthrough'  # Leave other columns unchanged
    )
    #pipeline 
    my_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor), 
        ('model', model)
    ])

        
    return my_pipeline


class CorrelationSelector(BaseEstimator, TransformerMixin):
    """
    Fearure selector that removes highly correlated features to minimize multicolliniarity.

    This transformers identifies a group of correlated features (>threhold) using Pearson correlation, 
    and retains only the most representative one using Ridge regression coefficients, while dropping
    the rest. 

    Arguments
        threshold : float
            Correlation threshold above which features are considered highly correlated.
        ridge_alpha: 
            Regularization strength for Ridge regression.
        random_state : 
            Random state for reproducibility in the Ridge regression.
    """
    def __init__(self, threshold=0.8, ridge_alpha=1.0, random_state = 42):
        """
        Constructs all the necessary attributes for CorrelationSelector object. 

        Parameters:

            threshold: Correlation threshold above which features are considered highly correlated.
            scoring: Method to determine the best feature in correlated groups.
            ridge_alpha: Regularization strength for Ridge regression.
        """
        self.threshold = threshold
        self.ridge_alpha = ridge_alpha
        self.random_state = random_state
        # Features to keep
        self.to_keep_ = None 

    def fit(self, X, y):
        # Convert to pandas DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Compute absolute Pearson Correlation on all DataFrame
        corr_matrix = X.corr().abs()
        # Return only upper triangular matrix to avoid doble calculations
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Append columns with high correlation
        correlated_groups = []
        # To avoid double checking of features
        already_checked = set()

        for col in upper.columns:
            # Group features with correlation greater than threshold
            high_corr_features = set(upper.index[upper[col] > self.threshold])

            # Check if feature is not already in other group
            if high_corr_features and col not in already_checked:
                high_corr_features.add(col)
                grouped_features = sorted(list(high_corr_features))

                correlated_groups.append(grouped_features)
                already_checked.update(high_corr_features)

        # Determine which feature to keep from each correlated group
        features_to_keep = set(X.columns)
        
        # Ridge coefficients implementation
        for group in correlated_groups:
            model = Ridge(alpha=self.ridge_alpha, random_state = self.random_state)
            # Train ridge with the columns of each group
            model.fit(X[group], y)
            # Compute absolute value 
            scores = np.abs(model.coef_)
            
            # Select the best feature from each group
            best_feature = group[np.argmax(scores)]
            # Remove entire group
            features_to_keep -= set(group)
            # Append the selected feature
            features_to_keep.add(best_feature)

        self.to_keep_ = sorted(list(features_to_keep))
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.to_keep_]
        else:
            X_df = pd.DataFrame(X)
            return X_df[self.to_keep_].values
        

def feature_pipeline(df_X, model):
    """
    Create a model pipeline with preprocessing for continuous variables, and automated feature
    selection using pearson correlation. 

    Implements scikit-learn Pipeline that standarize continuous variables (float64) columns 
    using StandardScaler while leaving the other columns unchanged. Further, it applies
    the custom CorrelationSelector to group correlated features (>threshold) and select through 
    Ridge the best feature and drop the others. 

    Parameters: 
        df_X (pd.Dataframe): Dataset without target column.
        model (object): An instantiated scikit-learn compatible model.
    """
    # Copy to avoid changes on original 
    X = df_X.copy()
    
    # Only scale continuos variables 
    columns_scale  = [col for col in X.columns if X[col].dtypes == "float64"]
    # Create Column Transformer
    preprocessor = ColumnTransformer(
        transformers = [
            ('continuous', StandardScaler(), columns_scale),
        ],
    remainder='passthrough' , # Leave other columns unchanged
    verbose_feature_names_out=False,
    )
    
    # To mantain same names after preprocessor
    preprocessor.set_output(transform="pandas")

    # Pipeline 
    my_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor), 
        ('feature_selection', CorrelationSelector(threshold=0.68)),
        ('model', model)
    ])

    return my_pipeline