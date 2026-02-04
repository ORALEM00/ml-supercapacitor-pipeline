from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler

from .selector import CorrelationSelector


def model_pipeline(model):
    """
    Create a model pipeline with preprocessing for continuous variables. 

    Implements scikit-learn Pipeline that standarize continuous variables (float64) columns 
    using StandardScaler while leaving the other columns unchanged

    Parameters: 
        model (object): An instantiated scikit-learn compatible model
    """
    #Create Column Transformer
    preprocessor = ColumnTransformer(
        transformers = [
            ('continuous', StandardScaler(), make_column_selector(dtype_include='float64')),
        ],
    remainder='passthrough',  # Leave other columns unchanged
    verbose_feature_names_out=False,
    )
    #pipeline 
    my_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor), 
        ('model', model)
    ])

    return my_pipeline




def feature_pipeline(model, threshold=0.68):
    """
    Create a model pipeline with preprocessing for continuous variables, and automated feature
    selection using pearson correlation. 

    Implements scikit-learn Pipeline that standarize continuous variables (float64) columns 
    using StandardScaler while leaving the other columns unchanged. Further, it applies
    the custom CorrelationSelector to group correlated features (>threshold) and select through 
    Ridge the best feature and drop the others. 

    Parameters: 
        model (object): An instantiated scikit-learn compatible model.
    """
    # Create Column Transformer
    preprocessor = ColumnTransformer(
        transformers = [
            ('continuous', StandardScaler(), make_column_selector(dtype_include='float64')),
        ],
    remainder='passthrough' , # Leave other columns unchanged
    verbose_feature_names_out=False,
    )
    
    # To mantain feature names after transformation
    preprocessor.set_output(transform="pandas")

    # Pipeline 
    my_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor), 
        ('feature_selection', CorrelationSelector(threshold=threshold)),
        ('model', model)
    ])

    return my_pipeline