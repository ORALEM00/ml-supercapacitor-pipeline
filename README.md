
# Leakproof ML

**Leakproof ML** is a an open-source, flexible, and simple to use Python package designed to systematically prevent data leakage across a complete modelling process. Focused on the most common sources of data leakage arising from improper validation strategies and inadequate isolation between training and test data. 

## Install

Leakproof ML can be installed from [PyPI](https://pypi.org/project/shap): 

<pre>
pip install leakproof_ml
</pre>

## Data Leakage Framework
Leakproof provides an unifed framework of leakafe-aware properties across its main functionalities. This is done by enforcing a standardized implementation of ML workflows to ensure that preprocessing, feature selection, tuning, and fitting are performed exclusively on the training sets. While, promoting the use of splitting strategies aligned with the structure of the data. 

## Quick start
The software provides three main functionalities integrated into the data leakage framework: training, tuning, and interpretability. 

Each functionality can be applied to both a standard single train-test split, and a cross-validation implementation for small data cases. 

```python
# Setting for the example
import xgboost
from src.leakproof_ml.validation import ShuffledGroupKFold

df = pd.read_csv("data.csv")

X = df.drop(columns=["target", "group_id"])
y = df["target"]
groups = df["group_id"]

# Splitter for group based splitter 
# (however can be any splitter)
splitter = ShuffledGroupKFold(n_splits = 10, random_state = 42)
```

### Training
The simplest function where a model can be used to fit in the dataset, avoiding data leakage in an easy way. 
```python
from leakproof_ml import cv_analysis
from leakproof_ml.plots import plot_predictions

# The class of the model is passed as parameter
# Results are gathered in dictionary format
results = cv_analysis(X, y, XGBRegressor, splitter, groups=groups, params = {"max_depth"= 4})

plot_predictions(results['y_true'], results['y_predict'])
```
<p align="center">
  <img width="616" src="./resulting_plots\XGBRegressor\metrics\groupedCV_predictions.png" />
</p>

### Tuning
For hyperparameter optimization, Leakproof ML employs the Tree-structured Parzen Estimator algorithm implemented in the Optuna library. 

In the train-test setting, a CV is applied on the train set to optimize parameters and subsequently evaluated on the held-out test set. In contrast, for the CV setting, Leakproof ML implements a nested CV scheme to avoid a possible optimistic bias present when tuning the parameters using the entire dataset  
```python
from leakproof_ml.tuning import nested_cv_tunning

# For nested cv an extra inner splitter needs to be
# defined
inner_splitter = ShuffledGroupKFold(n_splits = 3, random_state = 42)

# A function accepting parameter trial for Optuna tuning within the framework
def search_space(trial):
  return {
    "max_depth": trial.suggest_int("max_depth", 2, 5),
    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
      }

# Returns in addition the set of parameters optimized
results = nested_cv_tunning(X, y, XGBRegressor, splitter, inner_splitter, search_space, groups=groups) 
```

### Interpretability
To extract physical insights and underlying mechanisms from data-driven models, Leakproof ML uses two global, model-agnostic interpretability methods: permutation importance (PI) and SHAP, which allow for quantification of magnitude and direction of feature influence. By default, PI is used. 
```python
from leakproof_ml.interpretability import cv_interpretability
from leakproof_ml.plots import plot_interpretability_bar 

results = cv_interpretability(X, y, model, splitter, groups=groups)

plot_interpretability_bar(results)
```
<p align="center">
  <img width="616" src="./resulting_plots\XGBRegressor\pi\pi_groupedCV.png" />
</p>

## Custom Pipeline
Apart from the default pipelines in the functions, the package allows for any custom pipeline to be implemented within the functions. To construct a custom pipeline a function returning the pipeline must be used as parameter in the functions. With the final step of the pipeline always defining the model as: ('model', model). 

```python
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from leakproof_ml import cv_analysis

# Custom pipeline 
def polynomial_custom_factory(model, degree=2):
  numeric_pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
  ])
  preprocessor = ColumnTransformer(
    transformers=[
      ('num', numeric_pipe, make_column_selector(dtype_include='float64')),
    ],
    remainder='passthrough'
  )

  # Pipeline steps
  pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('poly', PolynomialFeatures(degree=degree)),
    ('model', model)
  ])
  return pipe

results = cv_analysis(X, y, XGBRegressor, splitter, groups=groups, params = {"max_depth"= 4}, pipeline_factory = polynomial_custom_factory)
```

## Citation
If used in a research project, please cite paper "Leakproof ML: Data Leakage Prevention with a Robust, Interpretable, and Reproducible Machine Learning Framework": 

<details open>
<summary>BibTeX</summary>

```bibtex
@inproceedings{,
  title={},
  author={},
  booktitle={},
  pages={},
  year={}
}
```
</details>


## License

MIT License (see [LICENSE](./LICENSE.txt)).