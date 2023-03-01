"""
This module defines the following routines used by the 'transform' step of the regression recipe:

- ``transformer_fn``: Defines customizable logic for transforming input data before it is passed
  to the estimator during model inference.
"""
import databricks.automl_runtime
from databricks.automl_runtime.sklearn.column_selector import ColumnSelector
from pandas import Timestamp
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from databricks.automl_runtime.sklearn import DatetimeImputer
from databricks.automl_runtime.sklearn import OneHotEncoder
from databricks.automl_runtime.sklearn import TimestampTransformer
from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
import pandas as pd


def col_selector():
    supported_cols = ["event_count", "gender", "total_amount", "creation_date",\
                  "country", "last_transaction", "order_count", "last_event",\
                  "total_item", "days_since_last_activity", "canal", "days_last_event",\
                  "days_since_creation", "session_count", "age_group",\
                  "last_activity_date", "platform"]
    
    return ColumnSelector(supported_cols)

def datetime_prep():
 
    ts_col = ["creation_date", "last_activity_date", "last_event", "last_transaction"]

    imputers = {
      "creation_date": DatetimeImputer(),
      "last_activity_date": DatetimeImputer(),
      "last_event": DatetimeImputer(),
      "last_transaction": DatetimeImputer(),
    }

    datetime_transformers = []

    for col in ts_col:
        ohe_transformer = ColumnTransformer(
            [("ohe", OneHotEncoder(sparse=False, handle_unknown="indicator"), [TimestampTransformer.HOUR_COLUMN_INDEX])],
            remainder="passthrough")
        timestamp_preprocessor = Pipeline([
            (f"impute_{col}", imputers[col]),
            (f"transform_{col}", TimestampTransformer()),
            (f"onehot_encode_{col}", ohe_transformer),
            (f"standardize_{col}", StandardScaler()),
        ])
        datetime_transformers.append((f"timestamp_{col}", timestamp_preprocessor, [col]))
        
    return datetime_transformers

def numerical_prep():
    num_imputers = []
    num_imputers.append(("impute_mean", SimpleImputer(), ["age_group", "days_last_event", "days_since_creation", "days_since_last_activity", "event_count", "gender", "order_count", "session_count", "total_amount", "total_item"]))

    numerical_pipeline = Pipeline(steps=[
        ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors="coerce"))),
        ("imputers", ColumnTransformer(num_imputers)),
        ("standardizer", StandardScaler()),
    ])

    numerical_transformers = [("numerical", numerical_pipeline, ["event_count", "gender", "total_amount", "order_count", "total_item", "days_since_last_activity", "days_last_event", "days_since_creation", "session_count", "age_group"])]
    
    return numerical_transformers

def categorical_prep():

    one_hot_imputers = []

    one_hot_pipeline = Pipeline(steps=[
        ("imputers", ColumnTransformer(one_hot_imputers, remainder="passthrough")),
        ("one_hot_encoder", OneHotEncoder(handle_unknown="indicator")),
    ])

    categorical_one_hot_transformers = [("onehot", one_hot_pipeline, ["age_group", "canal", "country", "event_count", "gender", "order_count", "platform", "session_count"])]
    
    return categorical_one_hot_transformers



def transformer_fn():
    """
    Returns an *unfitted* transformer that defines ``fit()`` and ``transform()`` methods.
    The transformer's input and output signatures should be compatible with scikit-learn
    transformers.
    """
    col_selector_fn = col_selector()
    datetime_transformers_fn = datetime_prep()
    numerical_transformers_fn = numerical_prep()
    categorical_one_hot_transformers_fn = categorical_prep()
    
    transformers = datetime_transformers_fn + numerical_transformers_fn + categorical_one_hot_transformers_fn 

    preprocessor = ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=0)
    
    return Pipeline([
    ("column_selector", col_selector_fn),
    ("preprocessor", preprocessor),
    ],
        verbose=True,)
    
