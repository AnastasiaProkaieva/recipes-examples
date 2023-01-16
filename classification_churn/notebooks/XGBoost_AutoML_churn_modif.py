# Databricks notebook source
# MAGIC %md
# MAGIC # XGBoost training
# MAGIC This is an auto-generated notebook. To reproduce these results, attach this notebook to the **Shared Autoscaling EMEA** cluster and rerun it.
# MAGIC - Compare trials in the [MLflow experiment](#mlflow/experiments/4156927442295141)
# MAGIC - Navigate to the parent notebook [here](#notebook/4156927442295142) (If you launched the AutoML experiment using the Experiments UI, this link isn't very useful.)
# MAGIC - Clone this notebook into your project folder by selecting **File > Clone** in the notebook toolbar.
# MAGIC 
# MAGIC Runtime Version: _12.0.x-cpu-ml-scala2.12_

# COMMAND ----------

import mlflow
import databricks.automl_runtime

target_col = "churn"
time_col = "last_transaction"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

display(spark.read.table("hive_metastore.dbdemos_lakehouse_churn_quentin_ambard_demo.churn_features"))

# COMMAND ----------

import mlflow
import os
import uuid
import shutil
import pandas as pd
import numpy as np 

# # Create temp directory to download input data from MLflow
# input_temp_dir = os.path.join(os.environ["SPARK_LOCAL_DIRS"], "tmp", str(uuid.uuid4())[:8])
# os.makedirs(input_temp_dir)


# # Download the artifact and read it into a pandas DataFrame
# input_data_path = mlflow.artifacts.download_artifacts(run_id="f37e0fc7c400400b84c59884e14c36f9", artifact_path="data", dst_path=input_temp_dir)

# df_loaded = pd.read_parquet(os.path.join(input_data_path, "training_data"))
# # Delete the temp data
# shutil.rmtree(input_temp_dir)

# # Preview data
# df_loaded.head(5)

# COMMAND ----------

df_loaded = spark.read.table("hive_metastore.dbdemos_lakehouse_churn_quentin_ambard_demo.churn_features").toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Select supported columns
# MAGIC Select only the columns that are supported. This allows us to train a model that can predict on a dataset that has extra columns that are not used in training.
# MAGIC `[]` are dropped in the pipelines. See the Alerts tab of the AutoML Experiment page for details on why these columns are dropped.

# COMMAND ----------

from databricks.automl_runtime.sklearn.column_selector import ColumnSelector
supported_cols = ["event_count", "gender", "total_amount", "creation_date", "country", "last_transaction", "order_count", "last_event", "total_item", "days_since_last_activity", "canal", "days_last_event", "days_since_creation", "session_count", "age_group", "last_activity_date", "platform"]
col_selector = ColumnSelector(supported_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocessors

# COMMAND ----------

# MAGIC %md
# MAGIC ### Datetime Preprocessor
# MAGIC For each datetime column, extract relevant information from the date:
# MAGIC - Unix timestamp
# MAGIC - whether the date is a weekend
# MAGIC - whether the date is a holiday
# MAGIC 
# MAGIC Additionally, extract extra information from columns with timestamps:
# MAGIC - hour of the day (one-hot encoded)
# MAGIC 
# MAGIC For cyclic features, plot the values along a unit circle to encode temporal proximity:
# MAGIC - hour of the day
# MAGIC - hours since the beginning of the week
# MAGIC - hours since the beginning of the month
# MAGIC - hours since the beginning of the year

# COMMAND ----------

from pandas import Timestamp
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from databricks.automl_runtime.sklearn import DatetimeImputer
from databricks.automl_runtime.sklearn import OneHotEncoder
from databricks.automl_runtime.sklearn import TimestampTransformer
from sklearn.preprocessing import StandardScaler

imputers = {
  "creation_date": DatetimeImputer(),
  "last_activity_date": DatetimeImputer(),
  "last_event": DatetimeImputer(),
  "last_transaction": DatetimeImputer(),
}

datetime_transformers = []

for col in ["creation_date", "last_activity_date", "last_event", "last_transaction"]:
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

# COMMAND ----------

# MAGIC %md
# MAGIC ### Numerical columns
# MAGIC 
# MAGIC Missing values for numerical columns are imputed with mean by default.

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

num_imputers = []
num_imputers.append(("impute_mean", SimpleImputer(), ["age_group", "days_last_event", "days_since_creation", "days_since_last_activity", "event_count", "gender", "order_count", "session_count", "total_amount", "total_item"]))

numerical_pipeline = Pipeline(steps=[
    ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors="coerce"))),
    ("imputers", ColumnTransformer(num_imputers)),
    ("standardizer", StandardScaler()),
])

numerical_transformers = [("numerical", numerical_pipeline, ["event_count", "gender", "total_amount", "order_count", "total_item", "days_since_last_activity", "days_last_event", "days_since_creation", "session_count", "age_group"])]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Categorical columns

# COMMAND ----------

# MAGIC %md
# MAGIC #### Low-cardinality categoricals
# MAGIC Convert each low-cardinality categorical column into multiple binary columns through one-hot encoding.
# MAGIC For each input categorical column (string or numeric), the number of output columns is equal to the number of unique values in the input column.

# COMMAND ----------

from databricks.automl_runtime.sklearn import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

one_hot_imputers = []

one_hot_pipeline = Pipeline(steps=[
    ("imputers", ColumnTransformer(one_hot_imputers, remainder="passthrough")),
    ("one_hot_encoder", OneHotEncoder(handle_unknown="indicator")),
])

categorical_one_hot_transformers = [("onehot", one_hot_pipeline, ["age_group", "canal", "country", "event_count", "gender", "order_count", "platform", "session_count"])]

# COMMAND ----------

from sklearn.compose import ColumnTransformer

transformers = datetime_transformers + numerical_transformers + categorical_one_hot_transformers

preprocessor = ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train - Validation - Test Split
# MAGIC The input data is split by AutoML into 3 sets:
# MAGIC - Train (60% of the dataset used to train the model)
# MAGIC - Validation (20% of the dataset used to tune the hyperparameters of the model)
# MAGIC - Test (20% of the dataset used to report the true performance of the model on an unseen dataset)
# MAGIC 
# MAGIC `_automl_split_col_c967` contains the information of which set a given row belongs to.
# MAGIC We use this column to split the dataset into the above 3 sets. 
# MAGIC The column should not be used for training so it is dropped after split is done.
# MAGIC 
# MAGIC Given that `last_transaction` is provided as the `time_col`, the data is split based on time order,
# MAGIC where the most recent data is split to the test data.

# COMMAND ----------

# # AutoML completed train - validation - test split internally and used _automl_split_col_c967 to specify the set
# split_train_df = df_loaded.loc[df_loaded._automl_split_col_c967 == "train"]
# split_val_df = df_loaded.loc[df_loaded._automl_split_col_c967 == "val"]
# split_test_df = df_loaded.loc[df_loaded._automl_split_col_c967 == "test"]

# # Separate target column from features and drop _automl_split_col_c967
# X_train = split_train_df.drop([target_col, "_automl_split_col_c967"], axis=1)
# y_train = split_train_df[target_col]

# X_val = split_val_df.drop([target_col, "_automl_split_col_c967"], axis=1)
# y_val = split_val_df[target_col]

# X_test = split_test_df.drop([target_col, "_automl_split_col_c967"], axis=1)
# y_test = split_test_df[target_col]

#
# Making a train test split myself 
#
from sklearn.model_selection import train_test_split

X = df_loaded.drop([target_col], axis=1)
y = df_loaded[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=62, stratify=y)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=35, stratify=y_train) # 0.25 x 0.8 = 0.2

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train classification model
# MAGIC - Log relevant metrics to MLflow to track runs
# MAGIC - All the runs are logged under [this MLflow experiment](#mlflow/experiments/4156927442295141)
# MAGIC - Change the model parameters and re-run the training cell to log a different trial to the MLflow experiment
# MAGIC - To view the full list of tunable hyperparameters, check the output of the cell below

# COMMAND ----------

from xgboost import XGBClassifier

help(XGBClassifier)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define the objective function
# MAGIC The objective function used to find optimal hyperparameters. By default, this notebook only runs
# MAGIC this function once (`max_evals=1` in the `hyperopt.fmin` invocation) with fixed hyperparameters, but
# MAGIC hyperparameters can be tuned by modifying `space`, defined below. `hyperopt.fmin` will then use this
# MAGIC function's return value to search the space to minimize the loss.

# COMMAND ----------

import mlflow
from mlflow.models import Model, infer_signature, ModelSignature
from mlflow.pyfunc import PyFuncModel
from mlflow import pyfunc
import sklearn
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from databricks.automl_runtime.sklearn import TransformedTargetClassifier

from hyperopt import hp, tpe, fmin, STATUS_OK, Trials

# Create a separate pipeline to transform the validation dataset. This is used for early stopping.
mlflow.sklearn.autolog(disable=True)
pipeline_val = Pipeline([
    ("column_selector", col_selector),
    ("preprocessor", preprocessor),
])
pipeline_val.fit(X_train, y_train)
X_val_processed = pipeline_val.transform(X_val)
label_encoder_val = LabelEncoder()
label_encoder_val.fit(y_train)
y_val_processed = label_encoder_val.transform(y_val)

def objective(params):
  with mlflow.start_run(experiment_id="4156927442295141", run_name="xgboost") as mlflow_run:
    if params:
        if "max_depth" in params:
            # hyperopt supplies values as float but must be int
            params["max_depth"] = int(params["max_depth"])
        if "min_child_weight" in params:
            # hyperopt supplies values as float but must be int
            params["min_child_weight"] = int(params["min_child_weight"])
        if "max_delta_step" in params:
            # hyperopt supplies values as float but must be int
            params["max_delta_step"] = int(params["max_delta_step"])
    xgbc_classifier = TransformedTargetClassifier(
        classifier=XGBClassifier(**params),
        transformer=LabelEncoder()  # XGBClassifier requires the target values to be integers between 0 and n_class-1
    )

    model = Pipeline([
        ("column_selector", col_selector),
        ("preprocessor", preprocessor),
        ("classifier", xgbc_classifier),
    ])

    # Enable automatic logging of input samples, metrics, parameters, and models
    mlflow.sklearn.autolog(
        log_input_examples=True,
        silent=True)

    model.fit(X_train, y_train, classifier__early_stopping_rounds=5, classifier__verbose=False, classifier__eval_set=[(X_val_processed,y_val_processed)])

    
    # Log metrics for the training set
    mlflow_model = Model()
    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.sklearn")
    pyfunc_model = PyFuncModel(model_meta=mlflow_model, model_impl=model)
    X_train[target_col] = y_train
    training_eval_result = mlflow.evaluate(
        model=pyfunc_model,
        data=X_train,
        targets=target_col,
        model_type="classifier",
        evaluator_config = {"log_model_explainability": False,
                            "metric_prefix": "training_" , "pos_label": 1 }
    )
    xgbc_training_metrics = training_eval_result.metrics
    # Log metrics for the validation set
    X_val[target_col] = y_val
    val_eval_result = mlflow.evaluate(
        model=pyfunc_model,
        data=X_val,
        targets=target_col,
        model_type="classifier",
        evaluator_config = {"log_model_explainability": False,
                            "metric_prefix": "val_" , "pos_label": 1 }
    )
    xgbc_val_metrics = val_eval_result.metrics
    # Log metrics for the test set
    X_test[target_col] = y_test
    test_eval_result = mlflow.evaluate(
        model=pyfunc_model,
        data=X_test,
        targets=target_col,
        model_type="classifier",
        evaluator_config = {"log_model_explainability": False,
                            "metric_prefix": "test_" , "pos_label": 1 }
    )
    xgbc_test_metrics = test_eval_result.metrics

    loss = xgbc_val_metrics["val_roc_auc"]

    # Truncate metric key names so they can be displayed together
    xgbc_val_metrics = {k.replace("val_", ""): v for k, v in xgbc_val_metrics.items()}
    xgbc_test_metrics = {k.replace("test_", ""): v for k, v in xgbc_test_metrics.items()}

    return {
      "loss": loss,
      "status": STATUS_OK,
      "val_metrics": xgbc_val_metrics,
      "test_metrics": xgbc_test_metrics,
      "model": model,
      "run": mlflow_run,
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configure the hyperparameter search space
# MAGIC Configure the search space of parameters. Parameters below are all constant expressions but can be
# MAGIC modified to widen the search space. For example, when training a decision tree classifier, to allow
# MAGIC the maximum tree depth to be either 2 or 3, set the key of 'max_depth' to
# MAGIC `hp.choice('max_depth', [2, 3])`. Be sure to also increase `max_evals` in the `fmin` call below.
# MAGIC 
# MAGIC See https://docs.databricks.com/applications/machine-learning/automl-hyperparam-tuning/index.html
# MAGIC for more information on hyperparameter tuning as well as
# MAGIC http://hyperopt.github.io/hyperopt/getting-started/search_spaces/ for documentation on supported
# MAGIC search expressions.
# MAGIC 
# MAGIC For documentation on parameters used by the model in use, please see:
# MAGIC https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier
# MAGIC 
# MAGIC NOTE: The above URL points to a stable version of the documentation corresponding to the last
# MAGIC released version of the package. The documentation may differ slightly for the package version
# MAGIC used by this notebook.

# COMMAND ----------

from sklearn.utils.class_weight import compute_class_weight
def compute_weights(y_train):
    """
    Define minimum positive class scale factor
    """
    weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    scale = weights[1] / weights[0]
    return scale
  
scale = compute_weights(y_train)

# COMMAND ----------

# space = {
#   "colsample_bytree": 0.5921479156652303,
#   "learning_rate": 0.5217633318946537,
#   "max_depth": 7,
#   "min_child_weight": 10,
#   "n_estimators": 13,
#   "n_jobs": 100,
#   "subsample": 0.7210922079631862,
#   "verbosity": 0,
#   "random_state": 235839369,
# }


space = {
      'max_depth' : hp.quniform('max_depth', 10, 30, 1)                                  # depth of trees (preference is for shallow trees or even stumps (max_depth=1))
      ,'learning_rate' : hp.loguniform('learning_rate', np.log(0.01), np.log(0.10))     # learning rate for XGBoost
      ,'gamma': hp.quniform('gamma', 0.0, 1.0, 0.001)                                   # minimum loss reduction required to make a further partition on a leaf node
      ,'min_child_weight' : hp.quniform('min_child_weight', 4, 25, 1)                   # minimum number of instances per node
      ,'subsample' : hp.loguniform('subsample', np.log(0.1), np.log(1.0))               # random selection of rows for training,
      ,'colsample_bytree' : hp.loguniform('colsample_bytree', np.log(0.1), np.log(1.0)) # proportion of columns to use per tree
      ,'colsample_bylevel': hp.loguniform('colsample_bylevel', np.log(0.1), np.log(1.0))# proportion of columns to use per level
      ,'colsample_bynode' : hp.loguniform('colsample_bynode', np.log(0.1), np.log(1.0)) # proportion of columns to use per node
      ,'scale_pos_weight' : hp.loguniform('scale_pos_weight', np.log(1), np.log(scale * 10))   # weight to assign positive label to manage imbalance
  }

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run trials
# MAGIC When widening the search space and training multiple models, switch to `SparkTrials` to parallelize
# MAGIC training on Spark:
# MAGIC ```
# MAGIC from hyperopt import SparkTrials
# MAGIC trials = SparkTrials()
# MAGIC ```
# MAGIC 
# MAGIC NOTE: While `Trials` starts an MLFlow run for each set of hyperparameters, `SparkTrials` only starts
# MAGIC one top-level run; it will start a subrun for each set of hyperparameters.
# MAGIC 
# MAGIC See http://hyperopt.github.io/hyperopt/scaleout/spark/ for more info.

# COMMAND ----------

from hyperopt import SparkTrials
trials = SparkTrials()
#trials = Trials()

fmin(objective,
     space=space,
     algo=tpe.suggest,
     max_evals=10,  # Increase this when widening the hyperparameter search space.
     trials=trials)


# COMMAND ----------


best_result = trials.best_trial["result"]
model = best_result["model"]
mlflow_run = best_result["run"]

display(
  pd.DataFrame(
    [best_result["val_metrics"], best_result["test_metrics"]],
    index=["validation", "test"]))

set_config(display="diagram")
model

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature importance
# MAGIC 
# MAGIC SHAP is a game-theoretic approach to explain machine learning models, providing a summary plot
# MAGIC of the relationship between features and model output. Features are ranked in descending order of
# MAGIC importance, and impact/color describe the correlation between the feature and the target variable.
# MAGIC - Generating SHAP feature importance is a very memory intensive operation, so to ensure that AutoML can run trials without
# MAGIC   running out of memory, we disable SHAP by default.<br />
# MAGIC   You can set the flag defined below to `shap_enabled = True` and re-run this notebook to see the SHAP plots.
# MAGIC - To reduce the computational overhead of each trial, a single example is sampled from the validation set to explain.<br />
# MAGIC   For more thorough results, increase the sample size of explanations, or provide your own examples to explain.
# MAGIC - SHAP cannot explain models using data with nulls; if your dataset has any, both the background data and
# MAGIC   examples to explain will be imputed using the mode (most frequent values). This affects the computed
# MAGIC   SHAP values, as the imputed samples may not match the actual data distribution.
# MAGIC 
# MAGIC For more information on how to read Shapley values, see the [SHAP documentation](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html).
# MAGIC 
# MAGIC > **NOTE:** SHAP run may take a long time with the datetime columns in the dataset.

# COMMAND ----------

# Set this flag to True and re-run the notebook to see the SHAP plots
shap_enabled = None

if shap_enabled:
    from shap import KernelExplainer, summary_plot
    # SHAP cannot explain models using data with nulls.
    # To enable SHAP to succeed, both the background data and examples to explain are imputed with the mode (most frequent values).
    mode = X_train.mode().iloc[0]

    # Sample background data for SHAP Explainer. Increase the sample size to reduce variance.
    train_sample = X_train.sample(n=min(1000, X_train.shape[0]), random_state=235839369).fillna(mode)

    # Sample some rows from the validation set to explain. Increase the sample size for more thorough results.
    example = X_val.sample(n=min(1000, X_val.shape[0]), random_state=235839369).fillna(mode)

    # Use Kernel SHAP to explain feature importance on the sampled rows from the validation set.
    predict = lambda x: model.predict(pd.DataFrame(x, columns=X_train.columns))
    explainer = KernelExplainer(predict, train_sample, link="identity")
    shap_values = explainer.shap_values(example, l1_reg=False)
    summary_plot(shap_values, example, class_names=model.classes_)

# COMMAND ----------



# COMMAND ----------

# let's try to distribute Shap
# 



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference
# MAGIC [The MLflow Model Registry](https://docs.databricks.com/applications/mlflow/model-registry.html) is a collaborative hub where teams can share ML models, work together from experimentation to online testing and production, integrate with approval and governance workflows, and monitor ML deployments and their performance. The snippets below show how to add the model trained in this notebook to the model registry and to retrieve it later for inference.
# MAGIC 
# MAGIC > **NOTE:** The `model_uri` for the model already trained in this notebook can be found in the cell below
# MAGIC 
# MAGIC ### Register to Model Registry
# MAGIC ```
# MAGIC model_name = "Example"
# MAGIC 
# MAGIC model_uri = f"runs:/{ mlflow_run.info.run_id }/model"
# MAGIC registered_model_version = mlflow.register_model(model_uri, model_name)
# MAGIC ```
# MAGIC 
# MAGIC ### Load from Model Registry
# MAGIC ```
# MAGIC model_name = "Example"
# MAGIC model_version = registered_model_version.version
# MAGIC 
# MAGIC model_uri=f"models:/{model_name}/{model_version}"
# MAGIC model = mlflow.pyfunc.load_model(model_uri=model_uri)
# MAGIC model.predict(input_X)
# MAGIC ```
# MAGIC 
# MAGIC ### Load model without registering
# MAGIC ```
# MAGIC model_uri = f"runs:/{ mlflow_run.info.run_id }/model"
# MAGIC 
# MAGIC model = mlflow.pyfunc.load_model(model_uri=model_uri)
# MAGIC model.predict(input_X)
# MAGIC ```

# COMMAND ----------

# model_uri for the generated model
print(f"runs:/{ mlflow_run.info.run_id }/model")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Confusion matrix, ROC and Precision-Recall curves for validation data
# MAGIC 
# MAGIC We show the confusion matrix, ROC and Precision-Recall curves of the model on the validation data.
# MAGIC 
# MAGIC For the plots evaluated on the training and the test data, check the artifacts on the MLflow run page.

# COMMAND ----------

# Paste the entire output (%md ...) to an empty cell, and click the link to see the MLflow run page
print(f"%md [Link to model run page](#mlflow/experiments/4156927442295141/runs/{ mlflow_run.info.run_id }/artifactPath/model)")

# COMMAND ----------

import uuid
from IPython.display import Image

# Create temp directory to download MLflow model artifact
eval_temp_dir = os.path.join(os.environ["SPARK_LOCAL_DIRS"], "tmp", str(uuid.uuid4())[:8])
os.makedirs(eval_temp_dir, exist_ok=True)

# Download the artifact
eval_path = mlflow.artifacts.download_artifacts(run_id=mlflow_run.info.run_id, dst_path=eval_temp_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Confusion matrix for validation dataset

# COMMAND ----------

eval_confusion_matrix_path = os.path.join(eval_path, "confusion_matrix.png")
display(Image(filename=eval_confusion_matrix_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ### ROC curve for validation dataset

# COMMAND ----------

eval_roc_curve_path = os.path.join(eval_path, "roc_curve_plot.png")
display(Image(filename=eval_roc_curve_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Precision-Recall curve for validation dataset

# COMMAND ----------

eval_pr_curve_path = os.path.join(eval_path, "precision_recall_curve_plot.png")
display(Image(filename=eval_pr_curve_path))

# COMMAND ----------


