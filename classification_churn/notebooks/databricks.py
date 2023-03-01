# Databricks notebook source
# MAGIC %md
# MAGIC # MLflow Classification Recipe Databricks Notebook
# MAGIC This notebook runs the MLflow Classification Recipe on Databricks and inspects its results.
# MAGIC 
# MAGIC For more information about the MLflow Classification Recipe, including usage examples,
# MAGIC see the [Classification Recipe overview documentation](https://mlflow.org/docs/latest/recipes.html#classification-recipe)
# MAGIC and the [Classification Recipe API documentation](https://mlflow.org/docs/latest/python_api/mlflow.recipes.html#module-mlflow.recipes.classification.v1.recipe).

# COMMAND ----------

# MAGIC %pip install -r ../../requirements.txt

# COMMAND ----------

# MAGIC %md ### Start with a recipe:

# COMMAND ----------

from mlflow.recipes import Recipe

r = Recipe(profile="databricks") 

# COMMAND ----------

r.clean()

# COMMAND ----------

# MAGIC %md ### Inspect recipe DAG:

# COMMAND ----------

r.inspect()

# COMMAND ----------

# MAGIC %md ### Ingest the dataset:

# COMMAND ----------

r.run("ingest")

# COMMAND ----------

# MAGIC %md ### Split the dataset into train, validation and test:

# COMMAND ----------

r.run("split")

# COMMAND ----------

r.run("transform")

# COMMAND ----------

# MAGIC %md ### Train the model:

# COMMAND ----------

r.run("train")

# COMMAND ----------

# MAGIC %md ### Evaluate the model:

# COMMAND ----------

r.run("evaluate")

# COMMAND ----------

# MAGIC %md ### Register the model:

# COMMAND ----------

r.run("register")

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Getting your artifacts back from all steps

# COMMAND ----------

r.inspect("train")

# COMMAND ----------

training_data = r.get_artifact("training_data")
training_data.describe()

# COMMAND ----------

trained_model = r.get_artifact("model")
print(trained_model)

# COMMAND ----------

r.inspect("transform")

# COMMAND ----------

#r.get_artifact("")

# COMMAND ----------

# MAGIC %md 
# MAGIC ### putting model in prod 

# COMMAND ----------

import mlflow
client = mlflow.tracking.MlflowClient()
model = r.get_artifact("registered_model_version")

# COMMAND ----------

print("registering model version "+model.version+" as production model")
client.transition_model_version_stage(name = model.name, version = model.version, stage = "Production", archive_existing_versions=True)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Inferences at scale using spark and distributed inference
# MAGIC Scaling to distributed / SQL inference can be done within 1 line of code:

# COMMAND ----------


pymodel = mlflow.pyfunc.load_model(f"models:/{model.name}/Production")
model_input_names = pymodel.metadata.signature.inputs.input_names()
ingested_data = r.get_artifact("ingested_data").head()
pymodel.predict(ingested_data)

# COMMAND ----------

display(spark.read.format("delta").load("/demos/dlt/lakehouse_churn/quentin_ambard_demo/tables/churn_features"))

# COMMAND ----------

from pyspark.sql import functions as F

batch_inference_udf = mlflow.pyfunc.spark_udf(spark, f"models:/{model.name}/Production")
df = spark.read.format("delta").load("/demos/dlt/lakehouse_churn/quentin_ambard_demo/tables/churn_features").limit(10)
df.withColumn("predicted_score", batch_inference_udf(*([F.col(f) for f in model_input_names]))).display()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md 
# MAGIC ### Serving 

# COMMAND ----------

import os
os.environ["DATABRICKS_TOKEN"] = "dapide87a503db75d6524cc848fcc57cb47d"

ingested_data = r.get_artifact("ingested_data").head()
# quick fix because of datetime64 not good for JSON
ingested_data[["creation_date","last_activity_date","last_transaction","last_event"]] = ingested_data[["creation_date","last_activity_date","last_transaction","last_event"]].astype(str)

# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd
import json

def create_tf_serving_json(data):
  return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
  url = 'https://e2-demo-field-eng.cloud.databricks.com/model-endpoint/churn_classifier_mlr/3/invocations'
  headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}', 'Content-Type': 'application/json'}
  ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
  data_json = json.dumps(ds_dict, allow_nan=True)
  response = requests.request(method='POST', headers=headers, url=url, data=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()

# COMMAND ----------

scored_df = score_model(ingested_data)
scored_df 

# COMMAND ----------


