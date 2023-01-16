# Databricks notebook source
df = spark.read.table("hive_metastore.dbdemos_lakehouse_churn_quentin_ambard_demo.churn_features")

# COMMAND ----------

df.display()

# COMMAND ----------

# MAGIC %sh ls ../data/

# COMMAND ----------

import pandas 
df_pd = pandas.read_csv("../data/churn_sample.csv")
df_pd.display()

# COMMAND ----------


