# Databricks notebook source
df = spark.read.table("hive_metastore.dbdemos_lakehouse_churn_quentin_ambard_demo.churn_features")
display(df)

# COMMAND ----------

# MAGIC %sh ls ../data/

# COMMAND ----------

# STEP 1: RUN THIS CELL TO INSTALL BAMBOOLIB

# You can also install bamboolib on the cluster. Just talk to your cluster admin for that
%pip install bamboolib  

# Heads up: this will restart your python kernel, so you may need to re-execute some of your other code cells.

# COMMAND ----------

# STEP 2: RUN THIS CELL TO IMPORT AND USE BAMBOOLIB

import bamboolib as bam

# This opens a UI from which you can import your data
bam  

# Already have a pandas data frame? Just display it!
# Here's an example
# import pandas as pd
# df_test = pd.DataFrame(dict(a=[1,2]))
# df_test  # <- You will see a green button above the data set if you display it

# COMMAND ----------

import pandas 
df_pd = pandas.read_csv("../data/churn_sample.csv")

# COMMAND ----------

df_pd

# COMMAND ----------

import pandas 
df_pd = pandas.read_csv("../data/churn_sample.csv")

import pandas as pd; import numpy as np
# Step: Replace missing values
df_pd[['canal']] = df_pd[['canal']].fillna('Unknown')

# Step: Replace missing values
df_pd[['platform']] = df_pd[['platform']].fillna('Unknown')

# Step: Change data type of creation_date to Datetime
df_pd['creation_date'] = pd.to_datetime(df_pd['creation_date'], infer_datetime_format=True)

# Step: Change data type of last_activity_date to Datetime
df_pd['last_activity_date'] = pd.to_datetime(df_pd['last_activity_date'], infer_datetime_format=True)

# Step: Change data type of firstname to String/Text
df_pd['firstname'] = df_pd['firstname'].astype('string')

# Step: Change data type of lastname to String/Text
df_pd['lastname'] = df_pd['lastname'].astype('string')

# Step: Change data type of address to String/Text
df_pd['address'] = df_pd['address'].astype('string')

# Step: Change data type of canal to String/Text
df_pd['canal'] = df_pd['canal'].astype('string')

# Step: Change data type of user_id to String/Text
df_pd['user_id'] = df_pd['user_id'].astype('string')

# Step: Change data type of last_event to Datetime
df_pd['last_event'] = pd.to_datetime(df_pd['last_event'], infer_datetime_format=True)

# Step: Change data type of country to String/Text
df_pd['country'] = df_pd['country'].astype('string')

# Step: Manipulate strings of 'country' via Find 'SPAIN' and Replace with 'SP'
df_pd["country"] = df_pd["country"].str.replace('SPAIN', 'SP', regex=False)

# Step: Change data type of platform to String/Text
df_pd['platform'] = df_pd['platform'].astype('string')

# Step: Extract datetime attribute(s) day of week number from 'last_activity_date'
df_pd['last_activity_date_dayofweek'] = df_pd['last_activity_date'].dt.dayofweek

# Step: Manipulate strings of 'address' and perform a split on ','
split_df = df_pd['address'].str.split(',', expand=True)
split_df.columns = ['address' + f"_{id_}" for id_ in range(len(split_df.columns))]
df_pd = pd.merge(df_pd, split_df, how="left", left_index=True, right_index=True)

# Step: Manipulate strings of 'address_1' and perform a split on ' '
split_df = df_pd['address_1'].str.split('\\ ', expand=True)
split_df.columns = ['address_1' + f"_{id_}" for id_ in range(len(split_df.columns))]
df_pd = pd.merge(df_pd, split_df, how="left", left_index=True, right_index=True)

# Step: Drop columns
df_pd = df_pd.drop(columns=['address_1_4', 'address_1_3'])

# Step: Drop columns
df_pd = df_pd.drop(columns=['address_1_0'])

# Step: Rename column
df_pd = df_pd.rename(columns={'address_1_1': 'state_province'})

# Step: Rename column
df_pd = df_pd.rename(columns={'address_1_2': 'postcode'})

# Step: Drop columns
df_pd = df_pd.drop(columns=['address_0', 'address_1'])

# Step: Databricks: Write to database table
spark.createDataFrame(df_pd).write.mode("overwrite").option("mergeSchema", "true").saveAsTable("ap.churn_features_bam")

# COMMAND ----------

df_pd.head()

# COMMAND ----------

import plotly.express as px
fig = px.histogram(df_pd, y='country', facet_col='gender', color='churn')
fig

# COMMAND ----------

import plotly.express as px
fig = px.histogram(df_pd, y='country', facet_col='canal', color='churn', facet_row='gender')
fig

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id
churn_df = spark.read.table("ap.churn_features_bam")
# .withColumn("id", monotonically_increasing_id()) # create unique id if does not exist to create a FS table
display(churn_df)

# COMMAND ----------

import databricks.feature_store as FS

# COMMAND ----------

fs_client = FS.FeatureStoreClient()

# COMMAND ----------

fs_client.register_table(delta_table="ap.churn_features_bam", 
                            primary_keys="user_id", 
                            description="Churn dataset, prepared with Bamboolib. ", 
                            tags={"version":"v1"}
                        )

# COMMAND ----------

fs_churn_table = fs_client.read_table("ap.churn_features_bam")

# COMMAND ----------

display(fs_churn_table)

# COMMAND ----------


