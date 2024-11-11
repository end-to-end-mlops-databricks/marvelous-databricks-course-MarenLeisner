# Databricks notebook source
from pyspark.sql import SparkSession
from databricks.connect import DatabricksSession
import os

from taxinyc.data_processor import DataProcessor
from taxinyc.config import ProjectConfig

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# path stuff is annoying, in vs code it seems like everything gets executed from repo main level
try:
    config = ProjectConfig.from_yaml(config_path="project_config.yml")
except Exception:
    config = ProjectConfig.from_yaml(config_path="../../project_config.yml")

spark = DatabricksSession.builder.profile(os.environ["DATABRICKS_PROFILE"]).getOrCreate()

try:
    path = (
        config.read_from["catalog_name"] + "." + config.read_from["schema_name"] + "." + config.read_from["table_name"]
    )
    logger.info("path_name is:", path)
except KeyError:
    logger.error("Error building path to data")

# COMMAND ----------
# Preprocess data
data_processor = DataProcessor(config=config)
data_processor.preprocess()
train_set, test_set = data_processor.split_data()
data_processor.save_to_catalog(train_set=train_set, test_set=test_set, spark=spark)
print("DONE")