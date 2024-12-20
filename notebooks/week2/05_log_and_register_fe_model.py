# Databricks notebook source
# MAGIC %pip install ../house_price-0.0.1-py3-none-any.whl
# COMMAND ----------

# dbutils.library.restartPython()

# COMMAND ----------

import logging

import mlflow
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from databricks.sdk import WorkspaceClient
from lightgbm import LGBMRegressor
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from taxinyc.config import ProjectConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Initialize the Databricks session and clients
spark = SparkSession.builder.getOrCreate()
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")


try:
    config = ProjectConfig.from_yaml(config_path="project_config.yml")
except Exception:
    config = ProjectConfig.from_yaml(config_path="../../project_config.yml")


# Extract configuration details
num_features = config.num_features
target = config.target
parameters = config.parameters
catalog_name = config.catalog_name
schema_name = config.schema_name
mlflow_experiment_name = config.mlflow_experiment_name

# Define table names and function name
feature_table_name = f"{catalog_name}.{schema_name}.features_an"
function_name = f"{catalog_name}.{schema_name}.calculate_travel_time_ma"


# COMMAND ----------
# Load training and test sets
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set_ma")
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set_ma")


# COMMAND ----------
# Create or replace the power_features table
spark.sql(f"""
CREATE OR REPLACE TABLE {catalog_name}.{schema_name}.features_ma
(pickup_zip INT NOT NULL,
 trip_distance INT);
""")

spark.sql(
    f"ALTER TABLE {catalog_name}.{schema_name}.features_ma " "ADD CONSTRAINT taxitrip_pk PRIMARY KEY(pickup_zip);"
)
# COMMAND ----------
spark.sql(
    f"ALTER TABLE {catalog_name}.{schema_name}.features_ma" "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
)

# Insert data into the feature table from both train and test sets
spark.sql(
    f"INSERT INTO {catalog_name}.{schema_name}.features_ma " f"SELECT * FROM {catalog_name}.{schema_name}.train_set_ma"
)
spark.sql(
    f"INSERT INTO {catalog_name}.{schema_name}.features_ma" f"SELECT * FROM {catalog_name}.{schema_name}.test_set_ma"
)

# COMMAND ----------
# Define a function to calculate the travel time
spark.sql(f"""
CREATE OR REPLACE FUNCTION {function_name}(tpep_pickup_datetime TIMESTAMP, tpep_dropoff_datetime TIMESTAMP)
RETURNS INT
LANGUAGE PYTHON AS
$$

    time_difference_seconds = (tpep_dropoff_datetime - tpep_pickup_datetime).total_seconds()
    return int(time_difference_seconds / 60)
$$
""")

# COMMAND ----------
# Load training and test sets
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set_ma").drop("trip_distance")
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set_ma").toPandas()

# Cast YearBuilt to int for the function input
# train_set = train_set.withColumn("RoundedTemp", train_set["RoundedTemp"].cast("int"))
# train_set = train_set.withColumn("DateTime", train_set["DateTime"].cast("string"))
# COMMAND ----------

# TODO: This leads to unauthorized error as Maria said

# Feature engineering setup

training_set = fe.create_training_set(
    df=train_set,
    label=target,
    feature_lookups=[
        FeatureLookup(
            table_name=feature_table_name,
            feature_names=["trip_distance"],
            lookup_key="pickup_zip",
        ),
        FeatureFunction(
            udf_name=function_name,
            output_name="travel_time",
            input_bindings={
                "tpep_pickup_datetime": "tpep_pickup_datetime",
                "tpep_dropoff_datetime": "tpep_dropoff_datetime",
            },
        ),
    ],
    exclude_columns=["update_timestamp_utc"],
)
# COMMAND ----------

# Load feature-engineered DataFrame
training_df = training_set.load_df().toPandas()

# Split features and target
X_train = training_df[num_features]
y_train = training_df[target]
X_test = test_set[num_features]
y_test = test_set[target]

# Setup preprocessing and model pipeline
pipeline = Pipeline(steps=[("regressor", LGBMRegressor(**parameters))])

# Set and start MLflow experiment
mlflow.set_experiment(experiment_name=config.mlflow_experiment_name)
git_sha = "bla"

with mlflow.start_run(tags={"branch": "week-2", "git_sha": f"{git_sha}"}) as run:
    run_id = run.info.run_id
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Calculate and print metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R2 Score: {r2}")

    # Log model parameters, metrics, and model
    mlflow.log_param("model_type", "LightGBM with preprocessing")
    mlflow.log_params(parameters)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2_score", r2)
    signature = infer_signature(model_input=X_train, model_output=y_pred)

    # Log model with feature engineering
    fe.log_model(
        model=pipeline,
        flavor=mlflow.sklearn,
        artifact_path="lightgbm-pipeline-model-fe",
        training_set=training_set,
        signature=signature,
    )
mlflow.register_model(
    model_uri=f"runs:/{run_id}/lightgbm-pipeline-model-fe",
    name=f"{catalog_name}.{schema_name}.power_consumptions_model_fe",
)
