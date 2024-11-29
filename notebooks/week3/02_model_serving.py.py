# Databricks notebook source
import time
import requests
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    TrafficConfig,
    Route,
)
from databricks.sdk.runtime import dbutils

from taxinyc.config import ProjectConfig
from pyspark.sql import SparkSession

# COMMAND ----------

workspace = WorkspaceClient()
spark = SparkSession.builder.getOrCreate()

try:
    config = ProjectConfig.from_yaml(config_path="project_config.yml")
except Exception:
    config = ProjectConfig.from_yaml(config_path="../../project_config.yml")

catalog_name = config.catalog_name
schema_name = config.schema_name

train_set = spark.table(f"{catalog_name}.{schema_name}.train_set_ma").toPandas()

# COMMAND ----------

workspace.serving_endpoints.create(
    name="taxinyc-model-serving_ma",
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=f"{catalog_name}.{schema_name}.nyctaxi_model_pyfunc_ma",
                scale_to_zero_enabled=True,
                workload_size="Small",
                entity_version=1,
            )
        ],
    # Optional if only 1 entity is served
    # traffic_config=TrafficConfig(
    #     routes=[
    #         Route(served_model_name="power_consumptions_model-2",
    #               traffic_percentage=100)
    #     ]
    #     ),
    ),
)


# COMMAND ----------

host = spark.conf.get("spark.databricks.workspaceUrl")
try:
    # notebook way
    token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
except AttributeError:
    # local/databricks connect way
    token = os.environ.get("DATABRICKS_TOKEN")

# COMMAND ----------

required_columns = [
    "trip_distance"
]

sampled_records = train_set[required_columns].sample(n=1000, replace=True).to_dict(orient="records")
# sampled_records = train_set.sample(n=1000, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

# COMMAND ----------

"""
Each body should be list of json with columns

[{"trip_distance": }]
"""


# COMMAND ----------

start_time = time.time()

model_serving_endpoint = (
    f"https://{host}/serving-endpoints/taxinyc-model-serving/invocations"
)
response = requests.post(
    f"{model_serving_endpoint}",
    headers={"Authorization": f"Bearer {token}"},
    json={"dataframe_records": dataframe_records[0]},
)

end_time = time.time()
execution_time = end_time - start_time

print("Response status:", response.status_code)
print("Reponse text:", response.text)
print("Execution time:", execution_time, "seconds")


# COMMAND ----------

model_serving_endpoint = (
    f"https://{host}/serving-endpoints/taxinyc-model-serving/invocations"
)

headers = {"Authorization": f"Bearer {token}"}
num_requests = 1000


# Function to make a request and record latency
def send_request():
    random_record = random.choice(dataframe_records)
    start_time = time.time()
    response = requests.post(
        model_serving_endpoint,
        headers=headers,
        json={"dataframe_records": random_record},
    )
    end_time = time.time()
    latency = end_time - start_time
    return response.status_code, latency


total_start_time = time.time()
latencies = []

# Send requests concurrently
with ThreadPoolExecutor(max_workers=100) as executor:
    futures = [executor.submit(send_request) for _ in range(num_requests)]

    for future in as_completed(futures):
        status_code, latency = future.result()
        latencies.append(latency)

total_end_time = time.time()
total_execution_time = total_end_time - total_start_time

# Calculate the average latency
average_latency = sum(latencies) / len(latencies)

print("\nTotal execution time:", total_execution_time, "seconds")
print("Average latency per request:", average_latency, "seconds")