import yaml
import logging
from databricks.connect import DatabricksSession
from taxinyc.data_processor import DataProcessor

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger(__name__)

with open('project_config.yml', 'r') as file:
    config = yaml.safe_load(file)

print("Configuration loaded:")
print(yaml.dump(config, default_flow_style=False))

spark = DatabricksSession.builder.profile(config['databricks']['profile_id']).getOrCreate()
path = "samples.nyctaxi.trips"

# Initialize DataProcessor
data_processor = DataProcessor(spark, path, config)
logger.info("DataProcessor initialized.")

# Preprocess the data
data_processor.preprocess_data()
logger.info("Data preprocessed.")
