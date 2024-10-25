import yaml
import logging
import os
from databricks.connect import DatabricksSession
from taxinyc.data_processor import DataProcessor

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger(__name__)

try:
    with open('project_config.yml', 'r') as file:
        config = yaml.safe_load(file)
    logger.info("Configuration loaded successfully")
    # Log only non-sensitive keys
    logger.debug("Loaded configuration keys: %s", list(config.keys()))
except FileNotFoundError:
    logger.error("Configuration file 'project_config.yml' not found")
    raise
except yaml.YAMLError as e:
    logger.error("Error parsing configuration file: %s", e)
    raise

spark = DatabricksSession.builder.profile(os.environ['DATABRICKS_PROFILE']).getOrCreate()

try:
    path = config['catalog_name'] + '.'+ config['schema_name'] + '.' + config['table_name']
    logger.info("path_name is:" , path)
except KeyError as e:
    logger.error("Error building path to data")

try:
    # Initialize DataProcessor
    data_processor = DataProcessor(spark, path, config)
    logger.info("DataProcessor initialized.")

    # Preprocess the data
    data_processor.preprocess_data()
    logger.info("Data preprocessed successfully")
except Exception as e:
    logger.error("Error during data processing: %s", str(e))
    raise
