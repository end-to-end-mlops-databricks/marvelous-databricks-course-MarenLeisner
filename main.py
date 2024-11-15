import logging

import yaml
from databricks.connect import DatabricksSession

from taxinyc.config import ProjectConfig
from taxinyc.data_processor import DataProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    config = ProjectConfig.from_yaml(config_path="./project_config.yml")
    logger.info("Configuration loaded successfully")
except FileNotFoundError:
    logger.error("Configuration file 'project_config.yml' not found")
    raise
except yaml.YAMLError as e:
    logger.error("Error parsing configuration file: %s", e)
    raise

spark = DatabricksSession.builder.getOrCreate()

try:
    path = f"{config.read_from["catalog_name"]}.{config.read_from["schema_name"]}.{config.read_from["table_name"]}"
    logger.info("path_name is:", path)
except KeyError:
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

try:
    # split Data
    train_set, test_set = data_processor.split_data()
    logger.info("Data split into Train and Test Sets")
    logger.debug(f"Training set shape: {train_set.shape}, Test set shape: {test_set.shape}")
except Exception as e:
    logger.error("Error during Splitting Test and Train split: %s", str(e))

try:
    # store Data in Db-Tables
    train_set, test_set = data_processor.split_data()
    data_processor.save_to_catalog(train_set=train_set, test_set=test_set, sparksession=spark)
except Exception as e:
    logger.error("Error during stroring train and test data: %s", str(e))
