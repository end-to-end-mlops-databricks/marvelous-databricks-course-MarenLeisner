from typing import Any, Dict, List

import yaml
from pydantic import BaseModel


class ProjectConfig(BaseModel):
    num_features: List[str]
    cat_features: List[str]
    target: str
    catalog_name: str
    schema_name: str
    read_from_catalog_name: str
    read_from_schema_name: str
    read_from_table_name: str
    parameters: Dict[str, Any]  # Dictionary to hold model-related parameters
    mlflow_experiment_name: str
    id_col: str

    @classmethod
    def from_yaml(cls, config_path: str):
        """Load configuration from a YAML file."""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
