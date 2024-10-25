import pandas as pd
from pyspark.sql import functions as F
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataProcessor:
    def __init__(self, sparksession, table, config):
        self.df = self.load_data(sparksession, table)
        self.config = config
        self.preprocessor = None

    def load_data(self, sparksession, table: str) -> pd.DataFrame:
        """Load data from Spark table into pandas DataFrame.

        Args:
            sparksession: Active Spark session
            table: Name of the table to load

        Returns:
            pd.DataFrame: Loaded data

        Raises:
            ValueError: If sparksession is not active or table doesn't exist
        """
        if not sparksession:
            raise ValueError("Invalid or inactive Spark session")
        try:
            return sparksession.read.table(table).toPandas()
        except Exception as e:
            raise ValueError(f"Failed to load table {table}: {str(e)}")

    def preprocess_data(self):
        # Remove rows with missing target
        target = self.config.target
        self.df = self.df.dropna(subset=[target])

        # Separate features and target
        self.X = self.df[self.config.num_features + self.config.cat_features]
        self.y = self.df[target]

        # Create preprocessing steps for numeric and categorical data
        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.config.num_features),
                ("cat", categorical_transformer, self.config.cat_features),
            ]
        )

    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> tuple:
        """Split data into training and testing sets.

        Args:
            test_size: Proportion of dataset to include in the test split
            random_state: Random seed for reproducibility

        Returns:
            tuple: (train_set, test_set)

        Raises:
            ValueError: If X or y is None or empty
        """

        train_set, test_set = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return train_set, test_set

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame, sparksession):
        """Save the train and test sets into Databricks tables."""

        train_set_with_timestamp = sparksession.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", F.to_utc_timestamp(F.current_timestamp(), "UTC")
        )

        test_set_with_timestamp = sparksession.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", F.to_utc_timestamp(F.current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set_ma"
        )

        test_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set_ma"
        )

        sparksession.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set_ma SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )
        print("juhu")
        sparksession.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set_ma SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )
