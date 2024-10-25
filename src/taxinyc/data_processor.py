import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataProcessor:
    def __init__(self, sparksession, table, config):
        self.df = self.load_data(sparksession, table)
        self.config = config
        self.X = None
        self.y = None
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
        if not sparksession or not sparksession.sparkContext:
            raise ValueError("Invalid or inactive Spark session")
        try:
            return sparksession.read.table(table).toPandas()
        except Exception as e:
            raise ValueError(f"Failed to load table {table}: {str(e)}")
    def preprocess_data(self):
        # Remove rows with missing target
        target = self.config['target']
        self.df = self.df.dropna(subset=[target])
        
        # Separate features and target
        self.X = self.df[self.config['num_features'] + self.config['cat_features']]
        self.y = self.df[target]
        
        # Create preprocessing steps for numeric and categorical data
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.config['num_features']),
                ('cat', categorical_transformer, self.config['cat_features'])
            ])

    def split_data(self, test_size: float = 0.2, random_state: int = 42, stratify: bool = False) -> tuple:
        """Split data into training and testing sets.
        
        Args:
            test_size: Proportion of dataset to include in the test split
            random_state: Random seed for reproducibility
            stratify: Whether to preserve target distribution in splits
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
            
        Raises:
            ValueError: If X or y is None or empty
        """
        if self.X is None or self.y is None:
            raise ValueError("Must run preprocess_data before splitting")
        
        stratify_param = self.y if stratify else None
        return train_test_split(
            self.X, self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_param
        )