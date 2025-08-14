import os
import sys
import pandas as pd
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.constant import CHURN_DATA_PATH
from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig:
    artifact_folder: str = "artifacts"
    raw_data_file: str = "raw_data.csv"
    train_data_file: str = "train.csv"
    test_data_file: str = "test.csv"


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started")
        try:
            # Create artifacts folder
            os.makedirs(self.config.artifact_folder, exist_ok=True)
            logging.info(f"Created artifacts folder at {self.config.artifact_folder}")

            # Load data
            df = pd.read_csv(CHURN_DATA_PATH)
            # Rename target column to match pipeline expectation
            df.rename(columns={"Exited": "Churn"}, inplace=True)
            logging.info(f"Data loaded from {CHURN_DATA_PATH}, shape: {df.shape}")

            # Save raw data
            raw_data_path = os.path.join(self.config.artifact_folder, self.config.raw_data_file)
            df.to_csv(raw_data_path, index=False)
            logging.info(f"Raw data saved at {raw_data_path}")

            # Train-test split
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

            # Save train data
            train_path = os.path.join(self.config.artifact_folder, self.config.train_data_file)
            train_df.to_csv(train_path, index=False)

            # Save test data
            test_path = os.path.join(self.config.artifact_folder, self.config.test_data_file)
            test_df.to_csv(test_path, index=False)

            logging.info("Data Ingestion completed successfully")
            return train_path, test_path

        except Exception as e:
            logging.error(f"Error during data ingestion: {str(e)}")
            raise CustomException(e, sys)
