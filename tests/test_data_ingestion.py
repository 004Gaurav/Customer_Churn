import sys
import os

# Add parent directory to path so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.components.data_ingestion import DataIngestion
from src.logger import logging
from src.exception import CustomException

if __name__ == "__main__":
    try:
        logging.info("==== Starting Data Ingestion ====")
        data_ingestion = DataIngestion()

        # This method should return paths for train, test, and raw data
        train_path, test_path = data_ingestion.initiate_data_ingestion()

        logging.info(f"Data ingestion completed successfully.\nTrain path: {train_path}\nTest path: {test_path}")
        print("✅ Data ingestion completed successfully.")

    except Exception as e:
        logging.error(f"Error during data ingestion: {str(e)}")
        raise CustomException(e, sys)

# python tests/test_data_ingestion.py
