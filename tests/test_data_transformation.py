import sys
import os

# Add parent directory to path so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.components.data_transformation import DataTransformation
from src.logger import logging
from src.exception import CustomException

if __name__ == "__main__":
    try:
        logging.info("==== Starting Data Transformation ====")
        print("Starting data transformation process...")

        transformer = DataTransformation()
        train_path, test_path, preprocessor_path = transformer.initiate_data_transformation()

        logging.info(f"Transformed train saved to: {train_path}")
        logging.info(f"Transformed test saved to: {test_path}")
        logging.info(f"Preprocessor saved to: {preprocessor_path}")

        print(f"✅ Transformed train saved to: {train_path}")
        print(f"✅ Transformed test saved to: {test_path}")
        print(f"✅ Preprocessor saved to: {preprocessor_path}")
        print("✅ Data transformation completed successfully.")

    except Exception as e:
        logging.error(f"Error during data transformation: {str(e)}")
        raise CustomException(e, sys)

# python tests/test_data_transformation.py