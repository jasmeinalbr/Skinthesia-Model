import os
import pandas as pd
import logging
from datetime import datetime


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../../logs/data_loading.log'),
        logging.StreamHandler()
    ]
)

def load_data():
    """
    Load raw data files from the data directory.
    
    Returns:
        tuple: (products_list, products_details, reviews) DataFrames
        
    Raises:
        FileNotFoundError: If any data file is missing
        Exception: For other loading errors
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting data loading process...")
        
        # Define file paths
        data_dir = '../../data'
        files = {
            'products_list': os.path.join(data_dir, 'products_list.csv'),
            'products_details': os.path.join(data_dir, 'products_detail.csv'),
            'reviews': os.path.join(data_dir, 'reviews.csv')
        }
        
        # Check if files exist
        for name, path in files.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Data file not found: {path}")
        
        logger.info("All data files found. Loading files...")
        
        # Load with progress reporting
        results = {}
        for name, path in files.items():
            start_time = datetime.now()
            logger.info(f"Loading {name} from {path}...")
            
            results[name] = pd.read_csv(path)
            
            load_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Successfully loaded {name} ({results[name].shape[0]} rows, "
                      f"{results[name].shape[1]} cols) in {load_time:.2f}s")
        
        logger.info("All data files loaded successfully")
        return results['products_list'], results['products_details'], results['reviews']
    
    except FileNotFoundError as e:
        logger.error(f"Data file missing: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        print("\n" + "="*50)
        print("DATA LOADING PROCESS STARTING")
        print("="*50 + "\n")
        
        products_list, products_details, reviews = load_data()
        
        print("\nData Summary:")
        print("-"*40)
        print(f"{'Products List:':<20} {products_list.shape[0]:>6} rows, {products_list.shape[1]:>2} columns")
        print(f"{'Products Details:':<20} {products_details.shape[0]:>6} rows, {products_details.shape[1]:>2} columns")
        print(f"{'Reviews:':<20} {reviews.shape[0]:>6} rows, {reviews.shape[1]:>2} columns")
        print("-"*40)
        print("\nData loading completed successfully!\n")
        
    except Exception as e:
        print(f"\nERROR: Data loading failed - {str(e)}\n")
        exit(1)