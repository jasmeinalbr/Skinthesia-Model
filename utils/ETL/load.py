import pandas as pd
import mysql.connector
from mysql.connector import Error
import logging
import os
from typing import List
import json
import ast

# Configure logging
log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs"))
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "data_load.log")),
        logging.StreamHandler()
    ],
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def connect_to_mysql(host: str, user: str, password: str, database: str) -> mysql.connector.connection.MySQLConnection:
    """Establish a connection to the MySQL database."""
    logger = logging.getLogger(__name__)
    try:
        connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        logger.info("Successfully connected to MySQL database")
        return connection
    except Error as e:
        logger.error(f"Error connecting to MySQL database: {e}")
        raise

def create_table(connection: mysql.connector.connection.MySQLConnection, table_name: str) -> None:
    """Create a table in the MySQL database to store the integrated data."""
    logger = logging.getLogger(__name__)
    cursor = connection.cursor()

    create_table_query = """
    CREATE TABLE IF NOT EXISTS products_integrated (
        url VARCHAR(255) PRIMARY KEY,
        image TEXT,
        product_name VARCHAR(255),
        brand VARCHAR(100),
        category VARCHAR(100),
        price INT,
        rating FLOAT,
        total_reviews INT,
        skin_type TEXT COMMENT 'JSON array of skin types',
        skin_concern TEXT COMMENT 'JSON array of skin concerns',
        skin_goal TEXT COMMENT 'JSON array of skin goals',
        ingredients TEXT COMMENT 'JSON array of ingredients',
        ingredient_category TEXT COMMENT 'JSON array of ingredient categories',
        age VARCHAR(50),
        rating_star FLOAT
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
    try:
        cursor.execute(create_table_query)
        connection.commit()
        logger.info(f"Table {table_name} created successfully")
    except Error as e:
        logger.error(f"Error creating table {table_name}: {e}")
        raise
    finally:
        cursor.close()

def safe_parse_list(value) -> List:
    """Safely parse a string that represents a list, handling malformed inputs."""
    logger = logging.getLogger(__name__)
    if pd.isna(value) or value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        # Clean up common issues: replace single quotes with double quotes, fix unterminated quotes
        value = value.strip()
        if value.startswith('[') and not value.endswith(']'):
            value = value + ']'
        # Replace malformed single quotes
        value = value.replace("'", '"')
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
            else:
                logger.warning(f"Parsed value is not a list: {value}")
                return [parsed]
        except (ValueError, SyntaxError) as e:
            logger.warning(f"Failed to parse '{value}' as list: {e}")
            # Fallback: split by comma and clean up
            items = [item.strip().strip('"\'') for item in value.strip('[]').split(',') if item.strip()]
            return items if items else []
    logger.warning(f"Unexpected value type {type(value)}: {value}")
    return [str(value)]

def list_to_string(lst: List) -> str:
    """Convert a list to a JSON string for storage in MySQL."""
    if isinstance(lst, list):
        return json.dumps(lst)
    return json.dumps([])

def load_data_to_mysql(df: pd.DataFrame, connection: mysql.connector.connection.MySQLConnection, 
                      table_name: str, logger: logging.Logger) -> None:
    """Load the DataFrame into the MySQL table."""
    cursor = connection.cursor()
    
    # Prepare the insert query
    insert_query = """
    INSERT INTO products_integrated (
        url, image, product_name, brand, category, price, rating, total_reviews,
        skin_type, skin_concern, skin_goal, ingredients, ingredient_category, age, rating_star
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    logger.info(f"Loading {len(df)} rows into {table_name}...")
    
    try:
        for _, row in df.iterrows():
            # Convert list-type columns to JSON strings
            skin_type = list_to_string(row['skin_type'])
            skin_concern = list_to_string(row['skin_concern'])
            skin_goal = list_to_string(row['skin_goal'])
            ingredients = list_to_string(row['ingredients'])
            ingredient_category = list_to_string(row['ingredient_category'])
            
            # Prepare the data tuple for insertion
            data = (
                row['url'],
                row['image'],
                row['product_name'],
                row['brand'],
                row['category'],
                int(row['price']) if pd.notna(row['price']) else None,
                float(row['rating']) if pd.notna(row['rating']) else None,
                int(row['total_reviews']) if pd.notna(row['total_reviews']) else None,
                skin_type,
                skin_concern,
                skin_goal,
                ingredients,
                ingredient_category,
                row['age'],
                float(row['rating_star']) if pd.notna(row['rating_star']) else None
            )
            
            cursor.execute(insert_query, data)
        
        connection.commit()
        logger.info(f"Successfully loaded {len(df)} rows into {table_name}")
    except Error as e:
        logger.error(f"Error loading data into {table_name}: {e}")
        raise
    finally:
        cursor.close()

def main():
    logger = logging.getLogger(__name__)
    
    try:
        # MySQL database configuration
        db_config = {
            'host': 'localhost',
            'user': 'root',
            'password': '',  # Update with your MySQL password
            'database': 'skinthesia'  # Update with your actual database name
        }
        
        # Path to the integrated CSV file
        csv_path = '../../data/products_integrated_features.csv'
        
        # Table name in MySQL
        table_name = 'products_integrated'
        
        # Load the CSV file
        logger.info(f"Loading CSV file from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Convert string representations of lists to actual lists
        list_columns = ['skin_type', 'skin_concern', 'skin_goal', 'ingredients', 'ingredient_category']
        for col in list_columns:
            df[col] = df[col].apply(safe_parse_list)
        
        # Log DataFrame stats
        logger.info(f"CSV DataFrame Stats: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Connect to MySQL
        connection = connect_to_mysql(**db_config)
        
        # Create table
        create_table(connection, table_name)
        
        # Load data into MySQL
        load_data_to_mysql(df, connection, table_name, logger)
        
        logger.info("Data loading process completed successfully")
        
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}", exc_info=True)
        raise
    finally:
        if 'connection' in locals() and connection.is_connected():
            connection.close()
            logger.info("MySQL connection closed")

if __name__ == "__main__":
    main()