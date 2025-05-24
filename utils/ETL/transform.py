import pandas as pd
import re
import logging
from datetime import datetime
from typing import Tuple, Dict, List, Optional

# Configure logging
import os
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../../logs/data_transformation.log'),
        logging.StreamHandler()
    ]
)

# Define all keyword lists and synonym mappings
INGREDIENTS_KEYWORDS = [
    'niacinamide', 'asam salisilat', 'salicylic acid', 'retinol', 'vitamin c',
    'asam hialuronat', 'hyaluronic acid', 'ceramide', 'asam azelaic', 'azelaic acid',
    'aha', 'bha', 'zinc', 'teh hijau', 'green tea', 'peptida', 'peptides',
    'gliserin', 'glycerin', 'squalane', 'panthenol', 'allantoin'
]

SKIN_CONCERN_KEYWORDS = [
    'jerawat', 'acne', 'flek hitam', 'dark spots', 'hiperpigmentasi', 'hyperpigmentation',
    'kerutan', 'wrinkles', 'kemerahan', 'redness', 'kulit berminyak', 'oiliness',
    'kulit kering', 'dryness', 'pori-pori', 'pores', 'tekstur tidak merata',
    'uneven texture', 'garis halus', 'fine lines'
]

SKIN_TYPE_KEYWORDS = [
    'kulit berminyak', 'oily skin',
    'kulit kering', 'dry skin',
    'kulit kombinasi', 'combination skin',
    'kulit normal', 'normal skin',
    'kulit sensitif', 'sensitive skin'
]

SKIN_GOAL_KEYWORDS = [
    'kulit cerah', 'mencerahkan', 'brighten', 'bright skin',
    'kulit glowing', 'bercahaya', 'glowing skin', 'radiant',
    'kulit kenyal', 'mengenyalkan kulit', 'plump skin', 'plumping',
    'kulit lembab', 'kulit lembap', 'melembapkan', 'melembabkan','menghidrasi', 'moisturize', 'hydrate',
    'mengencangkan kulit', 'firming skin',
    'menghaluskan kulit', 'kulit halus', 'smooth skin',
    'meratakan warna kulit', 'warna kulit merata', 'even skin tone',
    'anti-penuaan', 'mencegah penuaan', 'anti-aging', 'prevent aging',
    'memperbaiki skin barrier', 'memperkuat skin barrier', 'repair skin barrier', 'strengthen skin barrier',
    'mengontrol minyak', 'mengkontrol minyak', 'mengurangi minyak', 'oil control', 'reduce oil',
    'mengecilkan pori', 'pori-pori tampak kecil', 'minimize pores', 'refined pores',
    'menyamarkan bekas jerawat', 'memudarkan bekas jerawat', 'fade acne scars', 'fade acne marks',
    'kulit segar', 'menyegarkan kulit', 'fresh skin', 'refreshing skin',
    'mempercepat regenerasi kulit', 'membantu regenerasi','regenerasi kulit', 'skin regeneration',
    'kulit sehat', 'healthy skin',
    'menutrisi kulit', 'nourish skin'
]

# (All synonym mappings remain the same as before)

def log_dataframe_stats(df: pd.DataFrame, name: str, logger: logging.Logger) -> None:
    """Log basic statistics about a DataFrame."""
    logger.info(f"{name} DataFrame Stats:")
    logger.info(f"- Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"- Missing Values:")
    for col, count in df.isnull().sum().items():
        if count > 0:
            logger.info(f"  - {col}: {count} ({count/df.shape[0]:.1%})")
    logger.info(f"- Duplicates: {df.duplicated().sum()}")

def merge_datasets(products_list: pd.DataFrame, 
                  products_details: pd.DataFrame, 
                  reviews: pd.DataFrame,
                  logger: logging.Logger) -> pd.DataFrame:
    """Merge the three datasets into one combined DataFrame."""
    logger.info("Starting dataset merging...")
    
    # STEP 1 - Merge product list & details
    logger.info("Merging products_list and products_details...")
    df_products = products_list.merge(
        products_details, 
        on='url', 
        how='left', 
        suffixes=('_list', '_details')
    )
    log_dataframe_stats(df_products, "Merged Products", logger)

    # STEP 2 - Rename columns
    logger.info("Renaming columns...")
    df_products = df_products.rename(columns={
        'name_list': 'product_name',
        'brand_list': 'brand',
        'rating_list': 'rating',
        'description': 'description',
        'price': 'price',
        'category': 'category',
        'total_reviews': 'total_reviews'
    })

    # STEP 3 - Select needed columns
    logger.info("Selecting relevant columns...")
    df_products = df_products[[
        'product_name', 'brand', 'category', 'url',
        'rating', 'total_reviews', 'price', 'description'
    ]]
    log_dataframe_stats(df_products, "Final Products", logger)

    # Merge with reviews
    logger.info("Merging with reviews data...")
    df_combined = df_products.merge(
        reviews, 
        left_on='url', 
        right_on='product_url', 
        how='left', 
        suffixes=('_products', '_reviews')
    )
    
    # Rename columns
    df_combined = df_combined.rename(columns={
        'product_name_products': 'product_name'})

    # Select needed columns
    df_combined_used = df_combined[[
        'url','product_name', 'brand', 'category', 'price', 'rating', 'total_reviews',
        'description', 'review', 'skin_type', 'age', 'rating_star'
    ]]
    
    log_dataframe_stats(df_combined_used, "Final Combined Data", logger)
    logger.info("Dataset merging completed successfully")
    
    return df_combined_used

def extract_keywords(text: str, keywords: List[str]) -> Optional[str]:
    """Extract keywords from text using the provided keyword list."""
    if pd.isna(text):
        return None
    found = set()
    for keyword in keywords:
        if re.search(r'\b' + re.escape(keyword) + r'\b', text.lower()):
            found.add(keyword)
    return ', '.join(found) if found else None

def standardize_keywords(found_text: Optional[str], synonym_map: Dict[str, str]) -> Optional[str]:
    """Standardize keywords using the provided synonym mapping."""
    if pd.isna(found_text):
        return None
    keywords = [kw.strip() for kw in found_text.split(',')]
    standardized = set()
    for kw in keywords:
        if kw in synonym_map:
            standardized.add(synonym_map[kw])
        else:
            standardized.add(kw)
    return ', '.join(sorted(standardized)) if standardized else None

def transform_data(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Transform the raw data by extracting and standardizing features."""
    logger.info("Starting data transformation...")
    
    # Combine review and description text
    logger.info("Combining review and description text...")
    df['text_combined'] = (df['review'].fillna('') + ' ' + df['description'].fillna('')).str.lower()
    
    # Extract features with progress reporting
    logger.info("Extracting features from text...")
    extraction_start = datetime.now()
    
    features = [
        ('found_ingredients', INGREDIENTS_KEYWORDS),
        ('found_skin_concern', SKIN_CONCERN_KEYWORDS),
        ('found_skin_type', SKIN_TYPE_KEYWORDS),
        ('found_skin_goal', SKIN_GOAL_KEYWORDS)
    ]
    
    for col, keywords in features:
        start_time = datetime.now()
        logger.info(f"Extracting {col}...")
        df[col] = df['text_combined'].apply(lambda x: extract_keywords(x, keywords))
        duration = (datetime.now() - start_time).total_seconds()
        
        # Count non-null values
        non_null = df[col].notnull().sum()
        logger.info(f"Extracted {col} - {non_null} matches found ({duration:.2f}s)")
    
    extraction_time = (datetime.now() - extraction_start).total_seconds()
    logger.info(f"Feature extraction completed in {extraction_time:.2f}s")
    
    # Standardize features with progress reporting
    logger.info("Standardizing extracted features...")
    standardization_start = datetime.now()
    
    standardizations = [
        ('std_skin_type', 'found_skin_type', SKIN_TYPE_SYNONYMS),
        ('std_skin_concern', 'found_skin_concern', SKIN_CONCERN_SYNONYMS),
        ('std_ingredients', 'found_ingredients', INGREDIENT_SYNONYMS),
        ('std_skin_goal', 'found_skin_goal', SKIN_GOAL_SYNONYMS)
    ]
    
    for new_col, src_col, mapping in standardizations:
        start_time = datetime.now()
        logger.info(f"Standardizing {src_col} -> {new_col}...")
        df[new_col] = df[src_col].apply(lambda x: standardize_keywords(x, mapping))
        duration = (datetime.now() - start_time).total_seconds()
        
        # Count non-null values
        non_null = df[new_col].notnull().sum()
        logger.info(f"Standardized {new_col} - {non_null} values ({duration:.2f}s)")
    
    standardization_time = (datetime.now() - standardization_start).total_seconds()
    logger.info(f"Feature standardization completed in {standardization_time:.2f}s")
    
    log_dataframe_stats(df, "Final Transformed Data", logger)
    logger.info("Data transformation completed successfully")
    
    return df

def save_transformed_data(df: pd.DataFrame, path: str, logger: logging.Logger) -> None:
    """Save the transformed data to a CSV file."""
    try:
        logger.info(f"Saving transformed data to {path}...")
        start_time = datetime.now()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        df.to_csv(path, index=False)
        
        save_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Data successfully saved to {path} ({save_time:.2f}s)")
    except Exception as e:
        logger.error(f"Failed to save data: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    
    try:
        print("\n" + "="*60)
        print("DATA TRANSFORMATION PROCESS STARTING".center(60))
        print("="*60 + "\n")
        
        # Load data
        from load import load_data
        logger.info("Loading source data...")
        products_list, products_details, reviews = load_data()
        
        # Log initial stats
        print("\nInitial Data Statistics:")
        print("-"*60)
        print(f"{'Products List:':<25} {products_list.shape[0]:>8} rows, {products_list.shape[1]:>3} columns")
        print(f"{'Products Details:':<25} {products_details.shape[0]:>8} rows, {products_details.shape[1]:>3} columns")
        print(f"{'Reviews:':<25} {reviews.shape[0]:>8} rows, {reviews.shape[1]:>3} columns")
        print("-"*60 + "\n")
        
        # Transform data
        logger.info("Beginning data transformation pipeline...")
        start_time = datetime.now()
        
        df_combined = merge_datasets(products_list, products_details, reviews, logger)
        df_transformed = transform_data(df_combined, logger)
        
        # Save results
        save_transformed_data(df_combined, '../../data/products_used_features.csv', logger)
        save_transformed_data(df_transformed, '../../data/products_extracted_features.csv', logger)
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Final summary
        print("\n" + "="*60)
        print("TRANSFORMATION SUMMARY".center(60))
        print("="*60)
        print(f"{'Total processing time:':<30} {total_time:.2f} seconds")
        print(f"{'Final dataset size:':<30} {df_transformed.shape[0]} rows, {df_transformed.shape[1]} columns")
        print("\nExtracted Features Summary:")
        print("-"*60)
        for col in ['std_skin_type', 'std_skin_concern', 'std_ingredients', 'std_skin_goal']:
            non_null = df_transformed[col].notnull().sum()
            print(f"{col+':':<20} {non_null:>6} values ({non_null/df_transformed.shape[0]:.1%})")
        print("="*60 + "\n")
        print("Data transformation completed successfully!\n")
        
    except Exception as e:
        logger.error(f"Transformation failed: {str(e)}", exc_info=True)
        print(f"\nERROR: Transformation failed - {str(e)}\n")
        exit(1)