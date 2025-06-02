import pandas as pd
import numpy as np
import re
import logging
import ast
from datetime import datetime
from typing import Tuple, Dict, List, Optional
from collections import Counter

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

SKIN_CONCERN_SYNONYMS = {
    'jerawat': 'acne',
    'acne': 'acne',
    'flek hitam': 'dark spots',
    'dark spots': 'dark spots',
    'hiperpigmentasi': 'hyperpigmentation',
    'hyperpigmentation': 'hyperpigmentation',
    'kerutan': 'wrinkles',
    'wrinkles': 'wrinkles',
    'kemerahan': 'redness',
    'redness': 'redness',
    'kulit berminyak': 'oiliness',
    'oiliness': 'oiliness',
    'kulit kering': 'dryness',
    'dryness': 'dryness',
    'pori-pori': 'pores',
    'pores': 'pores',
    'tekstur tidak merata': 'uneven texture',
    'uneven texture': 'uneven texture',
    'garis halus': 'fine lines',
    'fine lines': 'fine lines'
}

INGREDIENT_SYNONYMS = {
    'salicylic acid': 'salicylic acid',
    'asam salisilat': 'salicylic acid',
    'hyaluronic acid': 'hyaluronic acid',
    'asam hialuronat': 'hyaluronic acid',
    'azelaic acid': 'azelaic acid',
    'asam azelaic': 'azelaic acid',
    'green tea': 'green tea',
    'teh hijau': 'green tea',
    'peptides': 'peptides',
    'peptida': 'peptides',
    'glycerin': 'glycerin',
    'gliserin': 'glycerin'
}

SKIN_GOAL_SYNONYMS = {
    # brightening
    'kulit cerah': 'brightening',
    'mencerahkan': 'brightening',
    'brighten': 'brightening',
    'bright skin': 'brightening',

    # glowing
    'kulit glowing': 'glowing',
    'bercahaya': 'glowing',
    'glowing skin': 'glowing',
    'radiant': 'glowing',

    # plumping
    'kulit kenyal': 'plumping',
    'mengenyalkan kulit': 'plumping',
    'plump skin': 'plumping',
    'plumping': 'plumping',

    # hydrating
    'kulit lembab':'hydrating',
    'kulit lembap':'hydrating',
    'melembabkan':'hydrating',
    'melembapkan': 'hydrating',
    'menghidrasi': 'hydrating',
    'moisturize': 'hydrating',
    'hydrate': 'hydrating',

    # firming
    'mengencangkan kulit': 'firming',
    'firming skin': 'firming',

    # smoothing
    'menghaluskan kulit': 'smoothing',
    'kulit halus': 'smoothing',
    'smooth skin': 'smoothing',

    # tone-evening
    'meratakan warna kulit': 'tone-evening',
    'warna kulit merata': 'tone-evening',
    'even skin tone': 'tone-evening',

    # anti-aging
    'anti-penuaan': 'anti-aging',
    'mencegah penuaan': 'anti-aging',
    'anti-aging': 'anti-aging',
    'prevent aging': 'anti-aging',

    # barrier-repair
    'memperbaiki skin barrier': 'barrier-repair',
    'memperkuat skin barrier': 'barrier-repair',
    'repair skin barrier': 'barrier-repair',
    'strengthen skin barrier': 'barrier-repair',

    # oil-control
    'mengontrol minyak': 'oil-control',
    'mengkontrol minyak': 'oil-control',
    'mengurangi minyak': 'oil-control',
    'oil control': 'oil-control',
    'reduce oil': 'oil-control',

    # pore-minimizing
    'mengecilkan pori': 'pore-minimizing',
    'pori-pori tampak kecil': 'pore-minimizing',
    'minimize pores': 'pore-minimizing',
    'refined pores': 'pore-minimizing',

    # scar-fading
    'menyamarkan bekas jerawat': 'scar-fading',
    'memudarkan bekas jerawat': 'scar-fading',
    'fade acne scars': 'scar-fading',
    'fade acne marks': 'scar-fading',

    # refreshing
    'kulit segar': 'refreshing',
    'menyegarkan kulit': 'refreshing',
    'fresh skin': 'refreshing',
    'refreshing skin': 'refreshing',

    # regenerating
    'mempercepat regenerasi kulit': 'regenerating',
    'skin regeneration': 'regenerating',
    'membantu regenerasi':'regenerating',
    'regenerasi kulit':'regenerating',

    # healthy
    'kulit sehat': 'healthy',
    'healthy skin': 'healthy',

    # nourishing
    'menutrisi kulit': 'nourishing',
    'nourish skin': 'nourishing'
}

def log_dataframe_stats(df: pd.DataFrame, name: str, logger: logging.Logger) -> None:
    """Log basic statistics about a DataFrame."""
    logger.info(f"{name} DataFrame Stats:")
    logger.info(f"- Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"- Missing Values:")

    for col in df.columns:
        def is_missing(x):
            try:
                if x is None:
                    return True
                if isinstance(x, float) and pd.isna(x):
                    return True
                if isinstance(x, str) and x.strip() == "":
                    return True
                if isinstance(x, (list, np.ndarray)) and len(x) == 0:
                    return True
                return False
            except Exception as e:
                logger.warning(f"  - Skipping value in column '{col}' due to error: {e}")
                return False

        missing_count = df[col].apply(is_missing).sum()

        if missing_count > 0:
            missing_pct = missing_count / df.shape[0]
            logger.info(f"  - {col}: {missing_count} ({missing_pct:.1%})")

    # Safe duplicate check: convert unhashable types (list/array) to tuples
    df_for_duplicates = df.copy()
    for col in df_for_duplicates.columns:
        if df_for_duplicates[col].apply(lambda x: isinstance(x, (list, np.ndarray))).any():
            df_for_duplicates[col] = df_for_duplicates[col].apply(
                lambda x: tuple(x) if isinstance(x, (list, np.ndarray)) else x
            )

    dup_count = df_for_duplicates.duplicated().sum()
    logger.info(f"- Duplicates: {dup_count}")

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
        'description', 'review', 'skin_type', 'age', 'rating_star', 'recommended'
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

def merge_unique(series: pd.Series) -> List[str]:
    """Merge unique items from a Series, handling strings and lists."""
    combined = set()
    for item in series.dropna():
        if isinstance(item, str):
            try:
                parsed = eval(item) if item.startswith('[') else [item]
            except:
                parsed = [item]
            combined.update(parsed)
        elif isinstance(item, list):
            combined.update(item)
    return list(combined)

def get_mode(series):
    try:
        return series.mode().iloc[0]
    except:
        return None

def get_top_2(series):
    try:
        # Gabungkan semua list dalam series jadi satu list datar
        all_values = [item for sublist in series.dropna() for item in sublist]
        counter = Counter(all_values)
        
        if not counter:
            return []

        top_values = counter.most_common(2)
        return [item for item, _ in top_values]
    except Exception:
        return []

def merge_rows(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Merge rows with same url by combining values in the features column using aggregation."""
    logger.info("Merging rows with same URL...")
    agg_df = df.groupby('url').agg({
    'product_name': 'first',
    'brand': 'first',
    'category': 'first',
    'price': 'first',
    'rating': 'first',
    'skin_type': get_top_2,
    'total_reviews': 'sum',
    'std_skin_concern': merge_unique,
    'std_ingredients': merge_unique,
    'std_skin_goal': merge_unique,
    'age': get_mode,
    'rating_star': 'mean'
}).reset_index()
    
    return agg_df

def to_list(value: Optional[str]) -> List[str]:
    """Convert a value to a list, handling strings, NaNs, and lists."""
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
            else:
                return [parsed]
        except:
            return [value]
    return [value]

def flatten_list(nested_lst: List) -> List:
    """Flatten a nested list into a single list."""
    flat = []
    for item in nested_lst:
        if isinstance(item, list):
            flat.extend(flatten_list(item))
        else:
            flat.append(item)
    return flat

def parse_and_clean(cell: Optional[str]) -> List[str]:
    """Parse a cell value and return a cleaned list of unique items."""
    # Kalau tipe string, coba literal_eval ke list Python
    if isinstance(cell, str):
        try:
            lst = ast.literal_eval(cell)
        except:
            # Kalau gagal eval, anggap saja string biasa dan bungkus jadi list
            lst = [cell]
    else:
        lst = cell

    # Flatten nested list (misal list di dalam list)
    flat_list = flatten_list(lst)

    # Split elemen yang masih berupa gabungan string dengan koma
    split_list = []
    for item in flat_list:
        if isinstance(item, str):
            parts = [part.strip() for part in item.split(',')]
            split_list.extend(parts)
        else:
            # Kalau bukan string, langsung masukkan
            split_list.append(str(item).strip())

    # Normalisasi dan hapus duplikat
    seen = set()
    result = []
    for item in split_list:
        item_norm = item.lower()
        if item_norm and item_norm not in seen:
            seen.add(item_norm)
            result.append(item_norm)

    return result

def impute_ingredients_relaxed(row, df: pd.DataFrame, matches_treshold: int = 2) -> List[str]:
    """Impute ingredients using a relaxed matching strategy."""
    if len(row['ingredients']) > 0:
        return row['ingredients']

    def relaxed_match(r):
        matches = 0
        if set(r['skin_concern']) == set(row['skin_concern']):
            matches += 1
        if set(r['skin_goal']) == set(row['skin_goal']):
            matches += 1
        if set(r['skin_type']) == set(row['skin_type']):
            matches += 1
        return matches >= matches_treshold and len(r['ingredients']) > 0

    candidates = df[df.apply(relaxed_match, axis=1)]

    if not candidates.empty:
        return candidates.iloc[0]['ingredients']
    else:
        return ["unknown"]

def impute_skin_type_relaxed(row, df: pd.DataFrame, matches_treshold: int) -> List[str]:
    """Impute skin type using a relaxed matching strategy."""
    if len(row['skin_type']) > 0:
        return row['skin_type']

    def relaxed_match(r):
        matches = 0
        if set(r['skin_concern']) == set(row['skin_concern']):
            matches += 1
        if set(r['skin_goal']) == set(row['skin_goal']):
            matches += 1
        if set(r['ingredients']) == set(row['ingredients']):
            matches += 1
        return matches >= matches_treshold and len(r['skin_type']) > 0

    candidates = df[df.apply(relaxed_match, axis=1)]

    if not candidates.empty:
        return candidates.iloc[0]['skin_type']
    else:
        return []

def integrate_data(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """integrate rows and columns."""
    logger.info("Starting data integration...")
    integration_start = datetime.now()

    df_integrated = merge_rows(df, logger)
    df_integrated = df_integrated.rename(columns={
    'std_skin_concern':'skin_concern',
    'std_skin_goal':'skin_goal',
    'std_ingredients':'ingredients'
    })

    logger.info("Parsing and cleaning integrated columns...")
    df_integrated = df_integrated.copy()
    cols = ['skin_concern', 'skin_goal', 'skin_type', 'ingredients']
    for col in cols:
        df_integrated[col] = df_integrated[col].apply(parse_and_clean)

    logger.info("Imputing missing ingredients using relaxed matching...")
    df_integrated['ingredients'] = df_integrated.apply(lambda row: impute_ingredients_relaxed(row, df_integrated), axis=1)

    logger.info("Imputing missing skin type using relaxed matching...")
    df_integrated['skin_type'] = df_integrated.apply(lambda row: impute_skin_type_relaxed(row, df_integrated, matches_treshold=2), axis=1)
    df_integrated['skin_type'] = df_integrated.apply(lambda row: impute_skin_type_relaxed(row, df_integrated, matches_treshold=1), axis=1)

    integration_time = (datetime.now() - integration_start).total_seconds()
    logger.info(f"Feature integration completed in {integration_time:.2f}s")
    
    log_dataframe_stats(df_integrated, "Final Integrated Data", logger)
    logger.info("Data Integration completed successfully")

    return df_integrated

def transform_data(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Transform the raw data by extracting and standardizing features."""
    logger.info("Starting data transformation...")
    
    # Filter only recommended products
    logger.info("Filtering only recommended products (recommended == 'True')...")
    df = df[df['recommended'] == 'True'].copy()
    logger.info(f"{len(df)} entries remaining after filtering.")

    # Combine review and description text
    logger.info("Combining review and description text...")
    df['text_combined'] = (df['review'].fillna('') + ' ' + df['description'].fillna('')).str.lower()
    
    # Clean and format 'skin_type' column
    logger.info("Cleaning and formatting 'skin_type' column...")
    df['skin_type'] = df['skin_type'].apply(lambda x: [x.lower()] if isinstance(x, str) else [])

    # Extract features with progress reporting
    logger.info("Extracting features from text...")
    extraction_start = datetime.now()
    
    features = [
        ('found_ingredients', INGREDIENTS_KEYWORDS),
        ('found_skin_concern', SKIN_CONCERN_KEYWORDS),
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
    
    df['rating_star'] = pd.to_numeric(df['rating_star'], errors='coerce')

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
        df_integrated = integrate_data(df_transformed, logger)
        
        # Save results
        save_transformed_data(df_combined, '../../data/products_used_features.csv', logger)
        save_transformed_data(df_transformed, '../../data/products_extracted_features.csv', logger)
        save_transformed_data(df_integrated, '../../data/products_integrated_features.csv', logger)

        total_time = (datetime.now() - start_time).total_seconds()
        
        # Final summary
        print("\n" + "="*60)
        print("TRANSFORMATION SUMMARY".center(60))
        print("="*60)
        print(f"{'Total processing time:':<30} {total_time:.2f} seconds")
        print(f"{'Final dataset size:':<30} {df_integrated.shape[0]} rows, {df_integrated.shape[1]} columns")
        print("\nExtracted Features Summary:")
        print("-"*60)
        for col in ['skin_type', 'skin_concern', 'skin_goal', 'ingredients']:
            non_null = df_integrated[col].notnull().sum()
            print(f"{col+':':<20} {non_null:>6} values ({non_null/df_integrated.shape[0]:.1%})")
        print("="*60 + "\n")
        print("Final DataFrame Info:")
        print(f"\n{df_integrated.info()}")
        print("Data transformation completed successfully!\n")
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        print("="*60 + "\n")

    except Exception as e:
        logger.error(f"Transformation failed: {str(e)}", exc_info=True)
        print(f"\nERROR: Transformation failed - {str(e)}\n")
        exit(1)