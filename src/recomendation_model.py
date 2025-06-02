import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import logging
import os
from datetime import datetime

# Setting up logging configuration
def setup_logging():
# make sure the logs directory/recometation_model.log exists
    if not os.path.exists('../logs'):
        os.makedirs('../logs')
    if not os.path.exists('../logs/recometation_model.log'):
        with open('../logs/recometation_model.log', 'w') as f:
            pass

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('../logs/recometation_model.log'),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging setup completed. Starting skincare recommendation system.")

# Loading and preprocessing the data from CSV
def load_and_preprocess_data(file_path):
    logging.info(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path)
        logging.info("Data loaded successfully")
        
        # Cleaning price column
        logging.debug("Cleaning price column")
        df['price'] = df['price'].str.replace('Rp. ', '', regex=False).str.replace('.', '', regex=False).astype(float)
        logging.info("Price column cleaned and converted to float")
        
        # Creating new column: rating x total_reviews
        logging.debug("Creating weighted_score column")
        df['weighted_score'] = df['rating'] * df['total_reviews']
        logging.info("Weighted score column created")
        
        # Handling missing values
        logging.debug("Handling missing values")
        df['skin_type'] = df['skin_type'].fillna('Unknown')
        df['age'] = df['age'].fillna('Unknown')
        df['category'] = df['category'].fillna('Unknown')
        logging.info("Missing values handled")
        
        # Normalizing numerical features
        logging.debug("Normalizing price and weighted_score")
        scaler = MinMaxScaler()
        df[['price', 'weighted_score']] = scaler.fit_transform(df[['price', 'weighted_score']])
        logging.info("Numerical features normalized")
        
        return df
    except Exception as e:
        logging.error(f"Error in data preprocessing: {str(e)}")
        raise

# Building the recommendation system
def recommend_skincare(df, user_skin_type, user_age, user_concerns, user_goals, max_price, preferred_category, preferred_ingredients):
    logging.info("Starting recommendation process")
    logging.debug(f"User preferences: skin_type={user_skin_type}, age={user_age}, concerns={user_concerns}, "
                  f"goals={user_goals}, max_price={max_price}, category={preferred_category}, "
                  f"ingredients={preferred_ingredients}")
    
    filtered_df = df.copy()
    
    # Applying filters
    try:
        if user_skin_type and user_skin_type != 'Unknown':
            filtered_df = filtered_df[filtered_df['skin_type'].str.contains(user_skin_type, case=False, na=False)]
            logging.info(f"Filtered by skin type: {user_skin_type}, {len(filtered_df)} products remain")
        
        if user_age and user_age != 'Unknown':
            filtered_df = filtered_df[filtered_df['age'].str.contains(user_age, case=False, na=False)]
            logging.info(f"Filtered by age: {user_age}, {len(filtered_df)} products remain")
        
        if max_price:
            filtered_df = filtered_df[filtered_df['price'] <= max_price]
            logging.info(f"Filtered by max price: {max_price}, {len(filtered_df)} products remain")
        
        if preferred_category and preferred_category != 'Unknown':
            filtered_df = filtered_df[filtered_df['category'].str.contains(preferred_category, case=False, na=False)]
            logging.info(f"Filtered by category: {preferred_category}, {len(filtered_df)} products remain")
        
        if user_concerns:
            filtered_df = filtered_df[filtered_df['description'].str.contains('|'.join(user_concerns), case=False, na=False) | 
                                     filtered_df['review'].str.contains('|'.join(user_concerns), case=False, na=False)]
            logging.info(f"Filtered by concerns: {user_concerns}, {len(filtered_df)} products remain")
        
        if user_goals:
            filtered_df = filtered_df[filtered_df['description'].str.contains('|'.join(user_goals), case=False, na=False) | 
                                     filtered_df['review'].str.contains('|'.join(user_goals), case=False, na=False)]
            logging.info(f"Filtered by goals: {user_goals}, {len(filtered_df)} products remain")
        
        if preferred_ingredients:
            filtered_df = filtered_df[filtered_df['description'].str.contains('|'.join(preferred_ingredients), case=False, na=False)]
            logging.info(f"Filtered by ingredients: {preferred_ingredients}, {len(filtered_df)} products remain")
        
        if filtered_df.empty:
            logging.warning("No products match the criteria after filtering")
            return pd.DataFrame(), 0
        
        # Creating a simple scoring system
        logging.debug("Calculating recommendation scores")
        filtered_df['recommendation_score'] = (0.6 * filtered_df['weighted_score']) + (0.4 * (1 - filtered_df['price']))
        logging.info("Recommendation scores calculated")
        
        # Sorting by recommendation score
        recommendations = filtered_df.sort_values(by='recommendation_score', ascending=False)
        logging.info("Products sorted by recommendation score")
        
        # Limiting to top 5 recommendations
        top_recommendations = recommendations.head(5)
        logging.info(f"Top 5 recommendations selected, total matches: {len(filtered_df)}")
        
        return top_recommendations, len(filtered_df)
    
    except Exception as e:
        logging.error(f"Error in recommendation process: {str(e)}")
        raise

# Example usage
def main():
    # Setting up logging
    setup_logging()
    
    # Loading data
    file_path = 'data/products_integrated_features.csv'  # Replace with your CSV file path
    try:
        df = load_and_preprocess_data(file_path)
        
        # User preferences (example)
        user_skin_type = 'Dry'
        user_age = '19 - 24'
        user_concerns = ['dry', 'kemerahan']
        user_goals = ['melembapkan', 'halus']
        max_price = 0.5  # Normalized price (adjust based on your data range after normalization)
        preferred_category = 'Cleanser / Toner'
        preferred_ingredients = ['Hyaluronic Acid']
        
        # Getting recommendations
        recommendations, total_matches = recommend_skincare(
            df, user_skin_type, user_age, user_concerns, user_goals, max_price, preferred_category, preferred_ingredients
        )
        
        # Displaying results
        if not recommendations.empty:
            logging.info(f"Found {total_matches} matching products. Displaying top 5 recommendations")
            for idx, row in recommendations.iterrows():
                logging.info(f"Recommendation: Product={row['product_name']}, Brand={row['brand']}, "
                             f"Category={row['category']}, Price=Rp. {row['price']*100000:.0f}, "
                             f"Rating={row['rating']}, Total Reviews={row['total_reviews']}, "
                             f"Weighted Score={row['weighted_score']:.2f}, "
                             f"Recommendation Score={row['recommendation_score']:.2f}, "
                             f"Skin Type={row['skin_type']}")
                print(f"\nProduct: {row['product_name']}")
                print(f"Brand: {row['brand']}")
                print(f"Category: {row['category']}")
                print(f"Price: Rp. {row['price']*100000:.0f}")
                print(f"Rating: {row['rating']}")
                print(f"Total Reviews: {row['total_reviews']}")
                print(f"Weighted Score: {row['weighted_score']:.2f}")
                print(f"Recommendation Score: {row['recommendation_score']:.2f}")
                print(f"Skin Type: {row['skin_type']}")
                print(f"Description: {row['description'][:100]}...")
        else:
            logging.warning("No matching products to display")
            print("No products match your criteria.")
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()