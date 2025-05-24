import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv('../data/products_extracted_features.csv')

# Data preprocessing
def preprocess_data(df):
    # Clean price column and convert to numeric
    df['price'] = df['price'].str.replace('Rp. ', '').str.replace('.', '').astype(float)
    
    # Fill missing values
    for col in ['std_skin_type', 'std_skin_concern', 'std_ingredients', 'std_skin_goal']:
        df[col] = df[col].fillna('')
    
    # Create a combined features column
    df['combined_features'] = df.apply(lambda row: 
                                      f"{row['category']} {row['std_skin_type']} {row['std_skin_concern']} "
                                      f"{row['std_ingredients']} {row['std_skin_goal']}", axis=1)
    
    return df

data = preprocess_data(data)

# TF-IDF Vectorization for content-based filtering
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['combined_features'])

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create a reverse mapping of indices and product names
indices = pd.Series(data.index, index=data['product_name']).drop_duplicates()

# Recommendation function
def get_recommendations(skin_type, skin_concerns, product_type, price_range, top_n=5):
    try:
        # Create query string
        query = f"{product_type} {skin_type} {skin_concerns}"
        
        # Transform query using TF-IDF
        query_vec = tfidf.transform([query])
        
        # Compute similarity
        sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        # Get top N most similar products
        sim_indices = np.argsort(sim_scores)[-top_n:][::-1]
        
        # Filter by price range
        price_min, price_max = price_range
        filtered_indices = [i for i in sim_indices 
                          if price_min <= data.iloc[i]['price'] <= price_max]
        
        # If not enough products in price range, get more and filter
        if len(filtered_indices) < top_n:
            sim_indices = np.argsort(sim_scores)[-(top_n*2):][::-1]
            filtered_indices = [i for i in sim_indices 
                              if price_min <= data.iloc[i]['price'] <= price_max]
        
        # Get recommended products
        recommendations = data.iloc[filtered_indices[:top_n]].copy()
        
        # Calculate importance scores for ingredients
        recommendations['ingredient_score'] = recommendations['std_ingredients'].apply(
            lambda x: len(x.split(', ')) if x else 0)
        
        # Sort by similarity and ingredient score
        recommendations = recommendations.sort_values(
            by=['ingredient_score'], ascending=False)
        
        return recommendations
    
    except Exception as e:
        print(f"Error in recommendation: {str(e)}")
        return pd.DataFrame()

# Analyze important ingredients for specific concerns
def analyze_ingredients(skin_concerns):
    concern_data = data[data['std_skin_concern'].str.contains(skin_concerns, case=False)]
    
    if concern_data.empty:
        return "No data available for this concern."
    
    # Count ingredient frequency
    ingredients = concern_data['std_ingredients'].str.split(', ').explode()
    ingredient_counts = ingredients.value_counts()
    
    return ingredient_counts.head(10)

# User interface function
def skincare_recommender():
    print("\n=== Skincare Recommendation System ===")
    print("\nPlease provide your skin information:\n")
    
    # Get user input
    skin_type = input("Skin type (combination/oily/dry/normal/sensitive): ").lower()
    concerns = input("Skin concerns (comma separated - acne,redness,etc): ").lower()
    product_type = input("Product type (cleanser, toner, etc): ").lower()
    min_price = float(input("Minimum price (in IDR): "))
    max_price = float(input("Maximum price (in IDR): "))
    
    # Get recommendations
    recs = get_recommendations(
        skin_type=skin_type,
        skin_concerns=concerns,
        product_type=product_type,
        price_range=(min_price, max_price)
    )
    
    # Get important ingredients
    top_ingredients = analyze_ingredients(concerns)
    
    # Display results
    print("\n=== Top Recommended Ingredients ===")
    print(top_ingredients)
    print("\n=== Top 5 Product Recommendations ===")
    if recs.empty:
        print("No products match your criteria. Try broadening your search.")
    else:
        for i, (_, row) in enumerate(recs.iterrows(), 1):
            print(f"\n{i}. {row['product_name']} - {row['brand']}")
            print(f"   Category: {row['category']}")
            print(f"   Price: Rp. {row['price']:,.0f}")
            print(f"   Rating: {row['rating']} ({row['total_reviews']} reviews)")
            print(f"   Suitable for: {row['std_skin_type']}")
            print(f"   Targets: {row['std_skin_concern']}")
            print(f"   Key ingredients: {row['std_ingredients']}")
            print(f"   Benefits: {row['std_skin_goal']}")

# Example usage
if __name__ == "__main__":
    skincare_recommender()