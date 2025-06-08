import pandas as pd
import ast
import logging
import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import re

# ------------------- SETUP LOGGING ------------------- #
def setup_logging():
    log_dir = "../logs"
    log_file = "recommendation_model.log"
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, log_file)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Logging started for skincare recommendation system")

# ------------------- CATEGORY TO INGREDIENTS MAPPING ------------------- #
CATEGORY_TO_INGREDIENTS = {
    'brightening': ['niacinamide', 'vitamin c', 'arbutin', 'licorice'],
    'acne': ['salicylic acid', 'tea tree'],
    'hydrating': ['hyaluronic acid', 'glycerin'],
    'calming': ['centella asiatica', 'green tea', 'aloe vera', 'ceramide', 'panthenol'],
    'anti_aging': ['vitamin e', 'zinc', 'retinol'],
    'exfoliant': ['aha', 'bha', 'pha', 'lactic acid', 'glycolic acid', 'mandelic acid']
}

def map_category_to_ingredients(categories):
    """Mengubah daftar kategori menjadi daftar bahan spesifik."""
    ingredients = set()
    for category in categories:
        if category in CATEGORY_TO_INGREDIENTS:
            ingredients.update(CATEGORY_TO_INGREDIENTS[category])
        else:
            logging.warning(f"Category '{category}' not found in mapping. Using as-is.")
            ingredients.add(category)
    return list(ingredients)

# ------------------- CLASSIFICATION MODEL INFERENCE ------------------- #
def clean_input(lst):
    if isinstance(lst, str):
        lst = [lst]
    seen = set()
    result = []
    for item in lst:
        item_norm = re.sub(r"[\\[\\]\'\"]", "", item.lower().strip())
        if item_norm and item_norm not in seen:
            seen.add(item_norm)
            result.append(item_norm)
    return result

# Placeholder class weights (GANTI DENGAN NILAI ASLI DARI PELATIHAN)
class_weights_np = np.array([1.0] * 6)  # Update dengan nilai dari "Class weights" di notebook klasifikasi
class_weights_tf = tf.constant(class_weights_np, dtype=tf.float32)

def weighted_binary_crossentropy(y_true, y_pred):
    bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    weight_vector = y_true * class_weights_tf + (1 - y_true) * 1.0
    weighted_bce = bce * weight_vector
    return tf.keras.backend.mean(weighted_bce)

def predict_ingredient_categories(
    skin_type,
    skin_concern,
    skin_goal,
    ingredient,
    threshold=0.3,
    use_class_thresholds=False,
    best_thresholds=None
):
    # Load MLB classes
    try:
        with open("src/models/mlb_classes.json", "r") as f:
            mlb_classes = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load mlb_classes.json: {str(e)}")
        raise

    mlb_skin_type_classes = mlb_classes['skin_type']
    mlb_skin_concern_classes = mlb_classes['skin_concern']
    mlb_skin_goal_classes = mlb_classes['skin_goal']
    mlb_ingredient_classes = mlb_classes['ingredients']
    mlb_ingredient_category_classes = mlb_classes['ingredient_category']

    # Normalize input
    skin_type = clean_input(skin_type)
    skin_concern = clean_input(skin_concern)
    skin_goal = clean_input(skin_goal)
    ingredient = clean_input(ingredient)

    logging.info(f"Normalized inputs: skin_type={skin_type}, skin_concern={skin_concern}, skin_goal={skin_goal}, ingredient={ingredient}")

    # Validate ingredient input
    unrecognized = [i for i in ingredient if i not in mlb_ingredient_classes]
    if unrecognized:
        logging.warning(f"Some ingredients not recognized: {unrecognized}")

    # Create multi-hot vectors for input features
    vec_skin_type = np.array([1 if c in skin_type else 0 for c in mlb_skin_type_classes])
    vec_skin_concern = np.array([1 if c in skin_concern else 0 for c in mlb_skin_concern_classes])
    vec_skin_goal = np.array([1 if c in skin_goal else 0 for c in mlb_skin_goal_classes])
    vec_ingredient = np.array([1 if c in ingredient else 0 for c in mlb_ingredient_classes])

    # Combine features into input vector
    input_vector = np.concatenate([
        vec_skin_type, vec_skin_concern, vec_skin_goal, vec_ingredient
    ]).reshape(1, -1).astype(np.float32)

    # Load model
    try:
        model = load_model(
            "src/models/ingredients_category_classification_model.keras",
            custom_objects={'weighted_binary_crossentropy': weighted_binary_crossentropy}
        )
    except Exception as e:
        logging.error(f"Failed to load classification model: {str(e)}")
        raise

    # Predict
    probs = model.predict(input_vector, verbose=0)[0]
    logging.info(f"Prediction probabilities: {probs}")

    # Thresholding
    if use_class_thresholds and best_thresholds is not None:
        if len(best_thresholds) != len(mlb_ingredient_category_classes):
            logging.error(f"Length of best_thresholds ({len(best_thresholds)}) does not match number of categories ({len(mlb_ingredient_category_classes)})")
            raise ValueError("Invalid best_thresholds length")
        predicted_labels = [
            label for label, prob, thres in zip(mlb_ingredient_category_classes, probs, best_thresholds)
            if prob >= thres
        ]
    else:
        predicted_labels = [
            label for label, prob in zip(mlb_ingredient_category_classes, probs)
            if prob >= threshold
        ]

    logging.info(f"Predicted ingredient categories: {predicted_labels}")

    # Map categories to specific ingredients
    mapped_ingredients = map_category_to_ingredients(predicted_labels)
    logging.info(f"Mapped ingredients: {mapped_ingredients}")

    return mapped_ingredients

# ------------------- LOAD & PREPROCESS ------------------- #
def load_and_prepare_data(file_path):
    logging.info(f"Loading dataset from: {file_path}")
    try:
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
    except Exception as e:
        logging.error(f"Failed to load dataset: {str(e)}")
        raise

    # Verify required columns
    required_cols = ["product_name", "brand", "image", "price", "rating", "total_reviews", "age", "category", "ingredients"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logging.error(f"Missing columns in dataset: {missing_cols}")
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

    # Calculate review_score
    try:
        df["review_score"] = df["rating"] * df["total_reviews"]
    except Exception as e:
        logging.error(f"Error calculating review_score: {str(e)}")
        raise

    # Select relevant columns
    used_cols = ["product_name", "brand", "image", "price", "rating", "total_reviews", "age", "category", "ingredients", "review_score"]
    df_filtered = df[used_cols].copy()

    # Convert ingredients from string to list
    def parse_ingredients(x):
        if isinstance(x, str):
            try:
                return ast.literal_eval(x)
            except:
                return [x]
        return x if isinstance(x, list) else []
    
    df_filtered["ingredients"] = df_filtered["ingredients"].apply(parse_ingredients)

    logging.info(f"Preprocessing selesai. Total data: {len(df_filtered)}")
    logging.info(f"Sample ingredients: {df_filtered['ingredients'].head().tolist()}")
    logging.info(f"Sample review_score: {df_filtered['review_score'].head().tolist()}")
    logging.info(f"Available columns: {df_filtered.columns.tolist()}")
    logging.info(f"Rating data types: {df['rating'].dtype}, Total Reviews data types: {df['total_reviews'].dtype}")

    return df_filtered

# ------------------- RECOMMENDATION FUNCTION ------------------- #
def recommend_products(df, user_age, user_price_min, user_price_max, user_category, user_ingredients, top_k=5):
    logging.info(f"Mulai filtering rekomendasi berdasarkan input user...")
    logging.info(f"Input user: age={user_age}, category={user_category}, price_range=[{user_price_min}, {user_price_max}], ingredients={user_ingredients}")

    logging.info(f"Initial dataset size: {len(df)} products")

    # Verify review_score column
    if "review_score" not in df.columns:
        logging.error("review_score column missing in DataFrame")
        raise ValueError("review_score column missing in DataFrame")

    # Filter by age
    df = df[df["age"] == user_age]
    logging.info(f"After age filter ({user_age}): {len(df)} products")

    # Filter by category
    df = df[df["category"] == user_category]
    logging.info(f"After category filter ({user_category}): {len(df)} products")

    # Filter by price range
    df = df[(df["price"] >= user_price_min) & (df["price"] <= user_price_max)]
    logging.info(f"After price filter ({user_price_min} - {user_price_max}): {len(df)} products")
    
    # Log details of remaining products
    if len(df) > 0:
        logging.info("Details of remaining products:")
        for i, row in df.iterrows():
            logging.info(f"Product: {row['product_name']} - Brand: {row['brand']} - Ingredients: {row['ingredients']} - Review Score: {row['review_score']}")

    # Filter by matching ingredients
    df_before_ingredient_filter = df.copy()  # Simpan untuk fallback
    def has_matching_ingredient(ings):
        if not user_ingredients:  # If no ingredients specified, pass all products
            return True
        
        # Normalize product ingredients by splitting compound strings
        normalized_ings = []
        for ing in ings:
            if isinstance(ing, str):
                split_ings = [i.strip().lower() for i in ing.split(",")]
                normalized_ings.extend(split_ings)
            elif isinstance(ing, list):
                normalized_ings.extend([i.lower() for i in ing])
        
        # Log normalized ingredients for debugging
        logging.debug(f"Normalized ingredients: {normalized_ings}")
        
        for user_ing in user_ingredients:
            # Match exact ingredient (case-insensitive)
            if user_ing.lower() in normalized_ings:
                logging.debug(f"Match found for {user_ing}")
                return True
        return False

    df = df[df["ingredients"].apply(has_matching_ingredient)]
    logging.info(f"After ingredients filter ({user_ingredients}): {len(df)} products")

    # Fallback jika tidak ada produk yang cocok
    if df.empty and not df_before_ingredient_filter.empty:
        logging.warning("No products match the predicted ingredients. Using products from other filters as fallback.")
        df = df_before_ingredient_filter
        user_ingredients = []  # Kosongkan untuk menampilkan semua produk

    # Log details of products that passed the ingredients filter
    if len(df) > 0:
        logging.info("Products passing ingredients filter (or fallback):")
        for i, row in df.iterrows():
            logging.info(f"Product: {row['product_name']} - Brand: {row['brand']} - Ingredients: {row['ingredients']} - Review Score: {row['review_score']}")

    # Sort by review_score
    try:
        df_sorted = df.sort_values(by="review_score", ascending=False)
    except Exception as e:
        logging.error(f"Error sorting by review_score: {str(e)}")
        raise

    logging.info(f"Rekomendasi ditemukan: {len(df_sorted)} produk, ambil top-{top_k}")
    return df_sorted.head(top_k)

# ------------------- MAIN ------------------- #
def main():
    setup_logging()

    # File path
    file_path = "data/products_integrated_features.csv"

    try:
        df_filtered = load_and_prepare_data(file_path)

        # User input for classification and recommendation
        user_input = {
            "skin_type": ["dry"],
            "skin_concern": ["redness", "sensitivity"],
            "skin_goal": ["hydration", "plumpness"],
            "ingredient": ["hyaluronic acid"],
            "age": "25 - 29",
            "price_min": 50000,
            "price_max": 200000,
            "category": "Toner"
        }

        # Best thresholds dari notebook klasifikasi
        best_thresholds = [0.5000000000000001, 0.5000000000000001, 0.5000000000000001, 0.45000000000000007, 0.5000000000000001, 0.45000000000000007]

        # Get predicted ingredient categories from classification model
        user_ingredients = predict_ingredient_categories(
            skin_type=user_input["skin_type"],
            skin_concern=user_input["skin_concern"],
            skin_goal=user_input["skin_goal"],
            ingredient=user_input["ingredient"],
            threshold=0.3,
            use_class_thresholds=True,
            best_thresholds=best_thresholds
        )

        top_products = recommend_products(
            df_filtered,
            user_age=user_input["age"],
            user_price_min=user_input["price_min"],
            user_price_max=user_input["price_max"],
            user_category=user_input["category"],
            user_ingredients=user_ingredients,
            top_k=5
        )

        if not top_products.empty:
            logging.info("Top 5 Produk Rekomendasi:")
            for i, row in top_products.iterrows():
                ingredients = row['ingredients']
                if isinstance(ingredients, list):
                    ingredients_str = ', '.join([str(ing) for ing in ingredients if isinstance(ing, str)])
                else:
                    ingredients_str = str(ingredients)
                print(f"\n {row['product_name']} - {row['brand']}")
                print(f"   ▪ Image: {row['image']}")
                print(f"   ▪ Harga: Rp. {int(row['price']):,}")
                print(f"   ▪ Rating: {row['rating']:.2f}")
                print(f"   ▪ Total Reviews: {int(row['total_reviews']):,}")
                print(f"   ▪ Ingredients: {ingredients_str}")
        else:
            logging.warning("Tidak ada produk yang cocok dengan semua filter.")
            print("Maaf, tidak ada produk yang cocok bahkan setelah fallback. Kemungkinan penyebab:")
            print("- Tidak ada toner untuk umur 25-29 dalam rentang harga tersebut.")
            print("Saran:")
            print("- Coba rentang harga lebih luas (misalnya, 0-500000).")
            print("- Coba skin type, concern, goal, atau ingredient lain.")
            print("- Periksa kategori lain (misalnya, 'Serum').")

    except Exception as e:
        logging.error(f"Gagal menjalankan sistem: {str(e)}")
        print(f"Error: {str(e)}")

# ------------------- RUN ------------------- #
if __name__ == "__main__":
    main()