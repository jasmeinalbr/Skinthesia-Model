import pandas as pd
import ast
import logging
import os
import sys

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

# ------------------- LOAD & PREPROCESS ------------------- #
def load_and_prepare_data(file_path):
    logging.info(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)

    # Verify required columns exist
    required_cols = ["product_name", "brand", "image", "price", "rating", "total_reviews", "age", "category", "ingredients"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logging.error(f"Missing columns in dataset: {missing_cols}")
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

    # Hitung review_score
    df["review_score"] = df["rating"] * df["total_reviews"]

    # Ambil kolom yang dipakai
    used_cols = ["product_name", "brand", "image", "price", "rating", "total_reviews", "age", "category", "ingredients", "review_score"]
    df_filtered = df[used_cols].copy()

    # Konversi ingredients dari str → list
    df_filtered["ingredients"] = df_filtered["ingredients"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    logging.info(f"Preprocessing selesai. Total data: {len(df_filtered)}")
    logging.info(f"Sample ingredients: {df_filtered['ingredients'].head().tolist()}")
    logging.info(f"Sample review_score: {df_filtered['review_score'].head().tolist()}")

    return df_filtered

# ------------------- RECOMMENDATION FUNCTION ------------------- #
def recommend_products(df, user_age, user_price_min, user_price_max, user_category, user_ingredients, top_k=5):
    logging.info(f"Mulai filtering rekomendasi berdasarkan input user...")
    logging.info(f"Input user: age={user_age}, category={user_category}, price_range=[{user_price_min}, {user_price_max}], ingredients={user_ingredients}")

    logging.info(f"Initial dataset size: {len(df)} products")

    # Verify review_score column exists
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
    def has_matching_ingredient(ings):
        if not user_ingredients:  # If no ingredients specified, pass all products
            return True
        
        # Normalize product ingredients by splitting compound strings
        normalized_ings = []
        for ing in ings:
            split_ings = [i.strip().lower() for i in ing.split(",")]
            normalized_ings.extend(split_ings)
        
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

    # Log details of products that passed the ingredients filter
    if len(df) > 0:
        logging.info("Products passing ingredients filter:")
        for i, row in df.iterrows():
            logging.info(f"Product: {row['product_name']} - Brand: {row['brand']} - Ingredients: {row['ingredients']} - Review Score: {row['review_score']}")

    # Sort by review_score
    df_sorted = df.sort_values(by="review_score", ascending=False)

    logging.info(f"Rekomendasi ditemukan: {len(df_sorted)} produk, ambil top-{top_k}")
    return df_sorted.head(top_k)

# ------------------- MAIN ------------------- #
def main():
    setup_logging()

    # File path
    file_path = "data/products_integrated_features.csv"

    try:
        df_filtered = load_and_prepare_data(file_path)

        # Contoh input user
        user_input = {
            "age": "25 - 29",
            "price_min": 50000,
            "price_max": 200000,
            "category": "Toner",
            "ingredients": ["niacinamide", "aha"]
        }

        top_products = recommend_products(
            df_filtered,
            user_age=user_input["age"],
            user_price_min=user_input["price_min"],
            user_price_max=user_input["price_max"],
            user_category=user_input["category"],
            user_ingredients=user_input["ingredients"],
            top_k=5
        )

        if not top_products.empty:
            logging.info("Top 5 Produk Rekomendasi:")
            for i, row in top_products.iterrows():
                print(f"\n {row['product_name']} - {row['brand']}")
                print(f"   ▪ Image: {row['image']}")
                print(f"   ▪ Harga: Rp. {int(row['price']):,}")
                print(f"   ▪ Rating: {row['rating']:.2f}")
                print(f"   ▪ Total Reviews: {int(row['total_reviews']):,}")
                print(f"   ▪ Ingredients: {', '.join(row['ingredients'])}")
        else:
            logging.warning("Tidak ada produk yang cocok dengan preferensi user.")
            print("Maaf, tidak ada produk yang cocok. Kemungkinan penyebab:")
            print("- Tidak ada toner untuk umur 25-29 dengan niacinamide atau AHA dalam rentang harga tersebut.")
            print("Saran:")
            print("- Coba rentang harga lebih luas (misalnya, 0-500000).")
            print("- Coba bahan lain (misalnya, hanya 'niacinamide').")
            print("- Periksa kategori lain (misalnya, 'Serum').")

    except Exception as e:
        logging.error(f"Gagal menjalankan sistem: {str(e)}")

# ------------------- RUN ------------------- #
if __name__ == "__main__":
    main()