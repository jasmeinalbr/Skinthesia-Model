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
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Logging started for skincare recommendation system")

# ------------------- LOAD & PREPROCESS ------------------- #
def load_and_prepare_data(file_path):
    logging.info(f"Loading dataset from: {file_path}")
    df = pd.read_excel(file_path)

    # Hitung review_score
    df["review_score"] = df["rating"] * df["total_reviews"]

    # Ambil kolom yang dipakai
    used_cols = ["product_name", "brand", "age", "review_score", "price", "category", "ingredients"]
    df_filtered = df[used_cols].copy()

    # Konversi ingredients dari str → list
    df_filtered["ingredients"] = df_filtered["ingredients"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    logging.info(f"Preprocessing selesai. Total data: {len(df_filtered)}")

    return df_filtered

# ------------------- RECOMMENDATION FUNCTION ------------------- #
def recommend_products(df, user_age, user_price_min, user_price_max, user_category, user_ingredients, top_k=5):
    logging.info(f"Mulai filtering rekomendasi berdasarkan input user...")

    # Filter by age
    df = df[df["age"] == user_age]

    # Filter by category
    df = df[df["category"] == user_category]

    # Filter by price range
    df = df[(df["price"] >= user_price_min) & (df["price"] <= user_price_max)]

    # Filter by matching ingredients
    def has_matching_ingredient(ings):
        return any(ing in ings for ing in user_ingredients)

    df = df[df["ingredients"].apply(has_matching_ingredient)]

    # Sort berdasarkan review_score
    df_sorted = df.sort_values(by="review_score", ascending=False)

    logging.info(f"Rekomendasi ditemukan: {len(df_sorted)} produk, ambil top-{top_k}")
    return df_sorted.head(top_k)

# ------------------- MAIN ------------------- #
def main():
    setup_logging()

    # File path
    file_path = "data/products_integrated_features.xlsx"

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
                print(f"\n🧴 {row['product_name']} - {row['brand']}")
                print(f"   ▪ Kategori: {row['category']}")
                print(f"   ▪ Umur: {row['age']}")
                print(f"   ▪ Harga: Rp. {int(row['price']):,}")
                print(f"   ▪ Skor Review: {row['review_score']:.2f}")
                print(f"   ▪ Ingredients: {', '.join(row['ingredients'])}")
        else:
            logging.warning("❗ Tidak ada produk yang cocok dengan preferensi user.")

    except Exception as e:
        logging.error(f"Gagal menjalankan sistem: {str(e)}")

# ------------------- RUN ------------------- #
if __name__ == "__main__":
    main()
