# scrape_products.py

import logging
import os
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from utils.ETL.extracts.category_scraper import get_categories
from utils.ETL.extracts.product_scraper import get_products_from_category

def setup_logger():
    if not os.path.exists("logs"):
        os.makedirs("logs")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/scraper_products_log.txt", mode='w'),
            logging.StreamHandler()
        ]
    )

def main():
    setup_logger()

    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)

    all_products = []

    try:
        categories = get_categories(limit=10)
        for cat in categories:
            logging.info(f"\n📂 Kategori: {cat['name']} ({cat['url']})")
            products = get_products_from_category(driver, cat['url'], cat['name'], max_products=100)
            logging.info(f"✅ {len(products)} produk ditemukan")
            for i, prod in enumerate(products, 1):
                logging.info(f"[{i}] {prod['brand']} - {prod['name']} | {prod['rating']} ({prod['total_reviews']} reviews)")
            all_products.extend(products)
    finally:
        driver.quit()

    if not os.path.exists("data"):
        os.makedirs("data")
    df = pd.DataFrame(all_products)
    df.to_csv("data/products_list.csv", index=False, encoding="utf-8-sig")
    logging.info(f"📝 Disimpan ke data/products_list.csv ({len(df)} baris)")

if __name__ == "__main__":
    main()
