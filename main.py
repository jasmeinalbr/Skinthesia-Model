# main.py

import logging
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from utils.ETL.extracts.category_scraper import get_categories
from utils.ETL.extracts.product_scraper import get_products_from_category
from utils.ETL.extracts.detail_scraper import get_product_details
import pandas as pd
import csv
import os

def setup_logger():
    # buat folder logs jika belum ada
    if not os.path.exists("logs"):
        os.makedirs("logs")
    # setup logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/scraper_log.txt", mode='w'),
            logging.StreamHandler()
        ]
    )

def run_scraper():
    setup_logger()

    # Setup headless browser
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)

    all_products = []
    all_details = []
    all_reviews = []

    try:
        categories = get_categories(limit=10)  # ganti sesuai kebutuhan
        for cat in categories:
            logging.info(f"\n📂 Memproses kategori: {cat['name']} ({cat['url']})")
            products = get_products_from_category(driver, cat['url'], cat['name'], max_products=100)

            logging.info(f"✅ Produk ditemukan: {len(products)}")
            for i, prod in enumerate(products, 1):
                logging.info(f"[{i}] {prod['brand']} - {prod['name']} | {prod['rating']} ({prod['total_reviews']} reviews)")
            
            all_products.extend(products)

            # Scrape detail tiap produk
            for prod in products:
                try:
                    detail = get_product_details(driver, prod["url"])
                    if detail:
                        all_details.append({k: v for k, v in detail.items() if k != "reviews"})  # exclude nested
                        for rev in detail["reviews"]:
                            rev["product_name"] = detail["name"]
                            rev["product_url"] = detail["url"]
                            all_reviews.append(rev)
                        logging.info(f"📌 Detail & review diambil: {detail['name']}")
                except Exception as e:
                    logging.warning(f"❌ Gagal ambil detail: {prod['name']} - {e}")

    finally:
        driver.quit()
        logging.info("Browser ditutup.")

    # Simpan data
    if not os.path.exists("data"):
        os.makedirs("data")

    if all_products:
        pd.DataFrame(all_products).to_csv("data/products_list.csv", index=False, encoding="utf-8-sig")
        logging.info(f"📝 Data produk disimpan ke data/products_list.csv ({len(all_products)} baris)")
    if all_details:
        pd.DataFrame(all_details).to_csv("data/products_detail.csv", index=False, encoding="utf-8-sig")

        logging.info(f"📝 Data detail disimpan ke data/products_detail.csv ({len(all_details)} baris)")
    if all_reviews:
        pd.DataFrame(all_reviews).to_csv("data/reviews.csv", index=False, encoding="utf-8-sig")
        logging.info(f"📝 Review disimpan ke data/reviews.csv ({len(all_reviews)} baris)")

if __name__ == "__main__":
    run_scraper()