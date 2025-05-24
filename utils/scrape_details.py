# scrape_details.py

import logging
import os
import time
import random
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import SessionNotCreatedException
from ETL.extracts.detail_scraper import get_product_details

def setup_logger():
    if not os.path.exists("logs"):
        os.makedirs("logs")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/detail_scraper_log.txt", mode='a'),
            logging.StreamHandler()
        ]
    )

def run_all_batches(batch_size=20, pause_per_batch=10, start_from=0, end_at=None):
    setup_logger()

    try:
        df_products = pd.read_csv("data/products_list.csv")
    except FileNotFoundError:
        logging.error("❌ File products_list.csv tidak ditemukan.")
        return

    total_products = len(df_products)
    if end_at is None or end_at > total_products:
        end_at = total_products

    logging.info(f"📦 Total produk tersedia: {total_products}")
    logging.info(f"🔢 Range scraping: index {start_from} sampai {end_at - 1}")

    for start in range(start_from, end_at, batch_size):
        logging.info(f"\n🔍 Mulai batch dari index {start} sampai {min(start + batch_size - 1, end_at - 1)}")

        # Delay sebelum membuka sesi browser baru (fix DevToolsActivePort)
        time.sleep(3)
        options = Options()
        # options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        try:
            driver = webdriver.Chrome(options=options)
        except SessionNotCreatedException as e:
            logging.error(f"🔥 Gagal membuat sesi browser baru: {e}")
            break

        batch = df_products[start: start + batch_size]
        details = []
        reviews = []

        for i, row in batch.iterrows():
            url = row["url"]
            try:
                logging.info(f"🔗 Ambil detail: {row['name']} ({url})")
                detail = get_product_details(driver, url)
                if detail:
                    details.append({k: v for k, v in detail.items() if k != "reviews"})
                    for rev in detail["reviews"]:
                        rev["product_name"] = detail["name"]
                        rev["product_url"] = detail["url"]
                        reviews.append(rev)
                    logging.info(f"✅ Detail & review diambil: {detail['name']} ({i})")
            except Exception as e:
                logging.warning(f"❌ Gagal ambil detail dari: {url} - {e}")
            time.sleep(random.uniform(1.5, 4))  # delay antar produk

        driver.quit()
        logging.info("🧹 Browser ditutup untuk batch ini.")

        if details:
            pd.DataFrame(details).to_csv("data-final 0-599/tmp_details.csv", mode='a', header=not os.path.exists("data-final 0-599/tmp_details.csv"), index=False, encoding='utf-8-sig')
        if reviews:
            pd.DataFrame(reviews).to_csv("data-final 0-599/tmp_reviews.csv", mode='a', header=not os.path.exists("data-final 0-599/tmp_reviews.csv"), index=False, encoding='utf-8-sig')

        logging.info(f"💾 Batch disimpan: {len(details)} detail, {len(reviews)} review")

        # Update full merge file
        if os.path.exists("data-final 0-599/tmp_details.csv"):
            pd.read_csv("data-final 0-599/tmp_details.csv").drop_duplicates().to_csv("data-final 0-599/products_detail.csv", index=False, encoding="utf-8-sig")
        if os.path.exists("data-final 0-599/tmp_reviews.csv"):
            pd.read_csv("data-final 0-599/tmp_reviews.csv").drop_duplicates().to_csv("data-final 0-599/reviews.csv", index=False, encoding="utf-8-sig")

        logging.info(f"⏳ Menunggu {pause_per_batch} detik sebelum batch selanjutnya...")
        time.sleep(pause_per_batch)

if __name__ == "__main__":
    # Ganti parameter sesuai kebutuhan kamu
    run_all_batches(batch_size=20, pause_per_batch=20, start_from=0, end_at=None)
