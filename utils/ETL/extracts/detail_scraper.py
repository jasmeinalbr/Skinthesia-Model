# utils/extracts/detail_scraper.py

from selenium.webdriver.common.by import By
from .review_scraper import scrape_reviews
import time

def get_product_details(driver, product_url):
    """
    Scrape detail dari halaman produk Female Daily.
    """
    driver.get(product_url)
    time.sleep(2)

    detail = {
        "url": product_url,
        "name": None,
        "brand": None,
        "price": None,
        "rating": None,
        "description": None,
    }

    try:
        detail["name"] = driver.find_element(By.CLASS_NAME, "product-name").text.strip()
        detail["brand"] = driver.find_element(By.CLASS_NAME, "product-brand").text.strip()
    except Exception as e:
        print(f"[!] Gagal ambil nama/brand: {e}")

    try:
        prices = driver.find_elements(By.CLASS_NAME, "product-price")
        detail["price"] = prices[0].text.strip() if prices else "N/A"
    except Exception as e:
        print(f"[!] Gagal ambil harga: {e}")

    try:
        detail["rating"] = driver.find_element(By.CSS_SELECTOR, ".data-wrapper.total p").text.strip()
    except Exception as e:
        print(f"[!] Gagal ambil rating: {e}")

    try:
        desc_button = driver.find_element(By.ID, "id_button_description")
        driver.execute_script("arguments[0].click();", desc_button)
        time.sleep(1)
    except:
        pass

    try:
        desc = driver.find_element(By.CLASS_NAME, "product-desc-wrapper")
        detail["description"] = desc.text.replace("\n", " ").replace("Description", "").strip()
    except Exception as e:
        print(f"[!] Deskripsi tidak ditemukan: {e}")

    try:
        detail["reviews"] = scrape_reviews(driver, max_pages=15)
    except Exception as e:
        print(f"[!] Gagal ambil review: {e}")

    return detail
