# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from .review_scraper import scrape_reviews  
# import time

# def get_product_details(driver, product_url):
#     """
#     Scrape detail dari halaman produk Female Daily.
#     """
#     driver.get(product_url)
#     time.sleep(2)

#     detail = {
#         "url": product_url,
#         "name": None,
#         "brand": None,
#         "price": None,
#         "rating": None,
#         "description": None,
#     }

#     try:
#         # Nama dan Brand
#         name = driver.find_element(By.CLASS_NAME, "product-name").text.strip()
#         brand = driver.find_element(By.CLASS_NAME, "product-brand").text.strip()
#         detail["name"] = name
#         detail["brand"] = brand
#     except Exception as e:
#         print(f"[!] Gagal ambil nama/brand: {e}")

#     try:
#     # cek apakah elemen harga ada jika tidak ada isi undefined
#         price_element = driver.find_elements(By.CLASS_NAME, "product-price")
#         if len(price_element) == 0:
#             price = "N/A"
#         else:
#             price = driver.find_element(By.CLASS_NAME, "product-price").text.strip()
#         detail["price"] = price
#     except Exception as e:
#         print(f"[!] Gagal ambil harga: {e}")

#     try:
#         # Rating (angka besar di kiri atas)
#         rating = driver.find_element(By.CSS_SELECTOR, ".data-wrapper.total p").text.strip()
#         detail["rating"] = rating
#     except Exception as e:
#         print(f"[!] Gagal ambil rating: {e}")

#     try:
#         # Klik tombol deskripsi jika belum muncul
#         desc_button = driver.find_element(By.ID, "id_button_description")
#         driver.execute_script("arguments[0].click();", desc_button)
#         time.sleep(1)
#     except:
#         pass  # tombol tidak selalu ada

#     try:
#         # Ambil deskripsi
#         description_container = driver.find_element(By.CLASS_NAME, "product-desc-wrapper")
#         desc_text = description_container.text.replace("\n", " ").replace("Description", "").strip()
#         detail["description"] = desc_text
#     except Exception as e:
#         print(f"[!] Deskripsi tidak ditemukan: {e}")

#     try:
#         reviews = scrape_reviews(driver, max_pages=15)
#         detail["reviews"] = reviews
#     except Exception as e:
#         print(f"[!] Gagal ambil review: {e}")

#     return detail

# detail_scraper.py (tidak banyak perubahan)
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
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
