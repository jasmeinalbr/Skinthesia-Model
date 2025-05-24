# utils/extracts/product_scraper.py

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
import time

def get_products_from_category(driver, category_url, category_name, max_products=100):
    """
    Scrap daftar produk dari suatu kategori di Female Daily.
    """
    driver.get(category_url)
    time.sleep(3)

    products = []
    product_urls = set()  # Untuk mencegah duplikat

    while len(products) < max_products:
        # Ambil semua product card
        product_cards = driver.find_elements(By.CSS_SELECTOR, 'a.product-card')

        for card in product_cards:
            try:
                url = card.get_attribute("href")
                if url in product_urls:
                    continue  # skip duplikat

                name = card.find_element(By.CSS_SELECTOR, "p.two-line").text.strip()
                brand = card.find_element(By.CLASS_NAME, 'fd-body-md-bold').text.strip()
                img = card.find_element(By.TAG_NAME, 'img').get_attribute("src")
                rating = card.find_element(By.CSS_SELECTOR, '.rating span.fd-body-sm-regular').text.strip()
                total_reviews = card.find_element(By.CSS_SELECTOR, '.rating span.grey').text.strip().strip("()")

                products.append({
                    "name": name,
                    "brand": brand,
                    "image": img,
                    "url": url,
                    "rating": rating,
                    "total_reviews": total_reviews,
                    "category": category_name
                })
                product_urls.add(url)

                if len(products) >= max_products:
                    break
            except Exception as e:
                print(f"[!] Error parsing product card: {e}")
                continue

        # Cari dan klik tombol Load More
        try:
            load_more_btn = driver.find_element(By.CSS_SELECTOR, "button.btn-load-more")
            driver.execute_script("arguments[0].scrollIntoView(true);", load_more_btn)
            time.sleep(1)
            driver.execute_script("arguments[0].click();", load_more_btn)

            # Tunggu produk tambahan muncul (menunggu card baru)
            WebDriverWait(driver, 10).until(
                lambda d: len(d.find_elements(By.CSS_SELECTOR, 'a.product-card')) > len(product_urls)
            )
            time.sleep(2)
        except Exception as e:
            print("[!] Tidak ada tombol Load More atau sudah habis. Stop.")
            break

    return products
