# utils/extracts/review_scraper.py

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time

def scrape_reviews(driver, max_pages=5):
    """
    Scrape review dari halaman produk
    """
    reviews = []
    page = 1

    while page <= max_pages:
        time.sleep(2)

        # Deteksi halaman error redirect
        if "Oops! We're sorry" in driver.page_source:
            print(f"[!] Halaman {page} kosong / redirect error.")
            break

        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "review-card"))
            )
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            review_blocks = soup.select("div.review-card")
            print(f"[✓] Halaman {page} berhasil ambil {len(review_blocks)} review")

            if not review_blocks:
                print(f"[!] Halaman {page} tidak ada review.")
                break

            for block in review_blocks:
                try:
                    username_tag = block.select_one("p.profile-username")
                    age_tag = block.select_one("p.profile-age")
                    skin_desc_tag = block.select_one("p.profile-description")
                    recommend_tag = block.select_one("p.recommend b")
                    review_text_tag = block.select_one("p.text-content")
                    review_date_tag = block.select_one("p.review-date")
                    usage_period_tag = block.select_one("div.information-wrapper b")

                    reviews.append({
                        "username": username_tag.text.strip() if username_tag else None,
                        "skin_type": skin_desc_tag.text.strip().split(",")[0].strip() if skin_desc_tag else None,
                        "age": age_tag.text.strip() if age_tag else None,
                        "rating_star": len(block.select("span.cardrv-starlist i.icon-ic_big_star_full")),
                        "recommended": None if not recommend_tag else "doesn't" not in recommend_tag.text.lower(),
                        "review": review_text_tag.text.strip() if review_text_tag else None,
                        "review_date": review_date_tag.text.strip() if review_date_tag else None,
                        "usage_period": usage_period_tag.text.strip() if usage_period_tag else None
                    })

                except Exception as e:
                    print(f"[!] Gagal parsing satu review: {e}")
                    continue

            # Klik tombol Next (dengan retry)
            try:
                next_button = driver.find_element(By.ID, "id_next_page")
                driver.execute_script("arguments[0].scrollIntoView(true);", next_button)
                time.sleep(1)
                driver.execute_script("arguments[0].click();", next_button)
                page += 1
            except Exception as e:
                print(f"[!] Gagal klik tombol Next di halaman {page}: {e}")
                break

        except Exception as e:
            print(f"[!] Gagal proses halaman {page}: {e}")
            break

    print(f"[✓] Review selesai sampai halaman {page - 1}")
    return reviews

