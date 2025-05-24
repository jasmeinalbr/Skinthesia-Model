# utils/extracts/category_scraper.py

def get_categories(limit=None):
    """
    Mengembalikan daftar kategori skincare yang akan di-scrape dari Female Daily,
    berdasarkan struktur URL kategori yang sudah ditentukan.
    """
    base_url = "https://reviews.femaledaily.com/products"
    category_paths = [
        ("Cleanser / Toner", "cleanser/toner"),
        ("Cleanser / Facial Wash", "cleanser/facial-wash"),
        ("Cleanser / Scrub-Exfoliator", "cleanser/scrub-exfoliator"),
        ("Treatment / Serum & Essence", "treatment/serum-essence"),
        ("Treatment / Acne Treatment", "treatment/acne-treatment"),
        ("Moisturizer / Lotion & Emulsion", "moisturizer/lotion-emulsion"),
        ("Moisturizer / Gel", "moisturizer/gel"),
        ("Moisturizer / Sun Protection", "moisturizer/sun-protection"),
        ("Moisturizer / Cream", "moisturizer/cream"),
        ("Treatment / Peeling", "treatment/peeling"),
    ]

    if limit:
        category_paths = category_paths[:limit]

    categories = []
    for name, path in category_paths:
        full_url = f"{base_url}/{path}?brand=&order=popular&page=1"
        categories.append({
            "name": name,
            "url": full_url,
            "path": path  # untuk identifikasi atau penyimpanan per kategori
        })
    return categories