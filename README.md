
# Skinthesia-Model

Skinthesia-Model adalah pipeline untuk proses **ETL (Extract, Transform, Load)**, **pelatihan model klasifikasi multi-label**, dan **sistem rekomendasi produk**.

---

## 📂 Struktur Proyek

```
Skinthesia-Model/
├── data/                      # Dataset mentah dan hasil ekstraksi fitur
├── logs/                     # Log proses ETL/modeling (jika digunakan)
├── notebooks/                # Exploratory Data Analysis (EDA) dan evaluasi
├── model_tfjs/               # Model TensorFlow (untuk deployment web)
├── models/                   # Model klasifikasi & konfigurasi
├── utils/                    # Fungsi bantu
├── ETL/                      # Pipeline ETL & web scraping
│   ├── extracts/             # Scraper: produk, detail, review, kategori
│   ├── load.py               # Loader data
│   ├── transform.py          # Transformasi & ekstraksi fitur
│   ├── scrape_*.py           # Runner scraping per tipe
├── src/
│   ├── recommendation_system.py            # Sistem rekomendasi produk
├── main.py                   # Runner pipeline end-to-end
├── requirements.txt          # Dependency Python
└── .gitignore
```

---

## ⚙️ Instalasi

1. **Clone repo**:
   ```bash
   git clone https://github.com/jasmeinalbr/Skinthesia-Model.git
   cd Skinthesia-Model
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

> 📌 Tidak perlu membuat virtual environment khusus, tapi sangat disarankan agar lingkungan tetap bersih.

---

## 🚀 Cara Menjalankan

### 🔎 1. Scraping & ETL
Jalankan masing-masing skrip scraping:
```bash
python ETL/scrape_products.py
python ETL/scrape_details.py
```

Lalu, transformasi datanya:
```bash
python ETL/transform.py
```

Terakhir, load datanya ke database:
```bash
python ETL/load.py
```

Hasil akhir ada di folder `data/`, misalnya:
- `products_integrated_features.csv`

---

### 🤖 2. Training Model Klasifikasi Multi-Label

Lakukan modeling & evaluasi dari notebook berikut:
```bash
notebooks/multi_label_classif_modeling.ipynb
```
Model disimpan di `models/` dalam format `.keras` dan JSON untuk inference.

---

### 🎯 3. Inference & Sistem Rekomendasi

Untuk menjalankan sistem rekomendasi produk:
```bash
python src/recommendation_system.py
```

Model default digunakan dari:
- `models/ingredients_category_classification_model.keras`
- `models/mlb_classes.json`

---

## 📒 Notebook Analisis

Tersedia beberapa Jupyter Notebook untuk eksplorasi data & evaluasi:
- `EDA_scrape_result.ipynb` – analisis data hasil scraping
- `EDA_transform_result.ipynb` – analisis hasil transformasi fitur
- `multi_label_classif_modeling.ipynb` – pelatihan & evaluasi model

---

## 📤 Output Penting

- 📁 `data/`: semua hasil scraping dan transformasi fitur.
- 📁 `models/`: model multi-label klasifikasi dan encoder.
- 📁 `model_tfjs/`: model dalam format TensorFlow.js (untuk web deployment).
- 📁 `logs/`: log proses (jika digunakan).

---

## 📌 Catatan

- Dataset menggunakan kombinasi **produk skincare** dan **review** dari scraping.
- Klasifikasi multi-label digunakan untuk memetakan produk ke beberapa kategori bahan.
- Sistem rekomendasi menyaring produk berdasarkan hasil klasifikasi dan fitur produk lainnya.

