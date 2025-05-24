# merge_batches.py

import pandas as pd

def merge_csv_batches(file_1, file_2, output_file):
    df1 = pd.read_csv(file_1)
    df2 = pd.read_csv(file_2)

    combined = pd.concat([df1, df2], ignore_index=True).drop_duplicates()
    combined.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"[✓] File df1: {file_1} ({len(df1)} baris)")
    print(f"[✓] File df2: {file_2} ({len(df2)} baris)")
    print(f"[✓] File digabung & disimpan ke: {output_file} ({len(combined)} baris)")

if __name__ == "__main__":
    # Gabungkan detail produk
    merge_csv_batches(
        "data-final-0-599/products_detail_first.csv",
        "data-final 600-100/products_detail.csv",
        "data/products_detail.csv"
    )

    # Gabungkan review
    merge_csv_batches(
        "data-final-0-599/reviews_first.csv",
        "data-final 600-100/reviews.csv",
        "data/reviews.csv"
    )

    merge_csv_batches(
        "data-final-0-599/tmp_details_first.csv",
        "data-final 600-100/tmp_details.csv",
        "data/tmp_details.csv"
    )

    merge_csv_batches(
        "data-final-0-599/tmp_reviews_first.csv",
        "data-final 600-100/tmp_reviews.csv",
        "data/tmp_reviews.csv"
    )

