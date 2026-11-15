from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

repo_id = "McAuley-Lab/Amazon-Reviews-2023"

filename = (
    "raw_review_Electronics/"
    "full-00000-of-00034.parquet"
)

file_path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    repo_type="dataset",
    revision="758b2e8f09ca2239eeac8fb9cd9ea83d19785f7a"
)

print("Downloaded successfully:", file_path)

parquet_file = pq.ParquetFile(file_path)

print("Total rows:", parquet_file.metadata.num_rows)
print("Columns:", parquet_file.schema.names)

parquet_file = pq.ParquetFile(file_path)

from collections import Counter

product_counts = Counter()

for batch in parquet_file.iter_batches(
    batch_size=10000,
    columns=["parent_asin"]
):
    product_ids = batch.column("parent_asin").to_pylist()

    product_counts.update(
        product_id
        for product_id in product_ids
        if product_id
    )

print("\nTop 10 products by review count:\n")

for product, count in product_counts.most_common(10):
    print(f"Product ID: {product} | Reviews: {count}")


import random
import pandas as pd
import os

TARGET_PRODUCT = "B01G8JO5F2"
SAMPLE_SIZE = 3000

# Reproducible random sampling
rng = random.Random(42)
selected_reviews = []
seen = 0

for batch in parquet_file.iter_batches(
    batch_size=10000,
    columns=["parent_asin", "text", "rating", "timestamp"]
):
    for row in batch.to_pylist():

        if row["parent_asin"] != TARGET_PRODUCT:
            continue

        if not row["text"] or not row["text"].strip():
            continue

        seen += 1

        review = {
            "parent_asin": row["parent_asin"],
            "text": row["text"],
            "rating": row["rating"],
            "timestamp": row["timestamp"]
        }

        if len(selected_reviews) < SAMPLE_SIZE:
            selected_reviews.append(review)
        else:
            position = rng.randint(0, seen - 1)

            if position < SAMPLE_SIZE:
                selected_reviews[position] = review

# Save dataset
os.makedirs("datasets", exist_ok=True)

df = pd.DataFrame(selected_reviews)

df.to_csv(
    "datasets/product_reviews.csv",
    index=False,
    encoding="utf-8"
)

print("\nExtraction complete!")
print("Matching reviews:", seen)
print("Saved reviews:", len(df))
print("Dataset shape:", df.shape)
print(df.head())