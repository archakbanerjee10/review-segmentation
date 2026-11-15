
import pandas as pd
import numpy as np
import re
import html
import unicodedata
import os

# 1. LOAD DATASET
df = pd.read_csv("../datasets/product_reviews.csv")

print("Original dataset shape:", df.shape)



# 2. HANDLE MISSING AND EMPTY REVIEWS

print("\nMissing values before cleaning:")
print(df.isnull().sum())

df = df.dropna(subset=["text"]).copy()

df["text"] = df["text"].astype(str)

df = df[df["text"].str.strip() != ""].copy()

print("\nShape after removing missing reviews:", df.shape)


# 3. REMOVE EXACT DUPLICATE RECORDS

print("\nExact duplicate rows:", df.duplicated().sum())

df = df.drop_duplicates().copy()

print("Shape after removing exact duplicates:", df.shape)



# 4. TEXT CLEANING FUNCTION


def clean_review(text):

    text = str(text)

    # Decode HTML entities
    text = html.unescape(text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Normalize Unicode
    text = unicodedata.normalize("NFKC", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()



# 5. APPLY TEXT CLEANING


df["cleaned_text"] = df["text"].apply(clean_review)


# 6. REVIEW LENGTH ANALYSIS


df["word_count"] = (
    df["cleaned_text"]
    .str.split()
    .str.len()
)

print("\nWord count statistics:")
print(df["word_count"].describe())

print("\nShortest reviews:")
print(
    df.nsmallest(10, "word_count")[
        ["rating", "cleaned_text", "word_count"]
    ]
)


# 7. COMPARE ORIGINAL AND CLEANED REVIEWS


print("\nOriginal vs cleaned reviews:")

for i in range(5):

    print("\nORIGINAL:")
    print(df["text"].iloc[i])

    print("\nCLEANED:")
    print(df["cleaned_text"].iloc[i])

    print("-" * 60)



# 8. SAVE CLEANED DATASET


os.makedirs("datasets", exist_ok=True)

df.to_csv(
    "datasets/cleaned_reviews.csv",
    index=False,
    encoding="utf-8"
)

print("\nPreprocessing completed!")
print("Final dataset shape:", df.shape)
print("Saved to datasets/cleaned_reviews.csv")
