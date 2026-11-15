
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "artifacts" / "clustered_reviews.csv"
EMBEDDINGS_FILE = PROJECT_ROOT / "artifacts" / "review_embeddings.npy"

OUTPUT_DIR = PROJECT_ROOT / "artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("Loading clustered reviews...")

df = pd.read_csv(INPUT_FILE)
embeddings = np.load(EMBEDDINGS_FILE)

print("Dataset shape:", df.shape)
print("Embeddings shape:", embeddings.shape)

if len(df) != len(embeddings):
    raise ValueError(
        "Review count and embedding count do not match."
    )

required_columns = ["cleaned_text", "cluster", "rating"]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")


# --------------------------------------------------
# 3. REMOVE NOISE FOR CLUSTER ANALYSIS
# --------------------------------------------------

clustered_df = df[df["cluster"] != -1].copy()

# Preserve original row positions for embedding lookup.
clustered_df["original_index"] = clustered_df.index

print("\nTotal reviews:", len(df))
print("Reviews assigned to clusters:", len(clustered_df))
print("Noise reviews:", (df["cluster"] == -1).sum())
print("Number of clusters:", clustered_df["cluster"].nunique())


# --------------------------------------------------
# 4. CLUSTER STATISTICS
# --------------------------------------------------

print("\nGenerating cluster statistics...")

cluster_stats = (
    clustered_df
    .groupby("cluster")
    .agg(
        review_count=("cleaned_text", "count"),
        average_rating=("rating", "mean"),
        median_rating=("rating", "median"),
        rating_std=("rating", "std")
    )
    .reset_index()
)

cluster_stats["average_rating"] = (
    cluster_stats["average_rating"].round(2)
)

cluster_stats["rating_std"] = (
    cluster_stats["rating_std"].round(2)
)

cluster_stats.to_csv(
    OUTPUT_DIR / "cluster_statistics.csv",
    index=False
)


# --------------------------------------------------
# 5. RATING DISTRIBUTION
# --------------------------------------------------

rating_distribution = pd.crosstab(
    clustered_df["cluster"],
    clustered_df["rating"]
)

rating_distribution.to_csv(
    OUTPUT_DIR / "cluster_rating_distribution.csv"
)


# --------------------------------------------------
# 6. TF-IDF KEYWORD EXTRACTION
# --------------------------------------------------

print("\nExtracting cluster keywords...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=5000,
    min_df=2,
    max_df=0.90
)

tfidf_matrix = vectorizer.fit_transform(
    clustered_df["cleaned_text"].fillna("")
)

terms = np.array(vectorizer.get_feature_names_out())

keyword_rows = []

for cluster_id in sorted(clustered_df["cluster"].unique()):

    positions = np.where(
        clustered_df["cluster"].to_numpy() == cluster_id
    )[0]

    cluster_tfidf = tfidf_matrix[positions]

    mean_scores = np.asarray(
        cluster_tfidf.mean(axis=0)
    ).ravel()

    top_indices = mean_scores.argsort()[-15:][::-1]

    for rank, idx in enumerate(top_indices, start=1):

        if mean_scores[idx] <= 0:
            continue

        keyword_rows.append({
            "cluster": cluster_id,
            "rank": rank,
            "keyword": terms[idx],
            "score": round(float(mean_scores[idx]), 4)
        })


keywords_df = pd.DataFrame(keyword_rows)

keywords_df.to_csv(
    OUTPUT_DIR / "cluster_keywords.csv",
    index=False
)


# --------------------------------------------------
# 7. REPRESENTATIVE REVIEWS
# --------------------------------------------------

print("\nFinding representative reviews...")

representative_rows = []

for cluster_id in sorted(clustered_df["cluster"].unique()):

    cluster_data = clustered_df[
        clustered_df["cluster"] == cluster_id
    ]

    original_indices = cluster_data["original_index"].to_numpy()

    cluster_embeddings = embeddings[original_indices]

    # Calculate cluster centroid.
    centroid = cluster_embeddings.mean(axis=0, keepdims=True)

    # Find reviews closest to the centroid.
    similarities = cosine_similarity(
        cluster_embeddings,
        centroid
    ).ravel()

    top_positions = np.argsort(similarities)[::-1][:5]

    for rank, position in enumerate(top_positions, start=1):

        row = cluster_data.iloc[position]

        representative_rows.append({
            "cluster": cluster_id,
            "rank": rank,
            "rating": row["rating"],
            "similarity_to_centroid": round(
                float(similarities[position]), 4
            ),
            "review": row["cleaned_text"]
        })


representatives_df = pd.DataFrame(representative_rows)

representatives_df.to_csv(
    OUTPUT_DIR / "representative_reviews.csv",
    index=False,
    encoding="utf-8"
)



# 8. FINAL SUMMARY


print("\n" + "=" * 60)
print("CLUSTER ANALYSIS COMPLETED")
print("=" * 60)

print("\nCluster statistics:")
print(cluster_stats.to_string(index=False))

print("\nGenerated files:")

print("1. cluster_statistics.csv")
print("2. cluster_rating_distribution.csv")
print("3. cluster_keywords.csv")
print("4. representative_reviews.csv")

print("\nAll files saved in:", OUTPUT_DIR)
