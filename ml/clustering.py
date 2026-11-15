
from pathlib import Path
import numpy as np
import pandas as pd
import umap
import hdbscan


# 1. PROJECT PATHS


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EMBEDDINGS_PATH = PROJECT_ROOT / "artifacts" / "review_embeddings.npy"
METADATA_PATH = PROJECT_ROOT / "artifacts" / "embedding_metadata.csv"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


# 2. LOAD EMBEDDINGS AND METADATA


embeddings = np.load(EMBEDDINGS_PATH)
df = pd.read_csv(METADATA_PATH)

print("Embeddings shape:", embeddings.shape)
print("Metadata shape:", df.shape)

# Ensure each embedding corresponds to one metadata row
assert embeddings.shape[0] == len(df), (
    "Mismatch: embedding count and metadata row count differ!"
)

# 3. UMAP DIMENSIONALITY REDUCTION

print("\nStarting UMAP...")

reducer = umap.UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    random_state=42,
    low_memory=True
)

reduced_embeddings = reducer.fit_transform(embeddings)

print("UMAP completed!")
print("Reduced embedding shape:", reduced_embeddings.shape)

np.save(
    ARTIFACTS_DIR / "reduced_embeddings.npy",
    reduced_embeddings
)

# 4. HDBSCAN CLUSTERING


print("\nStarting HDBSCAN...")

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=30,
    min_samples=5,
    metric="euclidean",
    cluster_selection_method="eom",
    core_dist_n_jobs=2
)

cluster_labels = clusterer.fit_predict(reduced_embeddings)


# 5. SAVE CLUSTER LABELS


df["cluster"] = cluster_labels

df.to_csv(
    ARTIFACTS_DIR / "clustered_reviews.csv",
    index=False,
    encoding="utf-8"
)

np.save(
    ARTIFACTS_DIR / "cluster_labels.npy",
    cluster_labels
)


# 6. CLUSTER SUMMARY


unique_labels, counts = np.unique(
    cluster_labels,
    return_counts=True
)

print("\nCluster distribution:")

for label, count in zip(unique_labels, counts):
    if label == -1:
        print(f"Noise (-1): {count} reviews")
    else:
        print(f"Cluster {label}: {count} reviews")

noise_count = np.sum(cluster_labels == -1)
cluster_count = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

print("\nClustering completed!")
print("Total clusters:", cluster_count)
print("Noise reviews:", noise_count)
print("Noise percentage:", round(noise_count / len(df) * 100, 2), "%")
print("Saved clustered reviews to:", ARTIFACTS_DIR)
