
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "datasets" / "english_reviews.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. LOAD ENGLISH-ONLY DATASET
# --------------------------------------------------

print("Loading English reviews...")

df = pd.read_csv(INPUT_FILE)

# Remove missing or empty reviews
df = df.dropna(subset=["cleaned_text"]).copy()

df["cleaned_text"] = df["cleaned_text"].astype(str).str.strip()

df = df[df["cleaned_text"] != ""].copy()

df.reset_index(drop=True, inplace=True)

print("Dataset shape:", df.shape)
print("Reviews to encode:", len(df))

# --------------------------------------------------
# 3. LOAD SENTENCE TRANSFORMER MODEL
# --------------------------------------------------

print("\nLoading Sentence Transformer model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

# Limit maximum sequence length
model.max_seq_length = 160

print("Model loaded successfully!")

# --------------------------------------------------
# 4. GENERATE EMBEDDINGS
# --------------------------------------------------

print("\nGenerating embeddings...")

embeddings = model.encode(
    df["cleaned_text"].tolist(),
    batch_size=8,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

embeddings = embeddings.astype(np.float32)

print("\nEmbedding generation completed!")
print("Embedding shape:", embeddings.shape)
print("Embedding data type:", embeddings.dtype)

# --------------------------------------------------
# 5. SAVE EMBEDDINGS
# --------------------------------------------------

embedding_file = ARTIFACTS_DIR / "review_embeddings.npy"

np.save(embedding_file, embeddings)

# --------------------------------------------------
# 6. SAVE METADATA
# --------------------------------------------------

metadata_file = ARTIFACTS_DIR / "embedding_metadata.csv"

df.to_csv(
    metadata_file,
    index=False,
    encoding="utf-8"
)

print("\nFiles saved successfully!")
print("Embeddings:", embedding_file)
print("Metadata:", metadata_file)

# --------------------------------------------------
# 7. VERIFY OUTPUTS
# --------------------------------------------------

saved_embeddings = np.load(embedding_file)
saved_metadata = pd.read_csv(metadata_file)

print("\nVerification:")
print("Embedding count:", saved_embeddings.shape[0])
print("Metadata count:", len(saved_metadata))
print("Embedding dimensions:", saved_embeddings.shape[1])

assert saved_embeddings.shape[0] == len(saved_metadata), \
    "ERROR: Embedding and metadata counts do not match!"

print("\nSUCCESS: English-only embeddings generated!")
