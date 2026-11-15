import pandas as pd
from langdetect import detect_langs, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from pathlib import Path

# Make language detection reproducible
DetectorFactory.seed = 42

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "datasets" / "cleaned_reviews.csv"
OUTPUT_FILE = PROJECT_ROOT / "datasets" / "english_reviews.csv"

# Load cleaned reviews
df = pd.read_csv(INPUT_FILE)

print("Original dataset:", df.shape)


def detect_language(text):
    text = str(text).strip()

    # Very short reviews are difficult to classify reliably
    if len(text.split()) < 3:
        return pd.Series(["uncertain", 0.0])

    try:
        predictions = detect_langs(text)
        best_prediction = predictions[0]

        return pd.Series([
            best_prediction.lang,
            best_prediction.prob
        ])

    except LangDetectException:
        return pd.Series(["unknown", 0.0])


# Detect language
df[["language", "language_confidence"]] = df["cleaned_text"].apply(
    detect_language
)

# Keep confidently detected English reviews
english_df = df[
    (df["language"] == "en") &
    (df["language_confidence"] >= 0.80)
].copy()

# Save English-only dataset
english_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

print("\nLanguage distribution:")
print(df["language"].value_counts())

print("\nOriginal reviews:", len(df))
print("English reviews retained:", len(english_df))
print("Reviews excluded:", len(df) - len(english_df))
print("\nSaved to:", OUTPUT_FILE)