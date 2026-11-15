from pathlib import Path
import pandas as pd
from transformers import pipeline

# Define project root and paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MENTIONS_CSV = PROJECT_ROOT / "artifacts" / "aspect_mentions.csv"
OUTPUT_SENTIMENT = PROJECT_ROOT / "artifacts" / "aspect_sentiment.csv"


def main():
  print(f"Loading aspect mentions from: {MENTIONS_CSV}")
  if not MENTIONS_CSV.exists():
    raise FileNotFoundError(
        f"Aspect mentions not found at {MENTIONS_CSV}. Run aspect_extraction.py"
        " first."
    )

  df = pd.read_csv(MENTIONS_CSV)
  print(f"Loaded {len(df)} aspect mentions for sentiment classification.")

  # Initialize lightweight CPU-friendly sentiment pipeline
  # distilbert-base-uncased-finetuned-sst-2-english is fast and lightweight for CPU
  print(
      "Loading lightweight sentiment model"
      " (distilbert-base-uncased-finetuned-sst-2-english)..."
  )
  sentiment_pipeline = pipeline(
      "sentiment-analysis",
      model="distilbert-base-uncased-finetuned-sst-2-english",
      device=-1,  # Forces CPU execution
  )

  sentiments = []
  confidences = []

  print("Classifying sentiment for each aspect mention...")
  # Process in batches or loop through records
  # Truncate text to 512 tokens max length for transformer safety
  texts = df["review_text"].astype(str).tolist()

  # Batch inference for efficiency
  batch_size = 16
  for i in range(0, len(texts), batch_size):
    batch_texts = texts[i : i + batch_size]
    # Truncate to avoid exceeding model sequence limits
    batch_texts = [t[:1000] for t in batch_texts]

    results = sentiment_pipeline(batch_texts, truncation=True, max_length=512)
    for res in results:
      # Map model labels (POSITIVE/NEGATIVE) to clean format
      label = res["label"].capitalize()
      score = res["score"]
      sentiments.append(label)
      confidences.append(score)

  df["sentiment"] = sentiments
  df["confidence"] = confidences

  # Save output
  OUTPUT_SENTIMENT.parent.mkdir(parents=True, exist_ok=True)
  df.to_csv(OUTPUT_SENTIMENT, index=False)

  print("\n--- Aspect Sentiment Analysis Complete ---")
  print(f"Saved aspect sentiment data to: {OUTPUT_SENTIMENT}")
  print(df[["aspect_name", "sentiment", "confidence"]].head(10))


if __name__ == "__main__":
  main()