from pathlib import Path
import pandas as pd

# Define project root and paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SENTIMENT_CSV = PROJECT_ROOT / "artifacts" / "aspect_sentiment.csv"
OUTPUT_SUMMARY = PROJECT_ROOT / "artifacts" / "aspect_summary.csv"


def main():
  print(f"Loading aspect sentiment data from: {SENTIMENT_CSV}")
  if not SENTIMENT_CSV.exists():
    raise FileNotFoundError(
        f"Aspect sentiment data not found at {SENTIMENT_CSV}. Run"
        " aspect_sentiment.py first."
    )

  df = pd.read_csv(SENTIMENT_CSV)
  print(f"Loaded {len(df)} sentiment records.")

  summary_records = []

  # Group by aspect_id and aspect_name
  grouped = df.groupby(["aspect_id", "aspect_name"])

  for (aspect_id, aspect_name), group in grouped:
    total_mentions = len(group)

    # Sentiment counts (handling label variations gracefully)
    positive_mask = group["sentiment"].str.lower() == "positive"
    negative_mask = group["sentiment"].str.lower() == "negative"

    positive_count = int(positive_mask.sum())
    negative_count = int(negative_mask.sum())
    neutral_count = total_mentions - (positive_count + negative_count)

    positive_pct = (
        (positive_count / total_mentions) * 100 if total_mentions > 0 else 0
    )
    negative_pct = (
        (negative_count / total_mentions) * 100 if total_mentions > 0 else 0
    )
    neutral_pct = (
        (neutral_count / total_mentions) * 100 if total_mentions > 0 else 0
    )

    avg_confidence = group["confidence"].mean()

    # Find representative positive and negative reviews (highest confidence ones)
    pos_subset = group[positive_mask].sort_values(
        by="confidence", ascending=False
    )
    neg_subset = group[negative_mask].sort_values(
        by="confidence", ascending=False
    )

    rep_positive = (
        pos_subset.iloc[0]["review_text"] if not pos_subset.empty else ""
    )
    rep_negative = (
        neg_subset.iloc[0]["review_text"] if not neg_subset.empty else ""
    )

    summary_records.append({
        "aspect_id": aspect_id,
        "aspect_name": aspect_name,
        "total_mentions": total_mentions,
        "positive_mentions": positive_count,
        "negative_mentions": negative_count,
        "neutral_mentions": neutral_count,
        "positive_percentage": round(positive_pct, 2),
        "negative_percentage": round(negative_pct, 2),
        "neutral_percentage": round(neutral_pct, 2),
        "avg_confidence": round(avg_confidence, 4),
        "representative_positive": rep_positive,
        "representative_negative": rep_negative,
    })

  summary_df = pd.DataFrame(summary_records)
  # Sort by total mentions descending
  summary_df = summary_df.sort_values(by="total_mentions", ascending=False)

  OUTPUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
  summary_df.to_csv(OUTPUT_SUMMARY, index=False)

  print("\n--- Aspect Aggregation Complete ---")
  print(f"Saved aspect summary to: {OUTPUT_SUMMARY}")
  print("\nTop aggregated aspects preview:")
  print(
      summary_df[[
          "aspect_name",
          "total_mentions",
          "positive_percentage",
          "negative_percentage",
      ]].head(10)
  )


if __name__ == "__main__":
  main()