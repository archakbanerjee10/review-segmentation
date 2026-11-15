
# ml/aspect_extraction.py

import pandas as pd
import re

from pathlib import Path
from aspect_config import ASPECTS, PURCHASE_INSIGHTS



# 1. PROJECT PATHS


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "artifacts" / "clustered_reviews.csv"

OUTPUT_DIR = PROJECT_ROOT / "artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "aspect_mentions.csv"
SUMMARY_FILE = OUTPUT_DIR / "aspect_extraction_summary.csv"
AUDIT_FILE = OUTPUT_DIR / "aspect_extraction_audit.csv"



# 2. LOAD DATA


print("Loading clustered reviews...")

df = pd.read_csv(INPUT_FILE)

required_columns = ["cleaned_text", "cluster"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")

if "rating" not in df.columns:
    df["rating"] = None

df = df.dropna(subset=["cleaned_text"]).copy()

df["cleaned_text"] = df["cleaned_text"].astype(str)

# Keep the original row position as a stable review reference.
df["review_id"] = df.index

print("Reviews loaded:", len(df))


# --------------------------------------------------
# 3. TEXT MATCHING FUNCTION
# --------------------------------------------------

def find_matching_keywords(text, keywords):

    """
    Finds keywords and phrases in a review.

    Matching is case-insensitive and uses word boundaries
    to avoid matching a term inside an unrelated word.
    """

    text = str(text).lower()

    matched_keywords = []

    for keyword in keywords:

        keyword_lower = keyword.lower().strip()

        # Allow flexible whitespace between words.
        pattern = re.escape(keyword_lower)
        pattern = pattern.replace(r"\ ", r"\s+")

        # Avoid matching inside unrelated words.
        pattern = r"(?<!\w)" + pattern + r"(?!\w)"

        if re.search(pattern, text):
            matched_keywords.append(keyword)

    return matched_keywords


# --------------------------------------------------
# 4. CLUSTER HINT FUNCTION
# --------------------------------------------------

def get_cluster_hint(cluster_id, source_clusters):

    """
    Checks whether the review's cluster was one of the
    clusters used to discover this aspect.

    This is supporting information, not proof that
    the aspect is present in the review.
    """

    if pd.isna(cluster_id):
        return False

    return int(cluster_id) in source_clusters


# --------------------------------------------------
# 5. EXTRACT PRODUCT ASPECTS
# --------------------------------------------------

print("\nExtracting product aspects...")

aspect_records = []

audit_records = []

for _, row in df.iterrows():

    review_text = row["cleaned_text"]
    review_id = row["review_id"]
    cluster_id = row["cluster"]
    rating = row["rating"]

    detected_labels = []

    # Check every product aspect.
    for aspect_id, aspect_info in ASPECTS.items():

        matched_keywords = find_matching_keywords(
            review_text,
            aspect_info["keywords"]
        )

        if not matched_keywords:
            continue

        cluster_hint = get_cluster_hint(
            cluster_id,
            aspect_info["source_clusters"]
        )

        detected_labels.append(aspect_id)

        aspect_records.append({

            "review_id": review_id,

            "review_text": review_text,

            "cluster": cluster_id,

            "rating": rating,

            "record_type": "product_aspect",

            "aspect_id": aspect_id,

            "aspect_name": aspect_info["name"],

            "matched_keywords": " | ".join(matched_keywords),

            "cluster_hint": cluster_hint

        })

    # Check purchasing-related insights separately.
    for insight_id, insight_info in PURCHASE_INSIGHTS.items():

        matched_keywords = find_matching_keywords(
            review_text,
            insight_info["keywords"]
        )

        if not matched_keywords:
            continue

        detected_labels.append(insight_id)

        aspect_records.append({

            "review_id": review_id,

            "review_text": review_text,

            "cluster": cluster_id,

            "rating": rating,

            "record_type": "purchase_insight",

            "aspect_id": insight_id,

            "aspect_name": insight_info["name"],

            "matched_keywords": " | ".join(matched_keywords),

            "cluster_hint": get_cluster_hint(
                cluster_id,
                insight_info["source_clusters"]
            )

        })

    # Save an audit record for every review,
    # including reviews with no detected aspect.
    audit_records.append({

        "review_id": review_id,

        "cluster": cluster_id,

        "detected_aspects": " | ".join(detected_labels),

        "number_of_detected_aspects": len(detected_labels),

        "has_detected_aspect": len(detected_labels) > 0

    })


# --------------------------------------------------
# 6. CREATE OUTPUT DATAFRAMES
# --------------------------------------------------

mentions_df = pd.DataFrame(aspect_records)

audit_df = pd.DataFrame(audit_records)


# --------------------------------------------------
# 7. SAVE ASPECT EXTRACTION RESULTS
# --------------------------------------------------

mentions_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

audit_df.to_csv(
    AUDIT_FILE,
    index=False,
    encoding="utf-8"
)


# --------------------------------------------------
# 8. GENERATE ASPECT SUMMARY
# --------------------------------------------------

if not mentions_df.empty:

    summary_df = (

        mentions_df

        .groupby(
            ["record_type", "aspect_id", "aspect_name"],
            as_index=False
        )

        .agg(

            review_count=("review_id", "nunique"),

            mention_count=("review_id", "count"),

            cluster_hint_count=("cluster_hint", "sum")

        )

        .sort_values(
            by="review_count",
            ascending=False
        )

    )

else:

    summary_df = pd.DataFrame(
        columns=[
            "record_type",
            "aspect_id",
            "aspect_name",
            "review_count",
            "mention_count",
            "cluster_hint_count"
        ]
    )


summary_df.to_csv(
    SUMMARY_FILE,
    index=False,
    encoding="utf-8"
)


# --------------------------------------------------
# 9. PRINT RESULTS
# --------------------------------------------------

print("\n" + "=" * 60)
print("ASPECT EXTRACTION COMPLETED")
print("=" * 60)

print("\nTotal reviews processed:", len(df))

print(
    "Reviews with detected aspects:",
    audit_df["has_detected_aspect"].sum()
)

print(
    "Reviews without detected aspects:",
    (~audit_df["has_detected_aspect"]).sum()
)

print(
    "Total extracted aspect records:",
    len(mentions_df)
)

print("\nAspect summary:")

if not summary_df.empty:
    print(summary_df.to_string(index=False))
else:
    print("No aspects detected.")

print("\nGenerated files:")

print("1. aspect_mentions.csv")
print("2. aspect_extraction_summary.csv")
print("3. aspect_extraction_audit.csv")

print("\nSaved in:", OUTPUT_DIR)
