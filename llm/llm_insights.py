from pathlib import Path
import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define project root and paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_CSV = PROJECT_ROOT / "artifacts" / "aspect_summary.csv"
OUTPUT_INSIGHTS = PROJECT_ROOT / "artifacts" / "llm_product_insights.txt"


def generate_insights_summary(summary_df):
  aspect_text_block = ""
  for _, row in summary_df.iterrows():
    aspect_text_block += (
        f"- Aspect: {row['aspect_name']} (Mentions: {row['total_mentions']}, "
        f"Positive: {row['positive_percentage']}%, Negative:"
        f" {row['negative_percentage']}%)\n"
    )

  prompt = f"""
You are an expert product intelligence and customer experience analyst. 
Analyze the following aggregated aspect-based sentiment statistics extracted from Amazon customer reviews for an electronics product:

{aspect_text_block}

Based on this structured data, provide a professional customer review intelligence report covering:
1. **Top Product Strengths**: What customers love the most, backed by data.
2. **Key Product Weaknesses & Complaints**: The most prominent pain points and negative drivers.
3. **Most Frequently Discussed Issues**: What drives conversation volume.
4. **Overall Product Verdict & Recommendation**: Would customers recommend this, and what is the summary outlook?

Keep your insights grounded strictly in the provided data. Do not hallucinate external facts.
"""
  return prompt


def main():
  print(f"Loading aspect summary from: {SUMMARY_CSV}")
  if not SUMMARY_CSV.exists():
    raise FileNotFoundError(
        f"Aspect summary not found at {SUMMARY_CSV}. Run aspect_aggregation.py"
        " first."
    )

  df = pd.read_csv(SUMMARY_CSV)
  prompt = generate_insights_summary(df)

  # Explicitly look for GROQ_API_KEY
  api_key = os.environ.get("GROQ_API_KEY")
  if not api_key:
    print(
        "\n[ERROR] GROQ_API_KEY not found! Make sure it is saved inside your .env"
        " file as GROQ_API_KEY=your_key"
    )
    return

  print("Calling Groq API for customer insights generation...")
  client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

  response = client.chat.completions.create(
      model="openai/gpt-oss-20b",  # Active Groq production model
      messages=[
          {
              "role": "system",
              "content": (
                  "You are an expert product review intelligence assistant."
              ),
          },
          {"role": "user", "content": prompt},
      ],
      temperature=0.3,
  )

  insights_text = response.choices[0].message.content

  OUTPUT_INSIGHTS.parent.mkdir(parents=True, exist_ok=True)
  with open(OUTPUT_INSIGHTS, "w", encoding="utf-8") as f:
    f.write(insights_text)

  print("\n--- LLM Insights Generation Complete ---")
  print(f"Saved insights to: {OUTPUT_INSIGHTS}")


if __name__ == "__main__":
  main()