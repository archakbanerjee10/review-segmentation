from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Customer Review Intelligence Platform",
    page_icon="📊",
    layout="wide",
)

# Define paths
PROJECT_ROOT = Path(__file__).resolve().parent
SUMMARY_CSV = PROJECT_ROOT / "artifacts" / "aspect_summary.csv"
SENTIMENT_CSV = PROJECT_ROOT / "artifacts" / "aspect_sentiment.csv"
CLUSTERS_CSV = PROJECT_ROOT / "artifacts" / "clustered_reviews.csv"
INSIGHTS_TXT = PROJECT_ROOT / "artifacts" / "llm_product_insights.txt"


@st.cache_data
def load_data():
  summary_df = (
      pd.read_csv(SUMMARY_CSV) if SUMMARY_CSV.exists() else pd.DataFrame()
  )
  sentiment_df = (
      pd.read_csv(SENTIMENT_CSV) if SENTIMENT_CSV.exists() else pd.DataFrame()
  )
  clustered_df = (
      pd.read_csv(CLUSTERS_CSV) if CLUSTERS_CSV.exists() else pd.DataFrame()
  )
  return summary_df, sentiment_df, clustered_df


summary_df, sentiment_df, clustered_df = load_data()

# App Header
st.title("🛍️ Customer Review Intelligence & Aspect Clustering Platform")
st.markdown(
    "An advanced machine learning system for aspect-based sentiment analysis,"
    " semantic clustering, and LLM-powered customer insights."
)
st.markdown("---")

# Sidebar Navigation
st.sidebar.header("Navigation & Filters")
page = st.sidebar.radio(
    "Select View",
    [
        "📊 Executive Overview",
        "🔍 Aspect-Level Analysis",
        "📝 Review Explorer & Quotes",
        "🤖 LLM Insights Report",
    ],
)

if page == "📊 Executive Overview":
  st.header("Executive Product Overview")

  if summary_df.empty:
    st.warning(
        "Aspect summary data not found. Please run your ML pipeline scripts"
        " first."
    )
  else:
    # Top metrics row
    total_reviews = (
        len(clustered_df) if not clustered_df.empty else "N/A"
    )
    total_aspects = len(summary_df)
    total_mentions = summary_df["total_mentions"].sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reviews Analysed", total_reviews)
    col2.metric("Aspect Categories", total_aspects)
    col3.metric("Total Aspect Mentions", total_mentions)
    col4.metric(
        "Clusters Discovered",
        (
            clustered_df["cluster"].nunique()
            if not clustered_df.empty
            else "N/A"
        ),
    )

    st.markdown("### Aspect Mentions & Sentiment Breakdown")

    # Plotly bar chart for aspect mentions and sentiment
    fig = px.bar(
        summary_df,
        x="aspect_name",
        y=["positive_mentions", "negative_mentions", "neutral_mentions"],
        title="Mention Volume and Sentiment Split by Product Aspect",
        labels={"value": "Mention Count", "aspect_name": "Product Aspect"},
        barmode="stack",
        color_discrete_map={
            "positive_mentions": "#2ecc71",
            "negative_mentions": "#e74c3c",
            "neutral_mentions": "#95a5a6",
        },
    )
    fig.update_layout(xaxis_tickangle=-45, legend_title="Sentiment")
    st.plotly_chart(fig, use_container_width=True)

elif page == "🔍 Aspect-Level Analysis":
  st.header("Detailed Aspect Sentiment Analysis")

  if summary_df.empty:
    st.warning("No summary data available.")
  else:
    selected_aspect = st.selectbox(
        "Choose an Aspect to Inspect", summary_df["aspect_name"].tolist()
    )
    aspect_row = summary_df[
        summary_df["aspect_name"] == selected_aspect
    ].iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Mentions", int(aspect_row["total_mentions"]))
    col2.metric("Positive Sentiment", f"{aspect_row['positive_percentage']}%")
    col3.metric("Negative Sentiment", f"{aspect_row['negative_percentage']}%")

    st.markdown("### Representative Customer Quotes")

    c1, c2 = st.columns(2)
    with c1:
      st.success("🟢 **Representative Positive Quote**")
      st.info(
          aspect_row["representative_positive"]
          if pd.notna(aspect_row["representative_positive"])
          else "No positive review captured."
      )

    with c2:
      st.error("🔴 **Representative Negative Quote**")
      st.warning(
          aspect_row["representative_negative"]
          if pd.notna(aspect_row["representative_negative"])
          else "No negative review captured."
      )

elif page == "📝 Review Explorer & Quotes":
  st.header("Raw Review Explorer")

  if clustered_df.empty:
    st.warning("Clustered reviews dataset not found.")
  else:
    search_term = st.text_input(
        "Search reviews by keyword:", ""
    )
    filtered_df = clustered_df
    if search_term:
      filtered_df = clustered_df[
          clustered_df["cleaned_text"]
          .str.contains(search_term, case=False, na=False)
      ]

    st.write(
        f"Showing {len(filtered_df)} reviews (filtered from"
        f" {len(clustered_df)} total):"
    )
    st.dataframe(
        filtered_df[[
            col
            for col in [
                "review_id",
                "cluster",
                "rating",
                "cleaned_text",
            ]
            if col in filtered_df.columns
        ]],
        use_container_width=True,
    )

elif page == "🤖 LLM Insights Report":
  st.header("🤖 LLM-Powered Customer Intelligence Report")

  if INSIGHTS_TXT.exists():
    with open(INSIGHTS_TXT, "r", encoding="utf-8") as f:
      report_content = f.read()
    st.markdown(report_content)
  else:
    st.warning(
        "LLM insights report not found. Run `python llm/llm_insights.py` first"
        " or ensure your API key was configured."
    )