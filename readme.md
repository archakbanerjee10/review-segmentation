# Customer Review Intelligence & Aspect Clustering Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Groq API](https://img.shields.io/badge/LLM-Groq%20Cloud-orange.svg)](https://groq.com/)

An end-to-end, CPU-optimized Machine Learning and Generative AI system that ingests raw, unstructured customer reviews from e-commerce platforms and converts them into structured, aspect-level product intelligence, semantic topic clusters, and executive business summaries[span_0](start_span)[span_0](end_span).

Designed specifically to operate efficiently on standard consumer-grade hardware without requiring dedicated GPU accelerators, this platform bridges the gap between high-level customer feedback and granular engineering actionable items[span_1](start_span)[span_1](end_span).

---

## Table of Contents
- [Overview & Problem Statement](#overview--problem-statement)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [End-to-End Execution Pipeline](#end-to-end-execution-pipeline)
- [Interactive Dashboard Features](#interactive-dashboard-features)

---

## Overview & Problem Statement

Traditional e-commerce feedback mechanisms fail to provide actionable clarity at scale due to three major bottlenecks[span_2](start_span)[span_2](end_span):
1. **The Opacity of Aggregate Star Ratings:** A 3-star rating indicates general customer hesitation but flattens multidimensional feedback into a single numeric value, concealing whether a customer praised visual design and sound quality while criticizing battery longevity and Bluetooth pairing reliability[span_3](start_span)[span_3](end_span).
2. **Scalability Limits of Manual Audit:** Popular consumer products frequently accumulate thousands of text reviews monthly, making manual qualitative coding labor-intensive, expensive, inconsistent, and unscalable across large product catalogs[span_4](start_span)[span_4](end_span).
3. **Surface-Level Sentiment vs. Fine-Grained Aspect Intelligence:** Standard global sentiment analysis classifies an entire review as positive, negative, or neutral, failing to isolate specific sub-system failures[span_5](start_span)[span_5](end_span).

**Our Hybrid Approach:**
> **Input:** *"The display is crisp, but the hinge broke after two days of light use."*[span_6](start_span)[span_6](end_span)  
> - **Traditional Global Sentiment:** `NEUTRAL` (50/50 score) — Low Actionability[span_7](start_span)[span_7](end_span)  
> - **Aspect: Display** $\rightarrow$ `POSITIVE` (Confidence: 0.94)[span_8](start_span)[span_8](end_span)  
> - **Aspect: Build/Hinge** $\rightarrow$ `NEGATIVE` (Confidence: 0.98)[span_9](start_span)[span_9](end_span)

By combining supervised rule/lexicon-based Aspect-Based Sentiment Analysis (ABSA), unsupervised vector embeddings with density clustering (`UMAP` + `HDBSCAN`), and zero-shot LLM reasoning for executive synthesis, the system automatically exposes exact product strengths, design defects, and emerging failure modes[span_10](start_span)[span_10](end_span)[span_11](start_span)[span_11](end_span).

---

## System Architecture

```text
                    ┌──────────────────────────────┐
                    │  Raw Amazon Review Dataset   │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ Filtering & Text Pre-Process │
                    └──────────────┬───────────────┘
                                   │
                 ┌─────────────────┴─────────────────┐
                 ▼                                   ▼
┌─────────────────────────────────┐ ┌─────────────────────────────────┐
│     Semantic Vector Space       │ │     Granular Aspect Engine      │
│ (all-MiniLM-L6-v2 Embeddings)   │ │   (Aspect Config Taxonomy)      │
└────────────────┬────────────────┘ └────────────────┬────────────────┘
                 │                                   │
                 ▼                                   ▼
┌─────────────────────────────────┐ ┌─────────────────────────────────┐
│   Dimensionality & Density      │ │   Aspect & Sentence Mapping     │
│       (UMAP + HDBSCAN)          │ │      + Sentiment Scoring        │
└────────────────┬────────────────┘ └────────────────┬────────────────┘
                 │                                   │
                 └─────────────────┬─────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │   LLM Executive Synthesis    │
                    │         (Groq API)           │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ Streamlit Interactive App    │
                    └──────────────────────────────┘

Dataset & Scope
 * Data Source: Amazon Reviews 2023 Dataset (Electronics Category).
 * Target Product Focus: Single-product deep-dive mode focusing on ASIN B01G8J05F2 (Wireless Audio Product) to demonstrate granular sub-system analysis.
Technology Stack
| Component | Technology / Library | Selection Rationale & Operational Parameters |
|---|---|---|
| Language | Python 3.10+ | Standard ecosystem for machine learning, data engineering, and stream manipulation. |
| Data Processing | Pandas, NumPy | High-performance in-memory tabular data manipulation and array vectorization. |
| Sentence Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Generates dense 384-dimensional vector space representations optimized for CPU inference speed (5x faster than base Transformer models while retaining 99% semantic accuracy). |
| Dimensionality Reduction | UMAP (umap-learn) | Uniform Manifold Approximation and Projection; reduces 384D embeddings to 2D for high-speed clustering while preserving local/global semantic relationships better than t-SNE or PCA. |
| Unsupervised Clustering | HDBSCAN (hdbscan) | Hierarchical Density-Based Spatial Clustering of Applications with Noise; identifies natural topic clusters of arbitrary shape without forcing soft points or requiring pre-specified cluster counts (k). |
| Lexicon & Aspect NLP | VADER / Custom Rule Patterns | Provides ultra-fast, deterministic sentiment scoring and aspect tagging per sentence without GPU requirements. |
| LLM Inference Engine | OpenAI Python SDK (via Groq Cloud API) | Connects to https://api.groq.com/openai/v1 accessing Llama-3/Mixtral architectures for sub-second executive report generation at zero-to-minimal cost. |
| Dashboard & UI | Streamlit, Plotly Express | Pure Python interactive web app framework paired with declarative, responsive data visualizers. |
| Environment Control | python-dotenv, standard venv | Encapsulates API keys securely and isolates system dependencies across environments. |
Repository Structure
review-segmentation/
├── artifacts/
│   ├── aspect_summary.csv          # Aggregated aspect sentiment metrics & representative quotes[span_33](start_span)[span_33](end_span)[span_34](start_span)[span_34](end_span)
│   ├── processed_reviews.csv       # Processed review dataset[span_35](start_span)[span_35](end_span)
│   └── llm_product_insights.txt    # Narrative executive summary generated by LLM[span_36](start_span)[span_36](end_span)[span_37](start_span)[span_37](end_span)
├── datasets/
│   └── raw_reviews_electronics.json # Raw Amazon electronics review records[span_38](start_span)[span_38](end_span)[span_39](start_span)[span_39](end_span)
├── llm/
│   ├── llm_models.py               # OpenAI API client configured with Groq base URL[span_40](start_span)[span_40](end_span)[span_41](start_span)[span_41](end_span)
│   └── llm_insights.py             # Formats aspect metrics & generates executive summary[span_42](start_span)[span_42](end_span)[span_43](start_span)[span_43](end_span)
├── ml/
│   ├── aspect_aggregation.py       # Computes product-level positive/negative ratios & extracts quotes[span_44](start_span)[span_44](end_span)[span_45](start_span)[span_45](end_span)
│   ├── aspect_config.py            # Central domain taxonomy, regex patterns, synonyms & anchor terms[span_46](start_span)[span_46](end_span)[span_47](start_span)[span_47](end_span)
│   ├── aspect_extraction.py        # Evaluates sentence-segmented text against aspect taxonomy[span_48](start_span)[span_48](end_span)[span_49](start_span)[span_49](end_span)
│   ├── aspect_sentiment.py         # Runs granular VADER sentiment scoring across aspect sentences[span_50](start_span)[span_50](end_span)[span_51](start_span)[span_51](end_span)
│   ├── cluster_analysis.py         # TF-IDF keyword & representative review extraction per cluster[span_52](start_span)[span_52](end_span)[span_53](start_span)[span_53](end_span)
│   ├── clustering.py               # UMAP dimensionality reduction & HDBSCAN density clustering[span_54](start_span)[span_54](end_span)[span_55](start_span)[span_55](end_span)
│   ├── embeddings.py               # Batch-encodes cleaned text into 384D vectors via all-MiniLM-L6-v2[span_56](start_span)[span_56](end_span)[span_57](start_span)[span_57](end_span)
│   ├── filter_english.py           # Language detection & invalid/corrupt string filtering[span_58](start_span)[span_58](end_span)
│   └── pre_processing.py           # HTML stripping, contraction expansion & sentence segmentation[span_59](start_span)[span_59](end_span)
├── notebook/
│   └── exploratory_analysis.ipynb  # Exploratory data analysis notebook[span_60](start_span)[span_60](end_span)
├── python/
│   └── load_dataset.py             # Raw JSON/CSV ingestion & target ASIN filtering[span_61](start_span)[span_61](end_span)
├── .env                            # Local environment variables & API keys[span_62](start_span)[span_62](end_span)[span_63](start_span)[span_63](end_span)
├── app.py                          # Single-entry Streamlit interactive dashboard[span_64](start_span)[span_64](end_span)[span_65](start_span)[span_65](end_span)
├── readme.md                       # Project documentation[span_66](start_span)[span_66](end_span)
└── requirements.txt                # Python package dependencies[span_67](start_span)[span_67](end_span)[span_68](start_span)[span_68](end_span)

Getting Started
1. Clone the Repository
git clone [https://github.com/archakbanerjee10/review-segmentation.git](https://github.com/archakbanerjee10/review-segmentation.git)
cd review-segmentation

2. Environment Initialization & Dependency Setup
Open PowerShell or Terminal in the project root directory and set up a clean Python virtual environment:
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate virtual environment (macOS/Linux)
# source venv/bin/activate

# Upgrade pip and install required dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```[span_70](start_span)[span_70](end_span)

### 3. Configure Environment Variables
Create a `.env` file in the project root directory to configure the Groq API key[span_71](start_span)[span_71](end_span):

```env
GROQ_API_KEY=your_actual_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```[span_72](start_span)[span_72](end_span)

---

## End-to-End Execution Pipeline

Run the pipeline scripts in sequence[span_73](start_span)[span_73](end_span). Each module saves its outputs to the `artifacts/` folder for downstream processing[span_74](start_span)[span_74](end_span).

### Step 1: Ingestion & Dataset Filtering
Extract the raw reviews and filter for the targeted ASIN (`B01G8J05F2`)[span_75](start_span)[span_75](end_span)[span_76](start_span)[span_76](end_span):
```bash
python python/load_dataset.py
```[span_77](start_span)[span_77](end_span)

### Step 2: Text Processing & Language Cleaning
Remove non-English text and sanitize review content[span_78](start_span)[span_78](end_span):
```bash
python ml/filter_english.py
python ml/pre_processing.py
```[span_79](start_span)[span_79](end_span)

### Step 3: Semantic Embedding Generation & HDBSCAN Clustering
Compute dense 384D vector representations, reduce dimensions via UMAP, and discover semantic clusters using HDBSCAN[span_80](start_span)[span_80](end_span)[span_81](start_span)[span_81](end_span):
```bash
python ml/embeddings.py
python ml/clustering.py
python ml/cluster_analysis.py
```[span_82](start_span)[span_82](end_span)

### Step 4: Aspect Extraction & Granular Sentiment Scoring
Execute aspect mapping (e.g., *Battery Life, Build Quality, Sound/Performance, Price/Value, Connectivity, Customer Support*) and score sentiment at the sentence level[span_83](start_span)[span_83](end_span)[span_84](start_span)[span_84](end_span):
```bash
python ml/aspect_extraction.py
python ml/aspect_sentiment.py
python ml/aspect_aggregation.py
```[span_85](start_span)[span_85](end_span)

### Step 5: Generative LLM Insight Synthesis
Pass aggregated aspect metrics to the Groq API to generate an executive intelligence report[span_86](start_span)[span_86](end_span):
```bash
python llm/llm_insights.py
```[span_87](start_span)[span_87](end_span)

### Step 6: Launch the Web Dashboard
Launch the interactive Streamlit presentation application[span_88](start_span)[span_88](end_span):
```bash
streamlit run app.py
```[span_89](start_span)[span_89](end_span)

---

## Interactive Dashboard Features

The interactive Streamlit dashboard (`app.py`) is structured into four functional modules accessible via the left sidebar navigation[span_90](start_span)[span_90](end_span):

1. **Executive Overview:** Presents overall review counts, global sentiment metrics, net sentiment scores, and interactive Plotly distributions of review volume categorized by aspect taxonomy[span_91](start_span)[span_91](end_span)[span_92](start_span)[span_92](end_span).
2. **Aspect-Level Analysis:** Dedicated deep-dive view allowing users to select specific product dimensions (e.g., *Battery Life* vs. *Connectivity*) to analyze isolated sentiment splits, distribution trends, and semantic vector clusters[span_93](start_span)[span_93](end_span)[span_94](start_span)[span_94](end_span).
3. **Review Explorer & Direct Quotes:** Interactive tabular data interface equipped with text search, rating sliders, and aspect filters, displaying verified positive and negative customer quotes mapped directly to specific aspect failures or praise points[span_95](start_span)[span_95](end_span)[span_96](start_span)[span_96](end_span).
4. **LLM Insights Report:** Displays the generated executive summary produced by the LLM pipeline (`llm/llm_insights.py`), offering plain-language synthesized business conclusions, emerging hardware defect flags, and prioritized engineering action items[span_97](start_span)[span_97](end_span)[span_98](start_span)[span_98](end_span).