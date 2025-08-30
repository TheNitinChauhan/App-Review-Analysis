# App Review Analysis Pipeline — README.md

**Auto-generated end-to-end pipeline for Google Play review intelligence**

Scrape → Clean → Sentiment → Topic Modeling → Embeddings → LLM-powered categorization & clustering → Charts → Auto PPT Report
This repository provides a robust end-to-end system that turns Google Play reviews into stakeholder-ready insights and a PowerPoint report. 
The pipeline combines classical NLP (VADER, TextBlob, LDA) with modern vector embeddings and LLMs (OpenAI) to produce package-specific findings and actionable recommendations.

-------
### Note on Embedding Step
The embedding step (Step 5) is useful for advanced semantic search and clustering.  
However, for some apps with very large or restricted review text (e.g., Instagram),  
the embedding step may fail or take very long.  

**Solution:**  
- If you face issues with embeddings, you can temporarily disable Step 5 in the code.  
- The pipeline will still run properly and generate sentiment analysis, word clouds, and charts without embeddings.  
-------


---


## Table of contents
1. [Project overview](#project-overview)
2. [Key features](#key-features)
3. [Architecture & flow](#architecture--flow)
4. [Quick demo (what you'll get)](#quick-demo-what-youll-get)
5. [Requirements & supported versions](#requirements--supported-versions)
6. [Installation](#installation)
7. [Configuration / Environment variables](#configuration--environment-variables)
8. [How to run (examples)](#how-to-run-examples)
9. [Outputs and file map and How findings & recommendations are generated (dynamic)](#outputs-and-file-map)
10. [Detailed step-by-step explanation of pipeline components](#detailed-step-by-step-explanation-of-pipeline-components)
11. [LLM usage, batching & cost-control tips](#llm-usage-batching--cost-control-tips)
12. [Troubleshooting — common errors & fixes ](#troubleshooting--common-errors--fixes)
13. [Productionization & extension ideas](#productionization--extension-ideas)
14. [Developer notes & contribution guide](#developer-notes--contribution-guide)
15. [License & contact](#license--contact)

---

## Project overview
This repository contains a single-file (prototype) Python pipeline that automates review intelligence for Google Play apps. Give the script a package name (e.g. `com.nextbillion.groww`) and it will:

- Scrape English reviews from Google Play (India locale by default)
- Clean and preprocess review text
- Run multi-method sentiment analysis (VADER + TextBlob)
- Extract topics with LDA
- Create embeddings using OpenAI `text-embedding-3-small` and cluster them (KMeans)
- Use **GPT-3.5 (gpt-3.5-turbo)** to auto-categorize reviews and name clusters
- Produce charts (pie, bar, Pareto, word clouds, time series)
- Auto-generate a branded PowerPoint report (`.pptx`) with dynamic recommendations

This is ideal for product managers, analysts, or engineers who want an automated, reproducible way to extract actionable insights from app reviews.

---

## Key features
- ✅ End-to-end automation: scraping → analysis → reporting
- ✅ Dual sentiment (VADER for short text, TextBlob for polarity summary)
- ✅ LDA topic modeling to surface common themes
- ✅ Semantic clustering with OpenAI embeddings + KMeans
- ✅ GPT-3.5 for human-friendly cluster names and automated issue categorization
- ✅ Dynamic recommendations generated from analysis outputs
- ✅ Exported charts + `python-pptx` report ready for stakeholders

---

## Architecture & flow
```text
[Google Play Scraper] -> [Cleaning] -> [Sentiment (VADER/TextBlob)]
 -> [LDA Topics] -> [OpenAI Embeddings] -> [KMeans Clustering]
 -> [GPT-3.5 cluster naming + categorization] -> [Charts / Wordclouds]
 -> [Dynamic Recommendation Generator] -> [PowerPoint Report]
```

---

## Quick demo (what you'll get)
After running the script for an app package you will find a folder named exactly as the package (e.g. `com.nextbillion.groww/`). Inside:
- CSVs with raw and cleaned reviews
- Excel files with categorized and clustered reviews
- PNG charts (sentiment pie, Pareto, word clouds, stacked bars, time-series)
- Final `{AppName}_Review_Analysis_Report.pptx` with insights & recommendations
- `execution_log.txt` with timing and any error traces

---

## Requirements & supported versions
- Python: **3.9 - 3.11** (recommended)
- OS: macOS / Linux / Windows. For headless Linux, set Matplotlib backend to `Agg`.

### Python packages (core)
- pandas
- numpy
- matplotlib
- wordcloud
- nltk
- textblob
- gensim
- scikit-learn
- google-play-scraper
- openai
- python-pptx
- python-dotenv

You can install quickly with:
```bash
pip install -r requirements.txt
```

If `requirements.txt` is not present, run:
```bash
pip install pandas numpy matplotlib wordcloud nltk textblob gensim scikit-learn google-play-scraper openai python-pptx python-dotenv
```

---

## Installation
1. Clone repository:
```bash
git clone <repo-url> && cd <repo-dir>
```
2. (Recommended) Create a virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```
3. Install dependencies (see previous section).
4. Create a `.env` or `new.env` file with your OpenAI API key (see next section).

---

## Configuration / Environment variables
Create a file named `new.env` in the project root (or set system env vars) with:
```env
OPENAI_API_KEY=sk-<your_openai_key>
```
Optional:
```env
OPENAI_API_BASE=https://api.openai.com  # only if using a custom endpoint
```

**Security note:** never commit your API key to source control. Use environment secrets for CI.

---

## How to run (examples)
Interactive run (default script prompts for package name):
```bash
python test2.py
# Enter package name when prompted, e.g.: com.nextbillion.groww
```

Suggested CLI wrapper (optional to implement):
```bash
python main.py --package com.nextbillion.groww --reviews 1000 --clusters 10
```

**Typical runtime:** ~3–6 minutes for 500–1000 reviews, depends on OpenAI API latency and batching.

---

## Outputs and file map
All outputs are saved inside the package-name folder created during run.

**CSV / Excel**
- `App_Reviews.csv` — raw scraped reviews
- `Cleaned_Reviews.csv` — cleaned text
- `Reviews_With_VADER_Sentiment.csv` — VADER labels
- `Reviews_With_Sentiment.csv` — TextBlob labels
- `Auto_Categorized_Reviews.xlsx` — GPT categories per review
- `Clustered_Reviews.xlsx` — GPT cluster names per review
- `Reviews_with_Embedding_Clusters.xlsx` — KMeans embedding clusters
- `Top_20_Cluster_Stats.xlsx` — top clusters + sample reviews
- `LDA_Topics.txt` — LDA topic keywords

**Images (PNG)**
- `time_series_average_score.png`
- `pareto_cluster_analysis_top20.png`
- `sentiment_pie.png`
- `wordcloud_positive.png`, `wordcloud_negative.png`, `wordcloud_neutral.png`
- `sentiment_bar_chart.png` / `top_20_clusters_bar_chart.png`
- `auto_category_bar_chart.png` / `auto_category_sentiment_stacked_bar.png`

**Report**
- `{AppName}_Review_Analysis_Report.pptx` — full stakeholder-ready PPT

**Log**
- `execution_log.txt` — timestamps, durations, tracebacks if errors occur

**How findings & recommendations are generated (dynamic)**

The pipeline creates data-driven findings and recommendations using:

Sentiment Summary (TextBlob polarity distribution)

LDA Topics (top words per topic)

Top Clusters from GPT & KMeans

Algorithmic flow:

Compute metrics (positive/negative/neutral %, top clusters, LDA topics).

Detect keywords/flags (e.g., login, payment, fraud, fees, support, performance).

Compose 5 detailed Key Findings (multi-line) reflecting the actual metrics and recent sentiment trend.

Compose 5 practical Recommendations aligned to those findings (triage sprint, UX fixes, payment improvements, support SLAs, roadmap transparency).

Because inputs come directly from analysis artifacts, the findings and recommendations change for every package.
---

## Detailed step-by-step explanation of pipeline components
This expands each step and why it exists.

### 1. Scraping
- `google_play_scraper.reviews(package, lang='en', country='in', sort=Sort.NEWEST, count=1000)`
- Saves raw JSON to `App_Reviews.csv` for reproducibility.

### 2. Preprocessing
- `basic_clean()` removes URLs, emojis/non-ASCII chars, punctuation and extra whitespace.
- `advanced_clean()` lowers, tokenizes, removes NLTK stopwords, and lemmatizes using WordNet.
- Drop duplicates and short reviews.

### 3. Sentiment
- **VADER:** best for short social-like text (uses lexicons). Compound thresholds: >=0.05 Positive, <=-0.05 Negative.
- **TextBlob:** polarity-based; used as a second signal for overall positive/negative split.
- Save both signals — they complement each other.

### 4. LDA topic modeling
- `gensim.corpora.Dictionary` + `LdaModel(corpus, num_topics=5, passes=10)`
- Output `LDA_Topics.txt` with topic-word distributions for human inspection.

### 5. Embeddings & semantic clustering
- Use OpenAI Embeddings (`text-embedding-3-small`) to convert reviews to vectors.
- Cluster vectors using **KMeans** (k default = 10). Save cluster ids to the dataframe.
- For each cluster, sample up to N reviews and ask **GPT-3.5** to provide a short cluster name.

### 6. GPT-powered auto-categorization
- Batch reviews and call `gpt-3.5-turbo` to assign short issue categories per review (e.g., "Login Issue").
- Save outputs into `Auto_Categorized_Reviews.xlsx`
- **Note:** batching size is crucial — large batches reduce number of API calls but increase token usage per request.

### 7. Charts and visualizations
- WordClouds: show dominant tokens per sentiment bucket.
- Pareto chart: identify the few clusters that account for most complaints.
- Stacked bar: Auto-category vs Sentiment.
- Time-series plot: average rating over time (requires `at` timestamp from scraper).

### 8. Dynamic recommendation generation
- A small rule-based function inspects:
  - sentiment percentages
  - LDA topic keywords
  - top GPT-derived cluster names
- Produces a prioritized list of up to 5 recommendations tailored to the app.
- Optionally attach sample review snippets as evidence for each recommendation.

### 9. PowerPoint report
- `python-pptx` used to build slides: title, methodology, charts, recommended actions, and sample review evidence.
- Add defensive checks: ensure each image exists before inserting to avoid crashes.

---

## LLM usage, batching & cost-control tips
**Models used in the pipeline:**
- Embeddings: `text-embedding-3-small` (OpenAI)
- LLM / Chat: `gpt-3.5-turbo` (OpenAI)

**Best practices**
- **Increase batch size** for GPT categorization/clustering (try 8–16) to reduce the number of API calls.
- **Cache** embeddings / GPT outputs locally (CSV/DB) to avoid re-calling for the same app.
- **Use retry/backoff** for `RateLimitError` and transient API errors (`tenacity` library helps).
- **Monitor costs** and sampling: if you process 1,000 reviews and send small batches of 2, you'll make ~500 calls — expensive. Larger batches reduce count.

**Security & privacy**
- Remove or mask personally identifiable information (PII) from reviews if you plan to share or publish the outputs.

---

## Troubleshooting — common errors & fixes
**1. NameError: `sentiment_summary` is not defined**
- Ensure you compute `sentiment_summary` from `df['Sentiment']` before calling recommendation generator.

**2. `.tolist` vs `.tolist()`**
- Always call `.tolist()` — forgetting parentheses returns a method object.

**3. PPTX crashes due to missing images**
- Before calling `add_image_with_description_slide`, check `os.path.exists(img_path)` and add a fallback slide if missing.

**4. OpenAI RateLimitError / API errors**
- Implement retry with exponential backoff; increase batch size; add usage logging.

**5. Matplotlib on headless servers**
- Use `matplotlib.use('Agg')` before importing `pyplot`.

**6. Very long prompts / token limits**
- If a batch prompt is too long, reduce batch size or truncate reviews (while preserving meaning).

---

## Productionization & extension ideas
- **Modularize** into `scraper.py`, `preprocess.py`, `analysis.py`, `report.py` and create `main.py` orchestrator.
- **Database persistence** (SQLite/Postgres) for incremental updates and faster re-runs.
- **Containerize** with Docker and add a `Dockerfile` for reproducible runs.
- **Switch clustering** to `UMAP + HDBSCAN` for variable cluster sizes and better separation.
- **Deploy a small web dashboard** (Streamlit/Flask) for interactive exploration of reviews and clusters.
- **Add CI/CD** tests for critical functions (cleaning, LDA output format, embedding shapes, PPT creation).

---

## Developer notes & contribution guide
- Keep code PEP8-compliant. Use docstrings for public functions.
- Replace `print()` with `logging` for production.
- Add `config.yaml` or `argparse` to make parameters (batch size, number of clusters, num_topics) configurable.
- When contributing, open a PR with a clear description and include tests for new features.

---

## License & contact
This project does not include a license by default — add one based on your preference. Example (MIT):

```text
MIT License
Copyright (c) <year> <owner>
Permission is hereby granted, free of charge, to any person obtaining a copy...
```

**Contact / author**: Nitin Chauhan (or update to your name/email in the file header).

---

## Appendix: Helpful snippets
### Set Matplotlib headless backend
```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
```

### Example retry for OpenAI calls (tenacity)
```python
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(5))
def call_openai(...):
    return openai.ChatCompletion.create(...)
```

---

*Generated by App Review Analysis Pipeline helper — if you want I can also produce `requirements.txt`, `.env.example`, and a small `run.sh` or `main.py` wrapper.*

