# 🌐 Website Product Analyzer

An AI-powered web application that automatically discovers products from a company website, collects customer feedback from multiple online platforms, performs sentiment analysis using **RoBERTa**, and generates an interactive dashboard with downloadable reports.

---

# 🚀 Features

## 🌍 Website Metadata Extraction

Automatically extracts:

- Company Name
- Logo
- Website URL
- Twitter (X)
- Instagram
- YouTube
- Facebook
- LinkedIn

---

## 📦 Product Discovery

- Crawls the provided website
- Automatically discovers available products
- Supports large e-commerce websites
- Identifies hundreds of products from a single website

Example:

```
Website

↓

234 Products Found
```

---

## ✅ Product Selection

Instead of analyzing every product, users can:

- Search products
- Select only the required products
- Analyze up to **5 selected products**

This significantly reduces execution time while allowing focused analysis.

---

## 🌐 Multi-Platform Review Collection

Collects customer reviews from:

- ⭐ Google Reviews
- 🐦 Twitter (X)
- 📸 Instagram
- ▶️ YouTube

The system is fault tolerant and continues execution even if one platform times out or returns no data.

---

## 🤖 AI Sentiment Analysis

Uses the Hugging Face model

```
cardiffnlp/twitter-roberta-base-sentiment-latest
```

Features:

- Positive / Neutral / Negative classification
- Batch inference
- Duplicate removal
- Text preprocessing
- Keyword-based fallback

---

## 📊 Interactive Dashboard

Provides:

- Brand Score
- Sentiment Distribution
- Product-wise Analysis
- Platform-wise Analysis
- Executive Summary
- Key Insights
- Recommendations
- Most Discussed Product
- Positive & Negative Highlights

---

## 📄 Export Options

Generate downloadable reports in:

- PDF

---

# 🏗 Architecture

```
Website URL
      │
      ▼
Website Metadata Extraction
      │
      ▼
Product Discovery
      │
      ▼
User Product Selection
      │
      ▼
Review Collection
      │
      ├── Google Reviews
      ├── Twitter
      ├── Instagram
      └── YouTube
      │
      ▼
Data Cleaning
      │
      ▼
RoBERTa Sentiment Analysis
      │
      ▼
Analytics Engine
      │
      ▼
Dashboard
      │
      ▼
CSV / Excel / JSON / PDF
```

---

# 🔄 Workflow

```
User enters Website URL

        │

        ▼

Extract Website Metadata

        │

        ▼

Discover Products

        │

        ▼

Select Products

        │

        ▼

Collect Reviews

        │

        ▼

Clean & Preprocess Data

        │

        ▼

RoBERTa Sentiment Analysis

        │

        ▼

Generate Dashboard

        │

        ▼

Download Reports
```

---

# ⚙ Installation

Clone the repository

```bash
git clone https://github.com/Arjunkalliyadath/Website-Product-Analyzer.git
```

Navigate to the project

```bash
cd Website-Product-Analyzer
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the virtual environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python -m uvicorn app:app --reload
```

Open your browser

```
http://127.0.0.1:8000
```

---

# 🛠 Tech Stack

### Backend

- Python
- FastAPI

### Frontend

- HTML
- CSS
- JavaScript
- Jinja2

### Machine Learning

- Hugging Face Transformers
- RoBERTa
- PyTorch

### Data Processing

- Pandas
- NumPy

### Web Scraping

- Playwright
- BeautifulSoup
- HTTPX

### Reports

- ReportLab
- CSV
- Excel
- JSON

---

# 📂 Project Structure

```
Website-Product-Analyzer/

│
├── app.py
├── config.py
├── company_discovery.py
├── product_discovery.py
├── sentiment.py
├── utils.py
│
├── scrapers/
│   ├── google_scraper.py
│   ├── twitter_scraper.py
│   ├── instagram_scraper.py
│   └── youtube_scraper.py
│
├── templates/
│   ├── index.html
│   ├── dashboard.html
│   └── select_products.html
│
├── static/
├── downloads/
├── requirements.txt
└── README.md
```

---


# 👨‍💻 Author

**Arjun K**

AI/ML Engineer • Data Science Enthusiast

GitHub: https://github.com/Arjunkalliyadath

---

⭐ If you found this project useful, consider giving it a star on GitHub.
