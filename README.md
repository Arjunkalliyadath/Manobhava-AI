# Manobhava - AI

A FastAPI-based web application that analyzes a company website, discovers its products, collects customer feedback from multiple online platforms, runs sentiment analysis, and generates an interactive dashboard with downloadable reports.

https://github.com/user-attachments/assets/5a446b86-50bf-4410-a17f-a2a6eede409c

## Overview

This project helps turn raw product mentions and customer reviews into actionable insights. It supports:

- company and website metadata extraction
- product discovery from an e-commerce or brand website
- product selection for focused analysis
- review collection from Google, Twitter, Instagram, YouTube, Reddit, and website pages
- sentiment analysis using a RoBERTa-based Hugging Face model
- summary and report generation for product and brand-level understanding

## Key Features

- Discover products automatically from a provided website
- Let users select a small set of products for targeted analysis
- Collect review data from multiple sources in parallel
- Clean and deduplicate scraped comments before analysis
- Generate an interactive dashboard with insights and recommendations
- Export reports in PDF format for sharing and presentation

## Tech Stack

- Backend: Python, FastAPI
- Frontend: HTML, CSS, JavaScript, Jinja2 templates
- Scraping: Playwright, BeautifulSoup4, httpx
- Data handling: pandas, numpy
- NLP: transformers, torch, RoBERTa
- Reporting: reportlab

## Project Structure

```text
.
├── app.py                  # FastAPI app and main analysis pipeline
├── config.py               # Tunable scraper and analysis settings
├── company_discovery.py    # Website/company metadata extraction
├── product_discovery.py    # Product discovery logic
├── product_intelligence.py # Product-level intelligence generation
├── aspect_intelligence.py  # Aspect-level analysis logic
├── sentiment.py            # Sentiment analysis pipeline
├── utils.py                # Shared helper utilities
├── url_utils.py            # URL and company-name utilities
├── scrapers/               # Platform-specific scraper modules
├── templates/              # HTML templates for the UI
├── static/                 # Static assets for the web app
├── downloads/              # Generated reports and exports
└── requirements.txt
```

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Arjunkalliyadath/Website-Product-Analyzer.git
cd Website-Product-Analyzer
```

### 2. Create and activate a virtual environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Playwright browser dependencies

```bash
playwright install chromium
```

### 5. Run the application

```bash
python -m uvicorn app:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

> Note: The first run may download the sentiment model from Hugging Face, so it may take a few minutes to become fully responsive.

## Usage Flow

1. Open the app in your browser.
2. Enter a company website or brand URL.
3. Review the discovered products.
4. Select the products to analyze.
5. Wait for the scrapers and sentiment pipeline to complete.
6. View the dashboard and download the generated report.

## Notes

- The app is optimized for fast analysis and may limit the amount of data collected per platform to keep runs responsive.
- Some websites may block automated browsing, which can affect the amount of data returned by the scrapers.

## Author

**Arjun K**
- GitHub: [@Arjunkalliyadath](https://github.com/Arjunkalliyadath)
- Email: arjunkalliyadath2001@gmail.com
