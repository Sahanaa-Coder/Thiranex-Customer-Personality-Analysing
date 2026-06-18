# 🧠 Customer Personality Analysis
### Data Cleaning · Exploratory Data Analysis · Business Insights · Interactive Dashboard

> **CodeAlpha Data Science Internship — Portfolio Project**  
> Built by **Sahanaa** · Python · Pandas · Seaborn · Plotly Dash

---

## 📌 Project Overview

This end-to-end data science project analyses the purchasing behaviour and demographic profiles of **2,240 retail customers**. Starting from a raw, messy marketing CSV, the pipeline cleans the data, derives rich features, generates 12+ professional visualisations, surfaces 10 actionable business insights, and presents everything in an **interactive real-time dashboard**.

---

## 🎯 Objectives

| # | Objective |
|---|-----------|
| 1 | Apply a production-grade data-cleaning pipeline |
| 2 | Perform exploratory data analysis (EDA) |
| 3 | Generate publication-quality visualisations |
| 4 | Derive 10+ business-actionable insights |
| 5 | Build an interactive, filterable Plotly Dash dashboard |
| 6 | Export an automated data quality report |

---

## 📂 Project Structure

```
Data-Cleaning-Visualization/
│
├── data/
│   ├── raw_dataset.csv           ← Original marketing campaign data
│   └── cleaned_dataset.csv       ← Output of cleaning pipeline
│
├── notebooks/
│   └── data_analysis.ipynb       ← Jupyter walkthrough (optional)
│
├── dashboard/
│   └── dashboard.py              ← Interactive Plotly Dash app
│
├── visualizations/               ← All saved PNG charts
│   ├── histogram.png
│   ├── boxplot.png
│   ├── scatter_income_vs_spend.png
│   ├── heatmap.png
│   ├── countplot_categories.png
│   ├── pairplot.png
│   ├── bar_spend_by_education.png
│   ├── pie_marital_status.png
│   ├── violin_income_by_marital.png
│   ├── bar_campaign_response.png
│   ├── bar_spend_by_age_group.png
│   ├── missing_values.png
│   └── feature_importance.png
│
├── reports/
│   └── data_quality_report.txt   ← Automated quality + insights report
│
├── main.py                        ← Master pipeline script
├── requirements.txt
└── README.md
```

---

## 📊 Dataset Description

| Property | Value |
|----------|-------|
| Source | UCI Marketing Campaign Dataset |
| Rows | 2,240 customers |
| Columns | 29 original → 35+ after engineering |
| Missing values | 24 (Income column) |
| Duplicates | 0 |
| Date range | ~2012 – 2014 |

### Key Columns

| Column | Type | Description |
|--------|------|-------------|
| `Year_Birth` | int | Customer birth year |
| `Education` | category | Highest qualification |
| `Marital_Status` | category | Relationship status |
| `Income` | float | Annual household income |
| `MntWines/Fruits/…` | int | Spend per product category |
| `NumWebPurchases` | int | Online transactions |
| `AcceptedCmp1-5` | binary | Campaign response flags |
| `Response` | binary | Final campaign response |

---

## 🔧 Technologies Used

```
Python 3.10+          Core language
Pandas 2.x            Data manipulation
NumPy                 Numerical operations
Matplotlib / Seaborn  Static visualisations
Plotly / Dash         Interactive dashboard
SciPy                 Statistical tests
scikit-learn          Feature importance (Random Forest)
```

---

## 🧹 Data Cleaning Process

### Steps Applied

1. **Column Standardisation** — lowercase, snake_case, special-char removal  
2. **Missing Value Imputation** — Income: median fill; others: type-appropriate fill  
3. **Duplicate Removal** — exact-row deduplication  
4. **Categorical Fixes** — consolidated `Alone`, `Absurd`, `YOLO` → `Single`; `2n Cycle` → `2nd Cycle`  
5. **Data Type Conversion** — `Dt_Customer` parsed to datetime; tenure & age derived  
6. **Outlier Treatment (Z-score)** — rows with |z| > 3.5 on Income removed (e.g. $666,666 entry)  
7. **Outlier Treatment (IQR)** — Winsorisation on Income, Age, Total Spend  
8. **Feature Engineering** — `total_spend`, `total_purchases`, `total_campaigns_accepted`, `total_children`, `spend_income_ratio`, `age`, `customer_tenure_days`

---

## 📈 Visualisations

| Chart | File | Insight |
|-------|------|---------|
| Missing Values | `missing_values.png` | Raw data quality snapshot |
| Histogram | `histogram.png` | Age & income distributions |
| Box Plot | `boxplot.png` | Spend spread per category |
| Scatter Plot | `scatter_income_vs_spend.png` | Income–spend correlation by education |
| Correlation Heatmap | `heatmap.png` | Feature relationships |
| Count Plot | `countplot_categories.png` | Customer segment sizes |
| Pair Plot | `pairplot.png` | Multi-feature relationships |
| Bar Chart | `bar_spend_by_education.png` | Education-level spend |
| Pie Chart | `pie_marital_status.png` | Marital mix |
| Violin Plot | `violin_income_by_marital.png` | Income distribution by status |
| Campaign Bar | `bar_campaign_response.png` | Acceptance rates |
| Age-Group Bar | `bar_spend_by_age_group.png` | Age cohort spend |
| Feature Importance | `feature_importance.png` | RF-based response predictor |

---

## 💡 Key Insights

| # | Insight | Action |
|---|---------|--------|
| 1 | Income is the strongest spend predictor (r ≈ 0.79) | Build income-segmented loyalty tiers |
| 2 | PhD/Master customers spend most on average | Invest in professional-network advertising |
| 3 | Children negatively correlate with spend | Offer value bundles for family segments |
| 4 | Campaign 4 outperforms all others | Replicate its creative & targeting strategy |
| 5 | Recent buyers (low recency) respond 2× better | Trigger offers within 30 days of purchase |
| 6 | 40–60 age group are peak spenders | Prioritise mid-life cohort in media planning |
| 7 | Web buyers are deal-driven | Deploy web-exclusive flash sales |
| 8 | Married/Together households have highest incomes | Target with premium couple bundles |
| 9 | Multi-campaign responders have 3× higher LTV | Fast-track to VIP programmes |
| 10 | Tenure strongly predicts lifetime spend | Invest in onboarding and anniversary rewards |

---

## 🚀 Dashboard Features

- **KPI Cards** — Live customers, avg income, avg spend, response rate, recency  
- **Filters** — Education, Marital Status, Income range (RangeSlider)  
- **Charts** — All 6 major charts update dynamically with filters  
- **Insights Section** — 10 colour-coded business insight cards  
- **Dark Theme** — Deep-space design system (`#0F0F1A` background)

---

## ⚙️ Setup & Execution

```bash
# 1. Clone / navigate to project
cd Data-Cleaning-Visualization

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run full pipeline (cleaning + EDA + visualizations + report)
python main.py

# 4. Launch interactive dashboard
python dashboard/dashboard.py
# → Open http://127.0.0.1:8050
```

---

## 📋 Output Files

After running `main.py`:

| Output | Location |
|--------|----------|
| Cleaned dataset | `data/cleaned_dataset.csv` |
| 13 visualisation PNGs | `visualizations/` |
| Data quality report | `reports/data_quality_report.txt` |

---

## 🔮 Future Improvements

- [ ] Customer segmentation with K-Means clustering  
- [ ] Predictive churn model (XGBoost)  
- [ ] Cohort analysis by join year  
- [ ] Export dashboard to PDF/HTML report  
- [ ] RFM (Recency-Frequency-Monetary) segmentation  
- [ ] A/B test significance calculator in dashboard  

---

## 👤 Author

**Sahanaa**  
AI Intern — Thiranex  

---

*This project was built as part of the CodeAlpha Data Science & Machine Learning Internship programme.*
