# CodeAlpha Data Analytics Internship: Task 2 - Exploratory Data Analysis (EDA)

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Data-Pandas-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scipy](https://img.shields.io/badge/Statistics-Scipy-8CAAE6.svg?logo=scipy&logoColor=white)](https://scipy.org/)
[![Seaborn](https://img.shields.io/badge/Visualization-Seaborn-3776AB.svg)](https://seaborn.pydata.org/)
[![Flask](https://img.shields.io/badge/App-Flask-black.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-grade Exploratory Data Analysis (EDA) and inferential statistical validation project developed for the **CodeAlpha Data Analytics Internship** ([www.codealpha.tech](https://www.codealpha.tech)).

---

## 📌 Project Overview & Core Requirements

This project comprehensively addresses all five mandatory pillars of **Task 2: Exploratory Data Analysis**:

1. **Pre-Analysis Question Formulation**: Framing 5 impactful business inquiries regarding discount margin erosion, customer segment lifetime value, delivery latency impacts, and category return dynamics before inspecting the data.
2. **Structural Data Exploration**: Auditing shape (2,500 rows × 20 columns), memory footprint, variable taxonomy (continuous, discrete, categorical, temporal), and missingness patterns.
3. **Trends, Patterns & Anomaly Detection**: Identifying revenue seasonality, promotional loss-leaders, and outlier thresholds via **Interquartile Range (IQR)** and **Z-Score ($|Z| > 3$)** models.
4. **Hypothesis Testing & Statistical Validation**: Executing 4 formal inferential hypothesis tests using `scipy.stats` (Welch's Two-Sample T-Test, One-Way ANOVA, Pearson/Spearman Rank Correlation, and Chi-Square Contingency Test) with effect size metrics (Cohen's $d$, $\eta^2$, Cramér's $V$).
5. **Data Issue Detection & Quality Diagnostics**: Auditing feature skewness/kurtosis (log-transformation requirements), analyzing missing data mechanisms (MCAR vs MAR), and validating operational data hygiene (valid shipping sequence, non-negative quantities).

---

## 🗂️ Project Directory Structure

```
CodeAlpha_Exploratory_Data_Analysis/
├── data/                                 # Datasets and Statistical Audit Reports
│   ├── ecommerce_customer_analytics.csv  # 2,500-transaction E-Commerce dataset
│   ├── eda_comprehensive_report.txt      # Formatted executive intelligence report
│   └── hypothesis_test_results.json      # Structured test statistics & p-values
├── visualizations/                       # 8 publication-grade charts (300 DPI PNG)
│   ├── 01_correlation_matrix.png         # Correlation heatmap & multicollinearity
│   ├── 02_sales_profit_distributions.png # Skewness, KDE & outlier boxplots
│   ├── 03_discount_vs_profit_erosion.png # Margin collapse scatter & 25% threshold
│   ├── 04_category_regional_performance.png # Regional & categorical breakdown
│   ├── 05_monthly_sales_trend.png        # Time-series revenue seasonality trend
│   ├── 06_customer_segment_comparison.png# AOV, margin %, and tenure profiles
│   ├── 07_delivery_vs_satisfaction.png   # Violin plot: delivery delay vs CSAT
│   └── 08_return_rate_analysis.png       # Chi-square return rates by category
├── notebooks/
│   └── CodeAlpha_EDA_Deep_Dive.ipynb     # Interactive Jupyter Notebook
├── templates/
│   └── index.html                        # Modern HTML5 EDA portal UI
├── static/
│   ├── style.css                         # Glassmorphic stylesheet
│   └── script.js                         # Dynamic table filtering & modal logic
├── dataset_generator.py                  # Generates realistic 2,500-record dataset
├── data_profiler.py                      # Structural audit, anomalies & quality checks
├── hypothesis_testing.py                 # Scipy inferential testing engine
├── visualization_engine.py               # Seaborn/Matplotlib chart renderer
├── app.py                                # Interactive Flask web dashboard
├── run_eda_pipeline.py                   # Master one-click CLI orchestrator
├── requirements.txt                      # Project dependencies
├── LINKEDIN_POST_GUIDE.md                # Video script & LinkedIn post copy
└── README.md                             # Documentation
```

---

## 🔬 Statistical Hypothesis Testing Summary

All hypothesis tests were evaluated against a significance threshold of $\alpha = 0.05$:

| ID | Hypothesis Tested | Statistical Test | p-value | Decision ($\alpha=0.05$) | Key Business Implication |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **H1** | Return Status vs. Order Value | Welch's Two-Sample T-Test | $p < 0.001$ | **Reject $H_0$** | High-ticket items suffer significantly different return rates requiring targeted sizing/packaging |
| **H2** | Profitability Across Customer Segments | One-Way ANOVA & Kruskal-Wallis | $p < 0.05$ | **Reject $H_0$** | Segment margins vary; Corporate accounts deliver higher AOV and stability |
| **H3** | Discount vs. Profit Margin Collapse | Pearson & Spearman Correlation | $r = -0.52, p < 0.001$ | **Reject $H_0$** | Strong negative correlation; discounts $>25\%$ trigger catastrophic margin collapse |
| **H4** | Product Category vs. Return Dependency | Chi-Square Test ($\chi^2$) | $p < 0.001$ | **Reject $H_0$** | Furniture (18.5% returns) and Tech (13.2%) require proactive return mitigation |

---

## ⚙️ Installation & How to Run

### 1. Install Dependencies
```bash
cd CodeAlpha_Exploratory_Data_Analysis
pip install -r requirements.txt
```

### 2. Option A: Run Full EDA Pipeline (One-Click)
```bash
python run_eda_pipeline.py
```
*Generates dataset, structural audit report, 4 hypothesis test outputs, and all 8 publication charts.*

### 3. Option B: Launch Interactive Web Dashboard
```bash
python app.py
```
Navigate to **`http://127.0.0.1:5001`** to interact with live hypothesis cards, outlier diagnostics, and the transaction filter table.

### 4. Option C: Interactive Jupyter Notebook
```bash
jupyter notebook notebooks/CodeAlpha_EDA_Deep_Dive.ipynb
```

---

## 🏆 CodeAlpha Submission Checklist

- [x] **Task 2: Exploratory Data Analysis completed** meeting all 5 requirements.
- [x] **Meaningful pre-analysis questions framed**.
- [x] **Data structure, variables, and data types profiled**.
- [x] **Trends, patterns, and IQR/Z-score anomalies detected**.
- [x] **Hypotheses statistically validated** using Scipy and Seaborn.
- [x] **Data hygiene issues documented** with mitigation strategies.
- [x] **Interactive Web Dashboard provided** for live demonstration.
- [x] **Repository naming followed**: `CodeAlpha_Exploratory_Data_Analysis`.
- [x] **LinkedIn Guide & Video Presentation Script** in [LINKEDIN_POST_GUIDE.md](LINKEDIN_POST_GUIDE.md).

---

## 📄 License
This project is licensed under the MIT License - open for educational and portfolio demonstration use.
