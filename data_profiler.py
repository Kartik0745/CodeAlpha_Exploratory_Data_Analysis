"""
CodeAlpha Data Analytics Internship - Task 2: Exploratory Data Analysis (EDA)
Data Structure Profiler, Anomaly Detector & Data Hygiene Engine

Examines variables, data types, missingness, statistical distributions,
detects outliers via IQR and Z-scores, and highlights potential data issues.
"""

import os
import json
import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("DataProfiler")


class DataProfiler:
    """Profiles dataset structure, detects distribution anomalies, and identifies hygiene issues."""

    PRE_ANALYSIS_QUESTIONS = [
        {
            "id": "Q1",
            "topic": "Discount Margin Erosion Threshold",
            "question": "At what specific discount percentage do transactions transition from net profitable to loss-making?",
            "rationale": "Retailers frequently employ promotional discounts, but high discounts without volume compensation can catastrophically erode profit margins."
        },
        {
            "id": "Q2",
            "topic": "Customer Segment Lifetime Dynamics",
            "question": "Does the Corporate client segment generate a statistically higher Average Order Value (AOV) and customer tenure compared to Consumer and Home Office?",
            "rationale": "B2B customers often order higher quantities with lower churn, warranting tailored sales team investments."
        },
        {
            "id": "Q3",
            "topic": "Delivery Latency & Customer Satisfaction",
            "question": "How severely does shipping latency (Delivery Days > 7) degrade Customer Satisfaction Scores (CSAT)?",
            "rationale": "Logistics bottlenecks directly influence NPS and customer churn risk."
        },
        {
            "id": "Q4",
            "topic": "Category Return Rate Disparities",
            "question": "Are high-ticket items (Technology, Furniture) returned at a significantly higher frequency than consumable Office Supplies?",
            "rationale": "High returns incur reverse logistics overhead, inventory depreciation, and customer dissatisfaction."
        },
        {
            "id": "Q5",
            "topic": "Regional Profitability Disparities",
            "question": "Which geographic region yields the highest top-line sales but the lowest net profit margin?",
            "rationale": "Regional shipping costs, local discounting, and tax differentials can mask poor regional operating efficiency."
        }
    ]

    def __init__(self, data_path: str = "data/ecommerce_customer_analytics.csv"):
        self.data_path = data_path
        self.df: pd.DataFrame = pd.DataFrame()

    def load_data(self) -> pd.DataFrame:
        """Loads and prepares the dataset."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        self.df["Order_Date"] = pd.to_datetime(self.df["Order_Date"])
        self.df["Ship_Date"] = pd.to_datetime(self.df["Ship_Date"])
        return self.df

    def explore_structure(self) -> Dict[str, Any]:
        """Explores the dataset schema, data types, nulls, and memory footprint."""
        if self.df.empty:
            self.load_data()

        total_rows, total_cols = self.df.shape
        memory_usage_kb = self.df.memory_usage(deep=True).sum() / 1024

        column_summary = []
        for col in self.df.columns:
            dtype_str = str(self.df[col].dtype)
            null_count = int(self.df[col].isnull().sum())
            null_pct = round((null_count / total_rows) * 100, 2)
            n_unique = int(self.df[col].nunique())

            col_info = {
                "name": col,
                "dtype": dtype_str,
                "null_count": null_count,
                "null_percentage": null_pct,
                "unique_values": n_unique,
            }

            if pd.api.types.is_numeric_dtype(self.df[col]):
                col_info["type_category"] = "Numerical"
                col_info["mean"] = round(float(self.df[col].mean()), 2)
                col_info["std"] = round(float(self.df[col].std()), 2)
                col_info["min"] = round(float(self.df[col].min()), 2)
                col_info["max"] = round(float(self.df[col].max()), 2)
            elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
                col_info["type_category"] = "Temporal"
                col_info["min"] = self.df[col].min().strftime("%Y-%m-%d")
                col_info["max"] = self.df[col].max().strftime("%Y-%m-%d")
            else:
                col_info["type_category"] = "Categorical"
                top_val = self.df[col].mode().iloc[0] if not self.df[col].empty else "N/A"
                col_info["top_value"] = str(top_val)

            column_summary.append(col_info)

        return {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "memory_kb": round(memory_usage_kb, 1),
            "columns": column_summary
        }

    def detect_anomalies(self) -> Dict[str, Any]:
        """Detects outliers using IQR (Interquartile Range) and Z-Score (> 3 standard deviations)."""
        if self.df.empty:
            self.load_data()

        numerical_cols = ["Sales", "Profit", "Shipping_Cost", "Delivery_Days", "Quantity"]
        anomaly_results = {}

        for col in numerical_cols:
            series = self.df[col].dropna()
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            iqr_outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]

            # Z-score method
            z_scores = np.abs(stats.zscore(series))
            z_outliers_count = int(np.sum(z_scores > 3.0))

            anomaly_results[col] = {
                "q1": round(float(q1), 2),
                "q3": round(float(q3), 2),
                "iqr": round(float(iqr), 2),
                "lower_bound": round(float(lower_bound), 2),
                "upper_bound": round(float(upper_bound), 2),
                "iqr_outliers_count": len(iqr_outliers),
                "iqr_outliers_pct": round((len(iqr_outliers) / len(self.df)) * 100, 2),
                "z_score_outliers_count": z_outliers_count,
                "min_value": round(float(series.min()), 2),
                "max_value": round(float(series.max()), 2),
            }

        # Specific business anomalies: Loss leaders (Extreme Negative Profits)
        loss_makers = self.df[self.df["Profit"] < 0]
        heavy_loss_makers = self.df[self.df["Profit"] < -100]

        anomaly_results["business_anomalies"] = {
            "total_loss_transactions": len(loss_makers),
            "loss_transaction_pct": round((len(loss_makers) / len(self.df)) * 100, 2),
            "total_losses_incurred": round(float(abs(loss_makers["Profit"].sum())), 2),
            "severe_loss_transactions": len(heavy_loss_makers),
            "avg_discount_on_losses": round(float(loss_makers["Discount"].mean() * 100), 1),
            "avg_discount_on_profits": round(float(self.df[self.df["Profit"] >= 0]["Discount"].mean() * 100), 1),
        }

        return anomaly_results

    def detect_data_issues(self) -> Dict[str, Any]:
        """Detects skewness, multicollinearity, missing data patterns, and hygiene violations."""
        if self.df.empty:
            self.load_data()

        issues = {
            "skewness_issues": [],
            "missingness_issues": [],
            "multicollinearity_issues": [],
            "hygiene_flags": []
        }

        # 1. Skewness and Kurtosis check
        for col in ["Sales", "Profit", "Shipping_Cost", "Quantity"]:
            series = self.df[col].dropna()
            skew_val = float(stats.skew(series))
            kurt_val = float(stats.kurtosis(series))

            if abs(skew_val) > 1.0:
                issues["skewness_issues"].append({
                    "column": col,
                    "skewness": round(skew_val, 2),
                    "kurtosis": round(kurt_val, 2),
                    "diagnosis": "Highly skewed distribution (Right-skewed)" if skew_val > 1 else "Highly Left-skewed",
                    "recommendation": "Apply logarithmic transformation (log1p) or robust median scaling before predictive modeling."
                })

        # 2. Missing data mechanism analysis
        null_counts = self.df.isnull().sum()
        for col, count in null_counts[null_counts > 0].items():
            pct = (count / len(self.df)) * 100
            issues["missingness_issues"].append({
                "column": col,
                "missing_count": int(count),
                "missing_pct": round(pct, 2),
                "pattern": "Missing Completely at Random (MCAR)" if pct < 5 else "Missing at Random (MAR)",
                "recommendation": "Impute with median for numerical scores or preserve as explicit 'Unrecorded' category."
            })

        # 3. Multicollinearity / High Correlation check
        num_df = self.df.select_dtypes(include=[np.number]).dropna()
        corr_matrix = num_df.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                c1 = corr_matrix.columns[i]
                c2 = corr_matrix.columns[j]
                r_val = float(corr_matrix.iloc[i, j])
                if abs(r_val) > 0.65:
                    issues["multicollinearity_issues"].append({
                        "feature_pair": f"{c1} <-> {c2}",
                        "correlation_coefficient": round(r_val, 3),
                        "recommendation": "Monitor for feature redundancy in regression models."
                    })

        # 4. Hygiene checks
        neg_sales = len(self.df[self.df["Sales"] < 0])
        invalid_dates = len(self.df[self.df["Ship_Date"] < self.df["Order_Date"]])
        zero_qty = len(self.df[self.df["Quantity"] <= 0])

        issues["hygiene_flags"] = {
            "negative_sales_records": neg_sales,
            "invalid_shipping_dates": invalid_dates,
            "zero_or_negative_quantities": zero_qty,
            "status": "PASS: No invalid records or dates detected" if (neg_sales == 0 and invalid_dates == 0 and zero_qty == 0) else "FAIL: Hygiene violations found"
        }

        return issues

    def generate_comprehensive_report(self, output_path: str = "data/eda_comprehensive_report.txt") -> str:
        """Generates a complete executive textual report fulfilling all Task 2 criteria."""
        structure = self.explore_structure()
        anomalies = self.detect_anomalies()
        issues = self.detect_data_issues()

        report = f"""================================================================================
CODEALPHA DATA ANALYTICS INTERNSHIP: TASK 2 - EXPLORATORY DATA ANALYSIS (EDA)
COMPREHENSIVE EXPLORATORY & STATISTICAL INTELLIGENCE REPORT
================================================================================

PART 1: MEANINGFUL INQUIRY QUESTIONS (FRAMED PRIOR TO ANALYSIS)
--------------------------------------------------------------------------------
"""
        for q in self.PRE_ANALYSIS_QUESTIONS:
            report += f"[{q['id']}] {q['topic'].upper()}\n"
            report += f"    Question   : {q['question']}\n"
            report += f"    Business Rationale: {q['rationale']}\n\n"

        report += f"""
PART 2: DATA STRUCTURE & VARIABLE TAXONOMY
--------------------------------------------------------------------------------
- Total Transactions  : {structure['total_rows']:,} records
- Total Features      : {structure['total_columns']} features
- In-Memory Footprint : {structure['memory_kb']} KB
- Date Span Range     : {self.df['Order_Date'].min().strftime('%Y-%m-%d')} to {self.df['Order_Date'].max().strftime('%Y-%m-%d')}

Variable Schema Breakdown:
"""
        for col in structure["columns"]:
            null_str = f"{col['null_count']} nulls ({col['null_percentage']}%)" if col['null_count'] > 0 else "0 nulls (100% complete)"
            report += f"- {col['name']:<28} | {col['type_category']:<11} | {col['dtype']:<8} | {col['unique_values']:>4} unique | {null_str}\n"

        biz = anomalies["business_anomalies"]
        report += f"""
PART 3: TRENDS, PATTERNS & ANOMALY DETECTION
--------------------------------------------------------------------------------
1. OUTLIER BOUNDS (IQR & Z-SCORE AUDIT):
"""
        for col in ["Sales", "Profit", "Shipping_Cost", "Delivery_Days"]:
            a = anomalies[col]
            report += f"- {col:<15}: IQR=[{a['lower_bound']:>8.2f} to {a['upper_bound']:>8.2f}] | IQR Outliers={a['iqr_outliers_count']:>4} ({a['iqr_outliers_pct']:>4.1f}%) | |Z|>3 Outliers={a['z_score_outliers_count']}\n"

        report += f"""
2. MARGIN EROSION & LOSS-MAKING ANOMALY:
- Total Loss-making Transactions : {biz['total_loss_transactions']} ({biz['loss_transaction_pct']}% of all orders)
- Cumulative Losses Incurred      : ${biz['total_losses_incurred']:,.2f}
- Avg Discount on Profitable Orders: {biz['avg_discount_on_profits']}%
- Avg Discount on Loss Orders     : {biz['avg_discount_on_losses']}%  <-- [Key Pattern: Over-discounting causes severe margin loss]

PART 4: DATA ISSUES & QUALITY ASSESSMENT
--------------------------------------------------------------------------------
1. SKEWNESS & ASYMMETRY:
"""
        for sk in issues["skewness_issues"]:
            report += f"- {sk['column']:<14}: Skewness = {sk['skewness']} ({sk['diagnosis']}). Recommendation: {sk['recommendation']}\n"

        report += f"""
2. MISSING DATA PATTERNS:
"""
        for m in issues["missingness_issues"]:
            report += f"- {m['column']:<28}: {m['missing_count']} missing ({m['missing_pct']}%) - Pattern: {m['pattern']}. Strategy: {m['recommendation']}\n"

        report += f"""
3. DATA HYGIENE AUDIT:
- Negative Sales Values : {issues['hygiene_flags']['negative_sales_records']}
- Invalid Shipping Dates: {issues['hygiene_flags']['invalid_shipping_dates']}
- Zero/Negative Quantity: {issues['hygiene_flags']['zero_or_negative_quantities']}
- Hygiene Audit Status  : {issues['hygiene_flags']['status']}

================================================================================
Generated automatically by CodeAlpha Exploratory Data Analysis Pipeline
Interactive Visualizations and Hypothesis Test readouts are available in visualizations/
================================================================================
"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

        return report


def run_profiler(data_path: str = "data/ecommerce_customer_analytics.csv") -> DataProfiler:
    """Convenience runner."""
    profiler = DataProfiler(data_path)
    profiler.load_data()
    profiler.generate_comprehensive_report()
    return profiler


if __name__ == "__main__":
    run_profiler()
