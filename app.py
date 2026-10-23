"""
CodeAlpha Data Analytics Internship - Task 2: Exploratory Data Analysis (EDA)
Interactive Web Dashboard & Statistical Inference Portal (Flask)

Enables users to explore dataset KPIs, inspect the 5 pre-analysis questions,
evaluate 4 formal statistical hypothesis tests, view HD visualizations,
audit outliers, and filter raw records in real time.
"""

import os
import json
from typing import Dict, Any, List

from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd

from data_profiler import DataProfiler
from hypothesis_testing import HypothesisTestingEngine

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
VIS_DIR = os.path.join(BASE_DIR, "visualizations")
CSV_PATH = os.path.join(DATA_DIR, "ecommerce_customer_analytics.csv")
REPORT_PATH = os.path.join(DATA_DIR, "eda_comprehensive_report.txt")
HYPO_PATH = os.path.join(DATA_DIR, "hypothesis_test_results.json")


def load_dataset() -> pd.DataFrame:
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        df["Order_Date"] = pd.to_datetime(df["Order_Date"])
        df["Ship_Date"] = pd.to_datetime(df["Ship_Date"])
        return df
    return pd.DataFrame()


@app.route("/")
def index():
    df = load_dataset()
    stats_data = {
        "total_orders": 0,
        "total_sales": "$0",
        "total_profit": "$0",
        "profit_margin": "0.0%",
        "return_rate": "0.0%",
        "avg_csat": "0.0",
        "has_data": False,
    }

    if not df.empty:
        total_rev = df["Sales"].sum()
        total_prof = df["Profit"].sum()
        margin = (total_prof / total_rev) * 100 if total_rev > 0 else 0
        ret_rate = (df["Returned"] == "Yes").mean() * 100
        csat_mean = df["Customer_Satisfaction_Score"].dropna().mean()

        stats_data = {
            "total_orders": f"{len(df):,}",
            "total_sales": f"${total_rev:,.2f}",
            "total_profit": f"${total_prof:,.2f}",
            "profit_margin": f"{margin:.1f}%",
            "return_rate": f"{ret_rate:.1f}%",
            "avg_csat": f"{csat_mean:.2f}",
            "has_data": True,
        }

    # Load questions
    profiler = DataProfiler(CSV_PATH)
    questions = profiler.PRE_ANALYSIS_QUESTIONS

    # Load hypothesis results
    hypo_data = {}
    if os.path.exists(HYPO_PATH):
        with open(HYPO_PATH, "r", encoding="utf-8") as f:
            hypo_data = json.load(f)

    # List visualizations
    vis_files = []
    if os.path.exists(VIS_DIR):
        vis_files = sorted([f for f in os.listdir(VIS_DIR) if f.endswith(".png")])

    return render_template(
        "index.html",
        stats=stats_data,
        questions=questions,
        hypo=hypo_data,
        vis_files=vis_files,
        categories=sorted(df["Category"].unique().tolist()) if not df.empty else [],
        regions=sorted(df["Region"].unique().tolist()) if not df.empty else [],
        segments=sorted(df["Customer_Segment"].unique().tolist()) if not df.empty else [],
    )


@app.route("/api/stats")
def api_stats():
    df = load_dataset()
    if df.empty:
        return jsonify({"status": "no_data"})

    return jsonify({
        "total_orders": len(df),
        "total_sales": float(df["Sales"].sum()),
        "total_profit": float(df["Profit"].sum()),
        "profit_margin_pct": float((df["Profit"].sum() / df["Sales"].sum()) * 100),
        "return_rate_pct": float((df["Returned"] == "Yes").mean() * 100),
        "avg_delivery_days": float(df["Delivery_Days"].mean()),
        "avg_csat": float(df["Customer_Satisfaction_Score"].dropna().mean()),
    })


@app.route("/api/hypothesis_tests")
def api_hypo():
    if os.path.exists(HYPO_PATH):
        with open(HYPO_PATH, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    engine = HypothesisTestingEngine(CSV_PATH)
    engine.load_data()
    return jsonify(engine.run_all_tests())


@app.route("/api/outliers")
def api_outliers():
    profiler = DataProfiler(CSV_PATH)
    profiler.load_data()
    return jsonify(profiler.detect_anomalies())


@app.route("/api/records")
def api_records():
    df = load_dataset()
    if df.empty:
        return jsonify({"count": 0, "records": []})

    category = request.args.get("category", "").strip()
    region = request.args.get("region", "").strip()
    segment = request.args.get("segment", "").strip()
    returned = request.args.get("returned", "").strip()
    search = request.args.get("search", "").strip().lower()

    filtered = df.copy()

    if category and category != "all":
        filtered = filtered[filtered["Category"] == category]
    if region and region != "all":
        filtered = filtered[filtered["Region"] == region]
    if segment and segment != "all":
        filtered = filtered[filtered["Customer_Segment"] == segment]
    if returned and returned != "all":
        filtered = filtered[filtered["Returned"] == returned]

    if search:
        filtered = filtered[
            filtered["Order_ID"].str.lower().str.contains(search, na=False) |
            filtered["Customer_ID"].str.lower().str.contains(search, na=False) |
            filtered["Sub_Category"].str.lower().str.contains(search, na=False)
        ]

    # Convert datetimes to strings
    filtered["Order_Date"] = filtered["Order_Date"].dt.strftime("%Y-%m-%d")
    filtered["Ship_Date"] = filtered["Ship_Date"].dt.strftime("%Y-%m-%d")

    records = filtered.head(100).to_dict(orient="records")
    return jsonify({
        "count": len(records),
        "total_matched": len(filtered),
        "total_available": len(df),
        "records": records
    })


@app.route("/visualizations/<path:filename>")
def serve_vis(filename):
    return send_from_directory(VIS_DIR, filename)


@app.route("/download/<path:filename>")
def download_data(filename):
    return send_from_directory(DATA_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
