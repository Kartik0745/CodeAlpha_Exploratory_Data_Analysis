"""
CodeAlpha Data Analytics Internship - Task 2: Exploratory Data Analysis (EDA)
Master Pipeline Orchestrator

One-click execution script that integrates:
1. Dataset Synthesis / Verification
2. Structural Profiling & Data Hygiene Audit
3. Inferential Hypothesis Testing (T-Test, ANOVA, Correlation, Chi-Square)
4. Publication Visualization Rendering (8 Charts)
5. Executive Reporting & Optional Web Dashboard
"""

import os
import sys
import time
import argparse
import logging

from dataset_generator import generate_ecommerce_dataset
from data_profiler import DataProfiler
from hypothesis_testing import HypothesisTestingEngine
from visualization_engine import EDAVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("MasterEDAPipeline")


def execute_full_eda(n_records: int = 2500, regenerate: bool = False, serve: bool = False):
    start_time = time.time()
    logger.info("=" * 75)
    logger.info("STARTING CODEALPHA DATA ANALYTICS TASK 2: EXPLORATORY DATA ANALYSIS")
    logger.info("=" * 75)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    vis_dir = os.path.join(base_dir, "visualizations")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    csv_path = os.path.join(data_dir, "ecommerce_customer_analytics.csv")
    report_path = os.path.join(data_dir, "eda_comprehensive_report.txt")
    hypo_path = os.path.join(data_dir, "hypothesis_test_results.json")

    # Step 1: Dataset Generation
    logger.info("\n[STEP 1/4] Checking & Synthesizing Multi-Dimensional Dataset...")
    if not os.path.exists(csv_path) or regenerate:
        df = generate_ecommerce_dataset(n_records=n_records, output_path=csv_path)
    else:
        logger.info(f"Using existing dataset at: {csv_path}")

    # Step 2: Data Structure Profiling, Outliers & Quality Audit
    logger.info("\n[STEP 2/4] Profiling Data Structure, Anomalies & Hygiene Issues...")
    profiler = DataProfiler(data_path=csv_path)
    profiler.load_data()
    profiler.generate_comprehensive_report(output_path=report_path)
    logger.info(f"Comprehensive EDA report generated: {report_path}")

    # Step 3: Statistical Hypothesis Testing
    logger.info("\n[STEP 3/4] Conducting Inferential Hypothesis Tests (T-Test, ANOVA, Chi-Square)...")
    hypo_engine = HypothesisTestingEngine(data_path=csv_path)
    hypo_engine.load_data()
    hypo_results = hypo_engine.run_all_tests(output_path=hypo_path)

    # Step 4: Publication Visualization Rendering
    logger.info("\n[STEP 4/4] Rendering 8 Publication-Grade Visual Analytics Figures...")
    visualizer = EDAVisualizer(data_path=csv_path, output_dir=vis_dir)
    visualizer.load_data()
    visualizer.generate_all_charts()

    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 75)
    logger.info(f"EDA PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS!")
    logger.info("=" * 75)

    print("\nSummary of Inferential Hypothesis Tests:")
    for hid, t in hypo_results.items():
        print(f"  [{hid}] {t['name']:<42} | p = {t['formatted_p_value']:<8} | {t['decision']}")

    print("\nGenerated Output Files:")
    print(f"  - [DATASET] {csv_path}")
    print(f"  - [REPORT]  {report_path}")
    print(f"  - [STATS]   {hypo_path}")
    print(f"  - [CHARTS]  8 Visualizations saved in: {vis_dir}/")
    print("=" * 75)

    if serve:
        logger.info("\nLaunching Interactive Web Dashboard...")
        from app import app
        app.run(host="127.0.0.1", port=5001, debug=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CodeAlpha EDA Pipeline")
    parser.add_argument("--records", type=int, default=2500, help="Number of records to synthesize (default: 2500)")
    parser.add_argument("--regenerate", action="store_true", help="Force regenerate dataset")
    parser.add_argument("--serve", action="store_true", help="Launch interactive Flask dashboard on port 5001")
    args = parser.parse_args()

    execute_full_eda(n_records=args.records, regenerate=args.regenerate, serve=args.serve)
