"""
CodeAlpha Data Analytics Internship - Task 2: Exploratory Data Analysis (EDA)
Publication-Grade Visual Analytics Engine

Generates 8 high-resolution statistical charts using Seaborn and Matplotlib:
- Correlation Heatmaps
- Multi-panel Distribution & Outlier Boxplots
- Discount vs. Margin Erosion Scatter with Breakeven Thresholds
- Category & Regional Performance Breakdown
- Time-Series Revenue Seasonality Trend
- Customer Segment Dynamics (AOV, Tenure, Profit Margin)
- Delivery Delay vs. Customer Satisfaction Score (Violin Plot)
- Return Rates by Category & Order Priority (Chi-Square Visualization)
"""

import os
import logging
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger("VisualizationEngine")

# Visualization style configuration
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "font.sans-serif": "Arial",
    "font.family": "sans-serif",
    "figure.titlesize": 15,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "figure.autolayout": True,
    "savefig.dpi": 300,
})


class EDAVisualizer:
    """Generates publication-quality charts for the E-Commerce analytics dataset."""

    def __init__(self, data_path: str = "data/ecommerce_customer_analytics.csv", output_dir: str = "visualizations"):
        self.data_path = data_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.df: pd.DataFrame = pd.DataFrame()

    def load_data(self) -> pd.DataFrame:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        self.df["Order_Date"] = pd.to_datetime(self.df["Order_Date"])
        self.df["Ship_Date"] = pd.to_datetime(self.df["Ship_Date"])
        return self.df

    def generate_all_charts(self) -> None:
        """Renders and saves all 8 analytical figures."""
        if self.df.empty:
            self.load_data()

        logger.info("Generating publication visualizations...")
        self._plot_correlation_matrix()
        self._plot_distributions_and_outliers()
        self._plot_discount_profit_erosion()
        self._plot_category_regional_performance()
        self._plot_monthly_sales_trend()
        self._plot_customer_segment_dynamics()
        self._plot_delivery_vs_satisfaction()
        self._plot_return_rate_analysis()
        logger.info(f"All 8 visualization charts saved successfully in '{self.output_dir}'")

    def _plot_correlation_matrix(self) -> None:
        """1. Annotated Heatmap of Numerical Features."""
        num_cols = ["Sales", "Quantity", "Discount", "Profit", "Shipping_Cost", "Customer_Age", "Customer_Tenure_Months", "Delivery_Days", "Customer_Satisfaction_Score"]
        corr = self.df[num_cols].corr()

        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        sns.heatmap(
            corr,
            mask=mask,
            cmap=cmap,
            vmax=1.0,
            vmin=-1.0,
            center=0,
            annot=True,
            fmt=".2f",
            square=True,
            linewidths=0.7,
            cbar_kws={"shrink": 0.8, "label": "Pearson Correlation (r)"}
        )
        plt.title("01. Numerical Features Correlation Matrix & Multicollinearity Audit", pad=15, weight="bold")
        save_path = os.path.join(self.output_dir, "01_correlation_matrix.png")
        plt.savefig(save_path)
        plt.close()

    def _plot_distributions_and_outliers(self) -> None:
        """2. Multi-panel Sales & Profit Distribution with Outlier Boxplots."""
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))

        # Panel A: Sales Distribution (Histogram + KDE)
        sns.histplot(self.df["Sales"], kde=True, ax=axes[0, 0], color="#1d3557", bins=30)
        axes[0, 0].set_title("Sales Distribution (Right-Skewed)", weight="bold")
        axes[0, 0].set_xlabel("Sales ($)")
        axes[0, 0].axvline(self.df["Sales"].mean(), color="red", linestyle="--", label=f"Mean (${self.df['Sales'].mean():.1f})")
        axes[0, 0].axvline(self.df["Sales"].median(), color="green", linestyle=":", label=f"Median (${self.df['Sales'].median():.1f})")
        axes[0, 0].legend()

        # Panel B: Sales Boxplot (Outliers)
        sns.boxplot(x=self.df["Sales"], ax=axes[0, 1], color="#457b9d", flierprops=dict(marker="o", color="red", alpha=0.5))
        axes[0, 1].set_title("Sales Outlier Inspection (IQR Bounds)", weight="bold")
        axes[0, 1].set_xlabel("Sales ($)")

        # Panel C: Profit Distribution (Histogram + KDE with Zero Line)
        sns.histplot(self.df["Profit"], kde=True, ax=axes[1, 0], color="#2a9d8f", bins=30)
        axes[1, 0].axvline(0, color="black", linestyle="-", linewidth=1.5, label="Breakeven ($0)")
        axes[1, 0].axvline(self.df["Profit"].mean(), color="red", linestyle="--", label=f"Mean (${self.df['Profit'].mean():.1f})")
        axes[1, 0].set_title("Profit Distribution (Loss-Leaders vs Profitable)", weight="bold")
        axes[1, 0].set_xlabel("Profit ($)")
        axes[1, 0].legend()

        # Panel D: Profit Boxplot (Loss Outliers)
        sns.boxplot(x=self.df["Profit"], ax=axes[1, 1], color="#e76f51", flierprops=dict(marker="o", color="darkred", alpha=0.5))
        axes[1, 1].axvline(0, color="black", linestyle="--", alpha=0.7)
        axes[1, 1].set_title("Profit Outliers (Extreme Negative Margin Anomalies)", weight="bold")
        axes[1, 1].set_xlabel("Profit ($)")

        plt.suptitle("02. Exploratory Distribution & Outlier Anomaly Analysis", y=1.02, weight="bold", fontsize=15)
        save_path = os.path.join(self.output_dir, "02_sales_profit_distributions.png")
        plt.savefig(save_path)
        plt.close()

    def _plot_discount_profit_erosion(self) -> None:
        """3. Scatter Plot: Discount vs Profit with Breakeven and 25% Threshold."""
        plt.figure(figsize=(11, 7))

        palette = {"Technology": "#2b5c8f", "Furniture": "#e63946", "Office Supplies": "#2a9d8f"}
        sns.scatterplot(
            data=self.df,
            x="Discount",
            y="Profit",
            hue="Category",
            palette=palette,
            alpha=0.65,
            s=45
        )

        # Regression line
        sns.regplot(
            data=self.df,
            x="Discount",
            y="Profit",
            scatter=False,
            color="black",
            line_kws={"linestyle": "-", "linewidth": 2, "label": "Fitted Margin Trend"}
        )

        # Breakeven line at Profit = 0
        plt.axhline(0, color="crimson", linestyle="--", linewidth=1.8, label="Breakeven Line ($0 Profit)")
        # Anomaly threshold at Discount = 25%
        plt.axvline(0.25, color="darkorange", linestyle=":", linewidth=2, label="25% Discount Danger Threshold")

        plt.title("03. Discount vs. Profit Margin Collapse & Negative Outlier Pattern", pad=15, weight="bold")
        plt.xlabel("Promotional Discount Rate (0% to 70%)")
        plt.ylabel("Net Profit ($)")
        plt.legend(frameon=True, loc="upper right")

        save_path = os.path.join(self.output_dir, "03_discount_vs_profit_erosion.png")
        plt.savefig(save_path)
        plt.close()

    def _plot_category_regional_performance(self) -> None:
        """4. Category & Regional Breakdown of Total Sales and Total Profit."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Sales by Region & Category
        sales_pivot = self.df.pivot_table(index="Region", columns="Category", values="Sales", aggfunc="sum")
        sales_pivot.plot(kind="bar", stacked=False, ax=axes[0], colormap="viridis", width=0.8)
        axes[0].set_title("Total Sales ($) by Region & Category", weight="bold")
        axes[0].set_ylabel("Total Sales ($)")
        axes[0].set_xlabel("Geographic Region")
        axes[0].legend(title="Category")
        axes[0].tick_params(axis="x", rotation=0)

        # Profit by Region & Category
        profit_pivot = self.df.pivot_table(index="Region", columns="Category", values="Profit", aggfunc="sum")
        profit_pivot.plot(kind="bar", stacked=False, ax=axes[1], colormap="coolwarm", width=0.8)
        axes[1].axhline(0, color="black", linestyle="--", alpha=0.5)
        axes[1].set_title("Total Profit ($) by Region & Category", weight="bold")
        axes[1].set_ylabel("Total Net Profit ($)")
        axes[1].set_xlabel("Geographic Region")
        axes[1].legend(title="Category")
        axes[1].tick_params(axis="x", rotation=0)

        plt.suptitle("04. Regional & Categorical Commercial Performance Comparison", y=1.02, weight="bold", fontsize=15)
        save_path = os.path.join(self.output_dir, "04_category_regional_performance.png")
        plt.savefig(save_path)
        plt.close()

    def _plot_monthly_sales_trend(self) -> None:
        """5. Time-Series Monthly Revenue and Profit Trend."""
        monthly = self.df.set_index("Order_Date").resample("ME")[["Sales", "Profit"]].sum().reset_index()
        monthly["Sales_MA3"] = monthly["Sales"].rolling(window=3, min_periods=1).mean()

        plt.figure(figsize=(12, 6))
        plt.plot(monthly["Order_Date"], monthly["Sales"], marker="o", color="#1d3557", linewidth=2.2, label="Monthly Sales ($)")
        plt.plot(monthly["Order_Date"], monthly["Sales_MA3"], linestyle="--", color="#457b9d", linewidth=1.8, label="3-Month Moving Average")
        plt.plot(monthly["Order_Date"], monthly["Profit"], marker="s", color="#2a9d8f", linewidth=2, label="Monthly Profit ($)")
        plt.axhline(0, color="red", linestyle=":", alpha=0.6)

        plt.title("05. Monthly Sales & Profitability Trend (2024–2025)", pad=15, weight="bold")
        plt.xlabel("Timeline (Month / Year)")
        plt.ylabel("Amount ($)")
        plt.xticks(rotation=30)
        plt.legend(frameon=True)

        save_path = os.path.join(self.output_dir, "05_monthly_sales_trend.png")
        plt.savefig(save_path)
        plt.close()

    def _plot_customer_segment_dynamics(self) -> None:
        """6. Customer Segment Comparison (AOV, Total Margin %, Average Tenure)."""
        seg_summary = self.df.groupby("Customer_Segment").agg(
            AOV=("Sales", "mean"),
            Margin_Pct=("Profit", lambda p: (p.sum() / self.df.loc[p.index, "Sales"].sum()) * 100),
            Avg_Tenure=("Customer_Tenure_Months", "mean")
        ).reset_index()

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        colors = ["#43aa8b", "#4d908e", "#577590"]

        # AOV
        sns.barplot(data=seg_summary, x="Customer_Segment", y="AOV", ax=axes[0], hue="Customer_Segment", palette=colors, legend=False)
        axes[0].set_title("Average Order Value (AOV)", weight="bold")
        axes[0].set_ylabel("Mean Sales ($)")
        for p in axes[0].patches:
            axes[0].annotate(f"${p.get_height():.1f}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                             ha="center", va="center", color="white", weight="bold")

        # Margin %
        sns.barplot(data=seg_summary, x="Customer_Segment", y="Margin_Pct", ax=axes[1], hue="Customer_Segment", palette=colors, legend=False)
        axes[1].set_title("Net Profit Margin (%)", weight="bold")
        axes[1].set_ylabel("Profit / Sales (%)")
        for p in axes[1].patches:
            axes[1].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                             ha="center", va="center", color="white", weight="bold")

        # Average Tenure
        sns.barplot(data=seg_summary, x="Customer_Segment", y="Avg_Tenure", ax=axes[2], hue="Customer_Segment", palette=colors, legend=False)
        axes[2].set_title("Average Customer Tenure (Months)", weight="bold")
        axes[2].set_ylabel("Months Active")
        for p in axes[2].patches:
            axes[2].annotate(f"{p.get_height():.1f}m", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                             ha="center", va="center", color="white", weight="bold")

        plt.suptitle("06. Customer Segment Strategic Profile (Hypothesis 2)", y=1.02, weight="bold", fontsize=15)
        save_path = os.path.join(self.output_dir, "06_customer_segment_comparison.png")
        plt.savefig(save_path)
        plt.close()

    def _plot_delivery_vs_satisfaction(self) -> None:
        """7. Split Violin Plot: Delivery Days impact on Customer Satisfaction."""
        clean_df = self.df.dropna(subset=["Customer_Satisfaction_Score", "Delivery_Days"]).copy()
        clean_df["Delivery_Bucket"] = pd.cut(
            clean_df["Delivery_Days"],
            bins=[0, 3, 6, 9, 20],
            labels=["1-3 Days (Fast)", "4-6 Days (Normal)", "7-9 Days (Delayed)", "10+ Days (Severe Delay)"]
        )

        plt.figure(figsize=(10, 6))
        sns.violinplot(
            data=clean_df,
            x="Delivery_Bucket",
            y="Customer_Satisfaction_Score",
            hue="Delivery_Bucket",
            palette="rocket_r",
            legend=False,
            inner="quartile"
        )
        plt.title("07. Delivery Delay Impact on Customer Satisfaction Score (CSAT)", pad=15, weight="bold")
        plt.xlabel("Logistics Delivery Duration")
        plt.ylabel("Satisfaction Rating (1 = Unsatisfied, 5 = Highly Satisfied)")

        save_path = os.path.join(self.output_dir, "07_delivery_vs_satisfaction.png")
        plt.savefig(save_path)
        plt.close()

    def _plot_return_rate_analysis(self) -> None:
        """8. Return Rate (%) by Category and Order Priority (Chi-Square Visualization)."""
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # By Category
        cat_returns = self.df.groupby("Category")["Returned"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
        sns.barplot(data=cat_returns, x="Category", y="Returned", ax=axes[0], hue="Category", palette="mako", legend=False)
        axes[0].set_title("Return Rate (%) by Product Category\n(Chi-Square Significant)", weight="bold")
        axes[0].set_ylabel("Return Rate (%)")
        for p in axes[0].patches:
            axes[0].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 0.5),
                             ha="center", weight="bold")

        # By Priority
        prio_returns = self.df.groupby("Order_Priority")["Returned"].apply(lambda s: (s == "Yes").mean() * 100).reindex(
            ["Low", "Medium", "High", "Critical"]).reset_index()
        sns.barplot(data=prio_returns, x="Order_Priority", y="Returned", ax=axes[1], hue="Order_Priority", palette="flare", legend=False)
        axes[1].set_title("Return Rate (%) by Order Priority", weight="bold")
        axes[1].set_ylabel("Return Rate (%)")
        for p in axes[1].patches:
            axes[1].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 0.5),
                             ha="center", weight="bold")

        plt.suptitle("08. Product Return Propensity & Risk Factor Analysis", y=1.02, weight="bold", fontsize=15)
        save_path = os.path.join(self.output_dir, "08_return_rate_analysis.png")
        plt.savefig(save_path)
        plt.close()


def run_visualizations(data_path: str = "data/ecommerce_customer_analytics.csv", output_dir: str = "visualizations"):
    visualizer = EDAVisualizer(data_path, output_dir)
    visualizer.load_data()
    visualizer.generate_all_charts()


if __name__ == "__main__":
    run_visualizations()
