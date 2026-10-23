"""
CodeAlpha Data Analytics Internship - Task 2: Exploratory Data Analysis (EDA)
Enterprise E-Commerce & Retail Customer Analytics Dataset Generator

Synthesizes a realistic, multi-dimensional transaction dataset of 2,500+ records
with real-world distributions, seasonal trends, edge cases, missingness, and anomalies.
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_ecommerce_dataset(n_records: int = 2500, random_seed: int = 42, output_path: str = "data/ecommerce_customer_analytics.csv") -> pd.DataFrame:
    """Generates a rich, multi-featured E-Commerce transaction dataset."""
    np.random.seed(random_seed)
    random.seed(random_seed)

    # 1. Customer IDs (500 unique customers for repeat behavior)
    customer_pool = [f"CUST-{1000 + i}" for i in range(500)]
    customer_assignments = {cid: {
        "segment": random.choices(["Consumer", "Corporate", "Home Office"], weights=[0.52, 0.30, 0.18])[0],
        "age": int(np.clip(np.random.normal(38, 11), 18, 72)),
        "tenure": int(np.clip(np.random.exponential(18), 1, 60))
    } for cid in customer_pool}

    # 2. Categories & Sub-Categories
    catalog = {
        "Technology": ["Phones", "Laptops", "Accessories", "Printers"],
        "Furniture": ["Chairs", "Tables", "Bookcases", "Furnishings"],
        "Office Supplies": ["Binders", "Storage", "Paper", "Art"]
    }
    categories = list(catalog.keys())
    cat_weights = [0.35, 0.30, 0.35]

    regions = ["West", "East", "Central", "South", "North"]
    priorities = ["Low", "Medium", "High", "Critical"]
    payment_methods = ["Credit Card", "Debit Card", "PayPal", "UPI", "Net Banking"]

    # 3. Base date range: 2024-01-01 to 2025-12-31 (730 days)
    start_date = datetime(2024, 1, 1)

    records = []
    for i in range(n_records):
        order_id = f"CA-{2024 + (i % 2)}-{10000 + i}"
        cid = random.choice(customer_pool)
        cust_profile = customer_assignments[cid]

        # Order Date with Q4 Black Friday / Holiday seasonality spike
        day_offset = random.randint(0, 729)
        order_date = start_date + timedelta(days=day_offset)
        # November/December seasonality weight
        if order_date.month in [11, 12] and random.random() < 0.4:
            day_offset = random.randint(0, 729)
            order_date = start_date + timedelta(days=day_offset)

        category = random.choices(categories, weights=cat_weights)[0]
        sub_category = random.choice(catalog[category])
        region = random.choice(regions)
        priority = random.choices(priorities, weights=[0.25, 0.40, 0.23, 0.12])[0]
        payment = random.choices(payment_methods, weights=[0.38, 0.22, 0.18, 0.14, 0.08])[0]

        # Numerical: Sales (Lognormal distribution for positive right skew)
        if category == "Technology":
            base_sales = np.random.lognormal(mean=5.8, sigma=0.85)  # $150 to $3,000
        elif category == "Furniture":
            base_sales = np.random.lognormal(mean=5.2, sigma=0.75)  # $80 to $1,800
        else:
            base_sales = np.random.lognormal(mean=4.1, sigma=0.70)  # $20 to $400

        sales = round(float(np.clip(base_sales, 12.0, 4800.0)), 2)
        quantity = int(np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], p=[0.25, 0.25, 0.18, 0.12, 0.08, 0.05, 0.03, 0.02, 0.01, 0.01]))

        # Discount: Discrete tiers (0% to 70%)
        discount = float(random.choices([0.0, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70],
                                        weights=[0.35, 0.18, 0.14, 0.12, 0.09, 0.05, 0.04, 0.02, 0.01])[0])

        # Profit Margin Dynamics: Base margin varies by category
        # Critical anomaly: Discounts > 0.25 lead to heavy profit erosion / loss leaders
        base_margin = {"Technology": 0.28, "Furniture": 0.18, "Office Supplies": 0.36}[category]
        # Noise
        margin_noise = np.random.normal(0, 0.05)
        effective_margin = base_margin - (discount * 1.6) + margin_noise

        profit = round(float(sales * effective_margin), 2)
        # Inject intentional extreme outliers (large profit wins and extreme return losses)
        if i % 180 == 0:
            profit = round(profit * 2.8, 2)  # High positive outlier
        elif i % 220 == 0 and discount >= 0.4:
            profit = round(-abs(sales * 0.45), 2)  # High negative anomaly

        # Shipping & Delivery
        delivery_days = int(np.clip(np.random.normal(loc=4.5, scale=2.1), 1, 14))
        ship_date = order_date + timedelta(days=delivery_days)
        shipping_cost = round(float(np.clip((sales * 0.04) + (quantity * 1.5) + (delivery_days * 0.8), 3.5, 95.0)), 2)

        # Customer Satisfaction (1 to 5 stars) - degrades with delivery delays & returns
        base_csat = np.random.choice([4, 5], p=[0.4, 0.6])
        if delivery_days >= 8:
            base_csat -= random.choice([2, 3])
        elif delivery_days >= 6:
            base_csat -= random.choice([1, 2])
        csat = int(np.clip(base_csat, 1, 5))

        # Return Rate: Furniture has highest return rate, Technology medium, Office supplies low
        return_prob = {"Furniture": 0.19, "Technology": 0.13, "Office Supplies": 0.07}[category]
        if csat <= 2:
            return_prob += 0.25
        returned = "Yes" if random.random() < return_prob else "No"
        if returned == "Yes" and csat > 3:
            csat = random.choice([1, 2, 3])

        # Missing values for realistic data cleaning requirements (~3.5% missing)
        cust_tenure = cust_profile["tenure"] if random.random() > 0.035 else np.nan
        final_csat = csat if random.random() > 0.04 else np.nan

        records.append({
            "Order_ID": order_id,
            "Customer_ID": cid,
            "Order_Date": order_date.strftime("%Y-%m-%d"),
            "Ship_Date": ship_date.strftime("%Y-%m-%d"),
            "Delivery_Days": delivery_days,
            "Customer_Segment": cust_profile["segment"],
            "Customer_Age": cust_profile["age"],
            "Customer_Tenure_Months": cust_tenure,
            "Region": region,
            "Category": category,
            "Sub_Category": sub_category,
            "Order_Priority": priority,
            "Payment_Method": payment,
            "Sales": sales,
            "Quantity": quantity,
            "Discount": discount,
            "Profit": profit,
            "Shipping_Cost": shipping_cost,
            "Customer_Satisfaction_Score": final_csat,
            "Returned": returned
        })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"E-Commerce dataset generated successfully: {output_path} ({len(df)} records, {len(df.columns)} columns)")
    return df

if __name__ == "__main__":
    generate_ecommerce_dataset()
