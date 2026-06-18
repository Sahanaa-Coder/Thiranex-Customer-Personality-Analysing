"""
Customer Personality Analysis: Data Cleaning, EDA & Business Insights
======================================================================
Author      : Sahanaa
Project     : CodeAlpha Data Science Internship — Portfolio Project
Dataset     : Marketing Campaign (2240 customers, 29 features)
Description : End-to-end pipeline covering data cleaning, EDA,
              professional visualizations, business insights, and
              an automated data-quality report.
"""

import os
import shutil
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from datetime import datetime

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
VIZ_DIR    = os.path.join(BASE_DIR, "visualizations")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VIZ_DIR,    exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

RAW_PATH     = os.path.join(DATA_DIR, "raw_dataset.csv")
CLEANED_PATH = os.path.join(DATA_DIR, "cleaned_dataset.csv")

# ─── Global Style ─────────────────────────────────────────────────────────────
PALETTE  = ["#4361EE", "#F72585", "#4CC9F0", "#7209B7", "#3A0CA3",
            "#4895EF", "#560BAD", "#F3722C", "#90BE6D", "#F9C74F"]
BG_COLOR = "#0F0F1A"
CARD_COLOR = "#1A1A2E"
TEXT_COLOR = "#E0E0FF"
ACCENT     = "#4361EE"

def set_dark_style():
    plt.rcParams.update({
        "figure.facecolor":  BG_COLOR,
        "axes.facecolor":    CARD_COLOR,
        "axes.edgecolor":    "#2A2A4A",
        "axes.labelcolor":   TEXT_COLOR,
        "axes.titlecolor":   TEXT_COLOR,
        "axes.titlesize":    14,
        "axes.labelsize":    11,
        "xtick.color":       TEXT_COLOR,
        "ytick.color":       TEXT_COLOR,
        "text.color":        TEXT_COLOR,
        "grid.color":        "#2A2A4A",
        "grid.linestyle":    "--",
        "grid.alpha":        0.5,
        "font.family":       "DejaVu Sans",
        "legend.facecolor":  CARD_COLOR,
        "legend.edgecolor":  "#2A2A4A",
        "legend.labelcolor": TEXT_COLOR,
    })

set_dark_style()


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def locate_raw_dataset() -> str:
    candidates = [
        RAW_PATH,
        os.path.join(BASE_DIR, "Customer Personality prediction", "data", "raw_dataset.csv"),
        os.path.join(BASE_DIR, "Customer personality dataset", "marketing_campaign.csv"),
        os.path.join(BASE_DIR, "data", "marketing_campaign.csv"),
        os.path.join(BASE_DIR, "data", "raw_dataset.csv"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "Raw dataset not found. Please add 'raw_dataset.csv' to the project data folder "
        "or place 'marketing_campaign.csv' inside 'Customer personality dataset' or the project root."
    )


def load_data(path: str) -> pd.DataFrame:
    """Load raw CSV; auto-detects tab vs comma separator."""
    path = path if os.path.exists(path) else locate_raw_dataset()
    if path != RAW_PATH and not os.path.exists(RAW_PATH):
        try:
            shutil.copy2(path, RAW_PATH)
            print(f"[✓] Copied raw dataset into project data folder: {os.path.relpath(RAW_PATH, BASE_DIR)}")
            path = RAW_PATH
        except Exception as exc:
            print(f"[!] Warning: could not copy raw dataset to project data folder: {exc}")
    try:
        df = pd.read_csv(path, sep="\t")
        if df.shape[1] == 1:
            df = pd.read_csv(path, sep=",")
        print(f"[✓] Loaded {df.shape[0]:,} rows × {df.shape[1]} columns from {os.path.basename(path)}")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load data: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DATA CLEANING
# ══════════════════════════════════════════════════════════════════════════════

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, strip spaces, replace spaces with underscores."""
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(r"\s+", "_", regex=True)
                  .str.replace(r"[^a-z0-9_]", "", regex=True)
    )
    print(f"[✓] Standardized {len(df.columns)} column names")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill or drop missing values per column type."""
    before = df.isnull().sum().sum()
    # Income: impute with median (robust to outliers)
    if "income" in df.columns:
        median_inc = df["income"].median()
        df["income"].fillna(median_inc, inplace=True)
    # Generic fallback: numeric → median, categorical → mode
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    after = df.isnull().sum().sum()
    print(f"[✓] Missing values: {before} → {after}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    print(f"[✓] Duplicates removed: {before - len(df)}")
    return df


def fix_inconsistent_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise known inconsistent categorical values."""
    if "marital_status" in df.columns:
        replace_map = {
            "Alone":  "Single",
            "Absurd": "Single",
            "YOLO":   "Single",
        }
        df["marital_status"] = df["marital_status"].replace(replace_map)
    if "education" in df.columns:
        df["education"] = df["education"].replace({"2n Cycle": "2nd Cycle"})
    print("[✓] Inconsistent category values fixed")
    return df


def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates and compute derived age/tenure columns."""
    if "dt_customer" in df.columns:
        df["dt_customer"] = pd.to_datetime(df["dt_customer"], dayfirst=True, errors="coerce")
        ref_date = pd.Timestamp("2015-01-01")
        df["customer_tenure_days"] = (ref_date - df["dt_customer"]).dt.days
    if "year_birth" in df.columns:
        df["age"] = 2015 - df["year_birth"]
    print("[✓] Data types converted; derived columns added")
    return df


def treat_outliers_iqr(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Cap outliers at IQR fences (Winsorisation)."""
    for col in cols:
        if col not in df.columns:
            continue
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        n_out = ((df[col] < lo) | (df[col] > hi)).sum()
        df[col] = df[col].clip(lo, hi)
        print(f"   • {col}: {n_out} outliers capped")
    return df


def treat_outliers_zscore(df: pd.DataFrame, col: str, threshold: float = 3.5) -> pd.DataFrame:
    """Remove rows with extreme Z-scores for a single critical column."""
    if col not in df.columns:
        return df
    z = np.abs(stats.zscore(df[col].dropna()))
    before = len(df)
    df = df[np.abs(stats.zscore(df[col].fillna(df[col].median()))) < threshold]
    print(f"[✓] Z-score filter on '{col}': {before - len(df)} rows removed")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create business-relevant derived features."""
    spend_cols = [c for c in ["mntWines","mntFruits","mntMeatProducts",
                               "mntFishProducts","mntSweetProducts","mntGoldProds"]
                  if c.lower() in df.columns]
    spend_cols_lower = [c.lower() for c in spend_cols]
    existing_spend = [c for c in spend_cols_lower if c in df.columns]

    if existing_spend:
        df["total_spend"] = df[existing_spend].sum(axis=1)

    purchase_cols = [c for c in df.columns if "numpurchases" in c or "num" in c and "purchases" in c]
    if purchase_cols:
        df["total_purchases"] = df[purchase_cols].sum(axis=1)

    campaign_cols = [c for c in df.columns if c.startswith("acceptedcmp")]
    if campaign_cols:
        df["total_campaigns_accepted"] = df[campaign_cols].sum(axis=1)

    if "kidhome" in df.columns and "teenhome" in df.columns:
        df["total_children"] = df["kidhome"] + df["teenhome"]

    if "total_spend" in df.columns and "income" in df.columns:
        df["spend_income_ratio"] = (df["total_spend"] / df["income"].replace(0, np.nan)).round(4)

    print("[✓] Feature engineering complete")
    return df


def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline in sequence."""
    print("\n" + "="*60)
    print("  DATA CLEANING PIPELINE")
    print("="*60)
    df = standardize_column_names(df)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = fix_inconsistent_categories(df)
    df = convert_data_types(df)
    df = treat_outliers_zscore(df, "income", threshold=3.5)
    outlier_targets = ["income", "age", "total_spend"] if "total_spend" in df.columns else ["income"]
    df = treat_outliers_iqr(df, outlier_targets)
    df = engineer_features(df)
    # Drop low-value constant columns
    for col in ["z_costcontact", "z_revenue"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    print(f"\n[✓] Final cleaned shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print("="*60 + "\n")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3.  EDA — SUMMARY & CORRELATION
# ══════════════════════════════════════════════════════════════════════════════

def print_eda_summary(df: pd.DataFrame):
    print("\n" + "="*60)
    print("  EDA SUMMARY STATISTICS")
    print("="*60)
    numeric_df = df.select_dtypes(include="number")
    print(numeric_df.describe().round(2).to_string())
    print("\nTop Correlations with Total Spend:")
    if "total_spend" in df.columns:
        corr = numeric_df.corr()["total_spend"].abs().sort_values(ascending=False)
        print(corr.head(10).to_string())
    print("="*60 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════

def save_fig(name: str):
    path = os.path.join(VIZ_DIR, f"{name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"  [saved] {name}.png")


def plot_histogram(df: pd.DataFrame):
    """Age & Income distribution histograms."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Customer Age & Income Distribution", fontsize=16, fontweight="bold",
                 color=TEXT_COLOR, y=1.02)
    for ax, col, color, label in zip(
        axes,
        ["age", "income"],
        [PALETTE[0], PALETTE[1]],
        ["Age (years)", "Annual Income (USD)"]
    ):
        if col in df.columns:
            ax.hist(df[col].dropna(), bins=35, color=color, alpha=0.85, edgecolor="white", linewidth=0.3)
            ax.axvline(df[col].median(), color="#F9C74F", linewidth=2, linestyle="--", label=f"Median {df[col].median():.0f}")
            ax.set_xlabel(label)
            ax.set_ylabel("Count")
            ax.set_title(f"{col.replace('_',' ').title()} Distribution")
            ax.legend()
            ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig("histogram")


def plot_boxplot(df: pd.DataFrame):
    """Box plots for spend categories."""
    spend_cols = [c for c in ["mntwines","mntfruits","mntmeatproducts",
                               "mntfishproducts","mntsweetproducts","mntgoldprods"]
                  if c in df.columns]
    if not spend_cols:
        return
    fig, ax = plt.subplots(figsize=(13, 6))
    data = [df[c].dropna().values for c in spend_cols]
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="#F9C74F", linewidth=2))
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    for whisker in bp["whiskers"]:
        whisker.set_color(TEXT_COLOR)
    for cap in bp["caps"]:
        cap.set_color(TEXT_COLOR)
    for flier in bp["fliers"]:
        flier.set(marker="o", color="#F72585", alpha=0.4, markersize=4)
    labels = [c.replace("mnt","").replace("products","").replace("prods","").title()
              for c in spend_cols]
    ax.set_xticks(range(1, len(spend_cols) + 1))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title("Spending Distribution Across Product Categories", fontsize=14, fontweight="bold")
    ax.set_ylabel("Amount Spent (USD)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    save_fig("boxplot")


def plot_scatter(df: pd.DataFrame):
    """Income vs Total Spend scatter coloured by Education."""
    if not all(c in df.columns for c in ["income", "total_spend", "education"]):
        return
    fig, ax = plt.subplots(figsize=(11, 7))
    edu_cats = df["education"].unique()
    for i, edu in enumerate(edu_cats):
        mask = df["education"] == edu
        ax.scatter(df.loc[mask, "income"], df.loc[mask, "total_spend"],
                   color=PALETTE[i % len(PALETTE)], alpha=0.6, s=22, label=edu, edgecolors="none")
    # Regression line
    x, y = df["income"].dropna(), df["total_spend"].dropna()
    mask = x.notna() & y.notna()
    m, b, r, *_ = stats.linregress(x[mask], y[mask])
    xs = np.linspace(x.min(), x.max(), 200)
    ax.plot(xs, m * xs + b, color="#F9C74F", linewidth=2, linestyle="--",
            label=f"Trend (r²={r**2:.2f})")
    ax.set_xlabel("Annual Income (USD)")
    ax.set_ylabel("Total Spend (USD)")
    ax.set_title("Income vs. Total Spend  ·  Coloured by Education Level", fontsize=14, fontweight="bold")
    ax.legend(framealpha=0.7, fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig("scatter_income_vs_spend")


def plot_correlation_heatmap(df: pd.DataFrame):
    """Correlation heatmap of key numeric features."""
    key_cols = [c for c in ["income", "age", "total_spend", "total_purchases",
                              "total_campaigns_accepted", "recency",
                              "customer_tenure_days", "total_children",
                              "spend_income_ratio", "response"]
                if c in df.columns]
    corr = df[key_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
                annot=True, fmt=".2f", annot_kws={"size": 9, "color": "white"},
                linewidths=0.5, linecolor="#2A2A4A", ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap", fontsize=15, fontweight="bold", pad=15)
    ax.tick_params(axis="x", rotation=40, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)
    plt.tight_layout()
    save_fig("heatmap")


def plot_countplot(df: pd.DataFrame):
    """Count plots for Education and Marital Status."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col, title in zip(axes,
                               ["education", "marital_status"],
                               ["Education Level", "Marital Status"]):
        if col not in df.columns:
            continue
        order = df[col].value_counts().index
        counts = df[col].value_counts()
        bars = ax.bar(order, [counts[k] for k in order],
                      color=PALETTE[:len(order)], alpha=0.85, edgecolor="white", linewidth=0.4)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 10, f"{int(bar.get_height())}",
                    ha="center", va="bottom", fontsize=9, color=TEXT_COLOR)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel(col.replace("_"," ").title())
        ax.set_ylabel("Number of Customers")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    save_fig("countplot_categories")


def plot_pairplot(df: pd.DataFrame):
    """Pair plot of key numeric features."""
    pair_cols = [c for c in ["income","age","total_spend","recency","total_children"]
                 if c in df.columns]
    if len(pair_cols) < 3:
        return
    sample = df[pair_cols + (["education"] if "education" in df.columns else [])].dropna().sample(
        min(500, len(df)), random_state=42)
    hue = "education" if "education" in df.columns else None
    palette_dict = {edu: PALETTE[i % len(PALETTE)]
                    for i, edu in enumerate(sample[hue].unique())} if hue else None
    fig = sns.pairplot(sample, vars=pair_cols, hue=hue, palette=palette_dict,
                        plot_kws={"alpha": 0.5, "s": 15, "edgecolor": "none"},
                        diag_kws={"alpha": 0.7},
                        corner=True)
    fig.figure.suptitle("Pair Plot — Key Customer Features", y=1.02,
                         fontsize=14, fontweight="bold", color=TEXT_COLOR)
    fig.figure.patch.set_facecolor(BG_COLOR)
    for ax_row in fig.axes:
        for ax in ax_row:
            if ax:
                ax.set_facecolor(CARD_COLOR)
    fig.savefig(os.path.join(VIZ_DIR, "pairplot.png"), dpi=120,
                bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print("  [saved] pairplot.png")


def plot_bar_chart(df: pd.DataFrame):
    """Average total spend by education level."""
    if not all(c in df.columns for c in ["education", "total_spend"]):
        return
    agg = df.groupby("education")["total_spend"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(agg.index, agg.values, color=PALETTE[:len(agg)],
                  alpha=0.88, edgecolor="white", linewidth=0.5)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 15, f"${bar.get_height():,.0f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color=TEXT_COLOR)
    ax.set_title("Average Total Spend by Education Level", fontsize=14, fontweight="bold")
    ax.set_xlabel("Education Level")
    ax.set_ylabel("Average Spend (USD)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    save_fig("bar_spend_by_education")


def plot_pie_chart(df: pd.DataFrame):
    """Pie chart of marital status distribution."""
    if "marital_status" not in df.columns:
        return
    counts = df["marital_status"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 7))
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=counts.index,
        colors=PALETTE[:len(counts)], autopct="%1.1f%%",
        startangle=140, pctdistance=0.82,
        wedgeprops=dict(width=0.55, edgecolor=BG_COLOR, linewidth=2))
    for text in texts:
        text.set_color(TEXT_COLOR)
        text.set_fontsize(10)
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
        autotext.set_fontsize(9)
    ax.set_title("Customer Marital Status Distribution", fontsize=14, fontweight="bold",
                 color=TEXT_COLOR, pad=20)
    plt.tight_layout()
    save_fig("pie_marital_status")


def plot_violin(df: pd.DataFrame):
    """Violin plot: Income by Marital Status."""
    if not all(c in df.columns for c in ["marital_status", "income"]):
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    order = df.groupby("marital_status")["income"].median().sort_values(ascending=False).index
    parts = ax.violinplot(
        [df.loc[df["marital_status"] == cat, "income"].dropna().values for cat in order],
        positions=range(len(order)), showmedians=True, showmeans=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(PALETTE[i % len(PALETTE)])
        pc.set_alpha(0.75)
    parts["cmedians"].set_color("#F9C74F")
    parts["cmedians"].set_linewidth(2)
    for key in ["cbars","cmins","cmaxes"]:
        parts[key].set_color(TEXT_COLOR)
        parts[key].set_alpha(0.6)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=10)
    ax.set_title("Income Distribution by Marital Status", fontsize=14, fontweight="bold")
    ax.set_ylabel("Annual Income (USD)")
    ax.set_xlabel("Marital Status")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    save_fig("violin_income_by_marital")


def plot_missing_values(df_raw: pd.DataFrame):
    """Visualise missing values in the raw dataset."""
    missing = df_raw.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(missing.index, missing.values,
                   color=PALETTE[1], alpha=0.85, edgecolor="white", linewidth=0.4)
    for bar in bars:
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{int(bar.get_width())}", va="center", fontsize=10, color=TEXT_COLOR)
    ax.set_title("Missing Values per Column (Raw Data)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Count of Missing Values")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    save_fig("missing_values")


def plot_campaign_response(df: pd.DataFrame):
    """Campaign acceptance rates."""
    cmp_cols = [c for c in df.columns if c.startswith("acceptedcmp")]
    if not cmp_cols:
        return
    rates = {c.replace("acceptedcmp","Cmp "): df[c].mean() * 100 for c in cmp_cols}
    if "response" in df.columns:
        rates["Final Offer"] = df["response"].mean() * 100
    fig, ax = plt.subplots(figsize=(10, 5))
    keys, vals = list(rates.keys()), list(rates.values())
    bars = ax.bar(keys, vals, color=PALETTE[:len(keys)],
                  alpha=0.85, edgecolor="white", linewidth=0.4)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2, f"{bar.get_height():.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color=TEXT_COLOR)
    ax.set_title("Campaign Acceptance Rates (%)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Acceptance Rate (%)")
    ax.set_xlabel("Campaign")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    save_fig("bar_campaign_response")


def plot_spend_by_age_group(df: pd.DataFrame):
    """Average spend per age bracket."""
    if not all(c in df.columns for c in ["age", "total_spend"]):
        return
    bins   = [18, 30, 40, 50, 60, 70, 90]
    labels = ["18-29", "30-39", "40-49", "50-59", "60-69", "70+"]
    df2 = df.copy()
    df2["age_group"] = pd.cut(df2["age"], bins=bins, labels=labels, right=False)
    agg = df2.groupby("age_group", observed=True)["total_spend"].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(agg.index.astype(str), agg.values,
                  color=PALETTE[:len(agg)], alpha=0.85, edgecolor="white", linewidth=0.4)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 8, f"${bar.get_height():,.0f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color=TEXT_COLOR)
    ax.set_title("Average Total Spend by Age Group", fontsize=14, fontweight="bold")
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Average Spend (USD)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    save_fig("bar_spend_by_age_group")


def run_all_visualizations(df: pd.DataFrame, df_raw: pd.DataFrame):
    print("\n" + "="*60)
    print("  GENERATING VISUALIZATIONS")
    print("="*60)
    plot_missing_values(df_raw)
    plot_histogram(df)
    plot_boxplot(df)
    plot_scatter(df)
    plot_correlation_heatmap(df)
    plot_countplot(df)
    plot_pairplot(df)
    plot_bar_chart(df)
    plot_pie_chart(df)
    plot_violin(df)
    plot_campaign_response(df)
    plot_spend_by_age_group(df)
    print("[✓] All visualizations saved to /visualizations/\n")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  BUSINESS INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════

def generate_business_insights(df: pd.DataFrame) -> list[dict]:
    insights = []

    # ── Insight 1: High-Income → High Spend ──────────────────────────────────
    if all(c in df.columns for c in ["income", "total_spend"]):
        pair = df[["income", "total_spend"]].dropna()
        r, _ = stats.pearsonr(pair["income"], pair["total_spend"])
        insights.append({
            "id": 1,
            "title": "Income is the strongest predictor of total spend",
            "finding": f"Pearson correlation between Income and Total Spend = {r:.2f}.",
            "significance": "High-income customers are the primary revenue drivers.",
            "action": "Create a premium loyalty tier targeting customers with income > $70K; "
                      "personalise offers with higher-margin products."
        })

    # ── Insight 2: Education and Spending ─────────────────────────────────────
    if all(c in df.columns for c in ["education", "total_spend"]):
        edu_spend = df.groupby("education")["total_spend"].mean().sort_values(ascending=False)
        top_edu = edu_spend.index[0]
        insights.append({
            "id": 2,
            "title": f"'{top_edu}' customers spend the most on average",
            "finding": f"Avg spend by education: {edu_spend.to_dict()}",
            "significance": "Education proxies disposable income and brand affinity.",
            "action": f"Allocate higher ad-spend budgets towards '{top_edu}' segments on LinkedIn "
                      "and professional networks."
        })

    # ── Insight 3: Children reduce spend ──────────────────────────────────────
    if all(c in df.columns for c in ["total_children", "total_spend"]):
        corr = df["total_children"].corr(df["total_spend"])
        insights.append({
            "id": 3,
            "title": "Having children is negatively correlated with spending",
            "finding": f"Correlation between total children and spend = {corr:.2f}.",
            "significance": "Customers with children have less discretionary income.",
            "action": "For family segments, focus on value bundles and discount promotions "
                      "rather than premium products."
        })

    # ── Insight 4: Campaign success ───────────────────────────────────────────
    cmp_cols = [c for c in df.columns if c.startswith("acceptedcmp")]
    if cmp_cols:
        rates = {c: df[c].mean() for c in cmp_cols}
        best  = max(rates, key=rates.get)
        worst = min(rates, key=rates.get)
        insights.append({
            "id": 4,
            "title": "Campaign acceptance rates vary significantly across campaigns",
            "finding": f"Best: {best} ({rates[best]*100:.1f}%), Worst: {worst} ({rates[worst]*100:.1f}%).",
            "significance": "Creative and targeting quality drives measurable conversion differences.",
            "action": f"Replicate the messaging strategy of {best} in future campaigns; "
                      f"audit and redesign {worst}."
        })

    # ── Insight 5: Recency & Response ─────────────────────────────────────────
    if all(c in df.columns for c in ["recency", "response"]):
        resp_rec  = df[df["response"] == 1]["recency"].mean()
        nresp_rec = df[df["response"] == 0]["recency"].mean()
        insights.append({
            "id": 5,
            "title": "Responders to the final offer had lower recency scores",
            "finding": f"Avg recency — Responders: {resp_rec:.0f} days vs. Non-responders: {nresp_rec:.0f} days.",
            "significance": "Recently-active customers are more responsive to marketing.",
            "action": "Trigger time-sensitive campaigns to customers who last purchased < 30 days ago."
        })

    # ── Insight 6: Age group & spend ──────────────────────────────────────────
    if all(c in df.columns for c in ["age", "total_spend"]):
        bins   = [18, 30, 40, 50, 60, 70, 90]
        labels = ["18-29","30-39","40-49","50-59","60-69","70+"]
        df2 = df.copy()
        df2["age_group"] = pd.cut(df2["age"], bins=bins, labels=labels, right=False)
        agg = df2.groupby("age_group", observed=True)["total_spend"].mean()
        top_age = agg.idxmax()
        insights.append({
            "id": 6,
            "title": f"Customers aged {top_age} are the highest-spending age group",
            "finding": f"Average spend by age group: {agg.round(0).to_dict()}",
            "significance": "Middle-aged customers have peak disposable income.",
            "action": f"Focus acquisition and retention efforts on the {top_age} age bracket."
        })

    # ── Insight 7: Web purchases vs deals ─────────────────────────────────────
    web_col  = next((c for c in df.columns if "webpurchases" in c), None)
    deal_col = next((c for c in df.columns if "dealspurchases" in c), None)
    if web_col and deal_col:
        r2, _ = stats.pearsonr(df[web_col], df[deal_col])
        insights.append({
            "id": 7,
            "title": "Customers who buy via web are deal-sensitive",
            "finding": f"Correlation between web purchases and deal purchases = {r2:.2f}.",
            "significance": "Online channel customers are coupon/deal-driven.",
            "action": "Deploy targeted web-exclusive flash-sale notifications to online shoppers."
        })

    # ── Insight 8: Marital status & income ────────────────────────────────────
    if all(c in df.columns for c in ["marital_status", "income"]):
        ms_inc = df.groupby("marital_status")["income"].median().sort_values(ascending=False)
        insights.append({
            "id": 8,
            "title": "Married customers have higher median incomes",
            "finding": f"Median income by status (top 3): {ms_inc.head(3).round(0).to_dict()}",
            "significance": "Dual-income households command greater purchasing power.",
            "action": "Position premium couple / family product bundles for Married / Together segments."
        })

    # ── Insight 9: Multi-campaign responders ──────────────────────────────────
    if all(c in df.columns for c in ["total_campaigns_accepted", "total_spend"]):
        multi = df[df["total_campaigns_accepted"] >= 2]["total_spend"].mean()
        single = df[df["total_campaigns_accepted"] <= 1]["total_spend"].mean()
        insights.append({
            "id": 9,
            "title": "Customers accepting 2+ campaigns spend significantly more",
            "finding": f"Avg spend — multi-campaign: ${multi:,.0f} vs single: ${single:,.0f}.",
            "significance": "Campaign engagement is a strong proxy for customer lifetime value.",
            "action": "Build a 'campaign loyalty' score; fast-track multi-responders into VIP programmes."
        })

    # ── Insight 10: Tenure and spend ──────────────────────────────────────────
    if all(c in df.columns for c in ["customer_tenure_days", "total_spend"]):
        corr_t = df["customer_tenure_days"].corr(df["total_spend"])
        insights.append({
            "id": 10,
            "title": "Longer-tenured customers spend more in total",
            "finding": f"Tenure–Spend correlation = {corr_t:.2f}.",
            "significance": "Long-term retention directly drives revenue growth.",
            "action": "Invest in onboarding journeys and anniversary rewards to increase early-life "
                      "engagement and reduce first-year churn."
        })

    return insights


def print_insights(insights: list[dict]):
    print("\n" + "="*60)
    print("  BUSINESS INSIGHTS")
    print("="*60)
    for ins in insights:
        print(f"\n#{ins['id']:02d} — {ins['title']}")
        print(f"  Finding    : {ins['finding']}")
        print(f"  Significance: {ins['significance']}")
        print(f"  Action     : {ins['action']}")
    print("="*60 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  AUTOMATED DATA QUALITY REPORT (TXT)
# ══════════════════════════════════════════════════════════════════════════════

def generate_quality_report(df_raw: pd.DataFrame, df_clean: pd.DataFrame, insights: list[dict]):
    """Write a plain-text data quality + insights report."""
    path = os.path.join(REPORT_DIR, "data_quality_report.txt")
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "w") as f:
        def w(line=""): f.write(line + "\n")
        w("="*70)
        w("  CUSTOMER PERSONALITY ANALYSIS — DATA QUALITY & INSIGHTS REPORT")
        w(f"  Generated: {ts}")
        w("="*70)
        w()
        w("1. DATASET OVERVIEW")
        w("-"*50)
        w(f"  Raw  : {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
        w(f"  Clean: {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns")
        w(f"  Rows removed : {df_raw.shape[0] - df_clean.shape[0]}")
        w(f"  Columns added: {df_clean.shape[1] - df_raw.shape[1]}")
        w()
        w("2. MISSING VALUES (Raw)")
        w("-"*50)
        missing = df_raw.isnull().sum()
        missing_nonzero = missing[missing > 0]
        if missing_nonzero.empty:
            w("  No missing values detected in raw dataset.")
        else:
            for col, cnt in missing_nonzero.items():
                pct = cnt / len(df_raw) * 100
                w(f"  {col:<30} {cnt:>5}  ({pct:.1f}%)")
        w()
        w("3. DATA TYPES (Cleaned)")
        w("-"*50)
        for col, dtype in df_clean.dtypes.items():
            w(f"  {col:<35} {str(dtype)}")
        w()
        w("4. SUMMARY STATISTICS (Cleaned — Numeric)")
        w("-"*50)
        desc = df_clean.select_dtypes(include="number").describe().round(2)
        w(desc.to_string())
        w()
        w("5. BUSINESS INSIGHTS")
        w("-"*50)
        for ins in insights:
            w(f"\n  #{ins['id']:02d} {ins['title']}")
            w(f"  Finding    : {ins['finding']}")
            w(f"  Significance: {ins['significance']}")
            w(f"  Action     : {ins['action']}")
        w()
        w("="*70)
        w("  END OF REPORT")
        w("="*70)
    print(f"[✓] Data quality report saved to reports/data_quality_report.txt")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# 7.  FEATURE IMPORTANCE (Random Forest proxy)
# ══════════════════════════════════════════════════════════════════════════════

def feature_importance_analysis(df: pd.DataFrame):
    """Quick RF feature-importance for predicting campaign response."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
    except ImportError:
        print("[!] scikit-learn not available — skipping feature importance")
        return

    target = "response"
    if target not in df.columns:
        return

    df2 = df.copy()
    cat_cols = df2.select_dtypes(include="object").columns
    le = LabelEncoder()
    for c in cat_cols:
        df2[c] = le.fit_transform(df2[c].astype(str))

    drop_cols = ["id", "dt_customer", target]
    feat_cols = [c for c in df2.select_dtypes(include="number").columns
                 if c not in drop_cols]
    X = df2[feat_cols].fillna(0)
    y = df2[target]

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    imp = pd.Series(rf.feature_importances_, index=feat_cols).sort_values(ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(imp))]
    ax.barh(imp.index, imp.values, color=colors, alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.set_title("Feature Importance for Campaign Response Prediction\n(Random Forest)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    save_fig("feature_importance")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "█"*60)
    print("  CUSTOMER PERSONALITY ANALYSIS")
    print("  CodeAlpha Data Science Internship Project")
    print("█"*60)

    # 1. Load
    df_raw = load_data(RAW_PATH)

    # 2. Clean
    df = clean_pipeline(df_raw.copy())
    df.to_csv(CLEANED_PATH, index=False)
    print(f"[✓] Cleaned dataset saved → data/cleaned_dataset.csv")

    # 3. EDA
    print_eda_summary(df)

    # 4. Visualizations
    run_all_visualizations(df, df_raw)

    # 5. Feature Importance
    feature_importance_analysis(df)

    # 6. Business Insights
    insights = generate_business_insights(df)
    print_insights(insights)

    # 7. Quality Report
    generate_quality_report(df_raw, df, insights)

    print("\n" + "█"*60)
    print("  ALL OUTPUTS GENERATED SUCCESSFULLY")
    print(f"  Visualizations : {VIZ_DIR}")
    print(f"  Cleaned Data   : {CLEANED_PATH}")
    print(f"  Report         : {REPORT_DIR}/data_quality_report.txt")
    print("  Dashboard      : python dashboard/dashboard.py")
    print("█"*60 + "\n")


if __name__ == "__main__":
    main()
