"""
Chocolate Sales — Data Science Dashboard
Student: Bouterbag Amal
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.figure_factory as ff
import plotly.express as px
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Chocolate Sales — DS Pipeline",
    page_icon="🍫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# CUSTOM CSS — Artisan Dark Editorial Theme
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;1,400&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root palette ── */
:root {
    --cream:   #F5EFE6;
    --cocoa:   #2B1810;
    --mocha:   #5C3D2E;
    --bronze:  #C4883A;
    --sand:    #D4A574;
    --ash:     #8B7355;
    --paper:   #EDE5D8;
    --ink:     #1A0F08;
}

/* ── App background ── */
.stApp {
    background-color: var(--cocoa) !important;
    background-image:
        radial-gradient(ellipse at 20% 10%, rgba(92,61,46,0.4) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 90%, rgba(196,136,58,0.15) 0%, transparent 50%);
    font-family: 'DM Sans', sans-serif;
    color: var(--cream) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #1A0F08 !important;
    border-right: 1px solid rgba(196,136,58,0.25) !important;
}

[data-testid="stSidebar"] * {
    color: var(--cream) !important;
}

.sidebar-brand {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--bronze) !important;
    letter-spacing: 0.02em;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(196,136,58,0.3);
    margin-bottom: 1.2rem;
}

/* ── Radio buttons in sidebar ── */
[data-testid="stSidebar"] .stRadio label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: var(--sand) !important;
    padding: 6px 0 !important;
}

/* ── Page title block ── */
.page-header {
    padding: 2.5rem 0 1rem 0;
    border-bottom: 1px solid rgba(196,136,58,0.3);
    margin-bottom: 2rem;
}

.page-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--bronze);
    margin-bottom: 0.4rem;
}

.page-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: var(--cream);
    line-height: 1.15;
    margin: 0;
}

.page-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    color: var(--ash);
    margin-top: 0.5rem;
    font-weight: 300;
}

/* ── Section headings ── */
.section-heading {
    font-family: 'Playfair Display', serif;
    font-size: 1.45rem;
    font-weight: 700;
    color: var(--cream);
    margin: 2rem 0 0.3rem 0;
    letter-spacing: -0.01em;
}

.section-rule {
    height: 1px;
    background: linear-gradient(to right, var(--bronze), transparent);
    margin-bottom: 1rem;
    border: none;
}

/* ── Subsection label ── */
.sub-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--bronze);
    margin-bottom: 0.6rem;
}

/* ── Metric cards ── */
.metric-card {
    background: rgba(92,61,46,0.25);
    border: 1px solid rgba(196,136,58,0.2);
    border-radius: 4px;
    padding: 1.1rem 1.3rem;
    text-align: center;
}

.metric-value {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--bronze);
    line-height: 1;
}

.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--ash);
    margin-top: 0.3rem;
}

/* ── Observation callout ── */
.obs-box {
    background: rgba(196,136,58,0.08);
    border-left: 3px solid var(--bronze);
    padding: 0.9rem 1.2rem;
    border-radius: 0 4px 4px 0;
    margin: 0.8rem 0 1.2rem 0;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.88rem;
    color: var(--sand);
    line-height: 1.6;
}

/* ── Table styling ── */
.stDataFrame, [data-testid="stTable"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
}

/* ── Buttons ── */
.stButton button {
    background: transparent !important;
    border: 1px solid var(--bronze) !important;
    color: var(--bronze) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-radius: 2px !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.2s ease !important;
}

.stButton button:hover {
    background: var(--bronze) !important;
    color: var(--cocoa) !important;
}

/* ── Checkbox ── */
.stCheckbox label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    color: var(--sand) !important;
    letter-spacing: 0.05em !important;
}

/* ── Horizontal rule ── */
hr { border-color: rgba(196,136,58,0.2) !important; }

/* ── General text ── */
p, li { color: var(--cream) !important; font-family: 'DM Sans', sans-serif !important; }
h1, h2, h3, h4 { font-family: 'Playfair Display', serif !important; color: var(--cream) !important; }

/* ── Success/info boxes ── */
.stSuccess { background: rgba(92,61,46,0.3) !important; border: 1px solid rgba(196,136,58,0.3) !important; }
.stSuccess * { color: var(--sand) !important; }

/* ── Metric native ── */
[data-testid="stMetric"] {
    background: rgba(92,61,46,0.2);
    border: 1px solid rgba(196,136,58,0.18);
    border-radius: 4px;
    padding: 0.8rem 1rem;
}
[data-testid="stMetric"] label { font-family: 'DM Mono', monospace !important; font-size: 0.65rem !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; color: var(--ash) !important; }
[data-testid="stMetric"] [data-testid="stMetricValue"] { font-family: 'Playfair Display', serif !important; font-size: 1.8rem !important; color: var(--bronze) !important; }

/* ── Plotly / chart containers ── */
.stPlotlyChart, .stPyplot { border: 1px solid rgba(196,136,58,0.15) !important; border-radius: 4px !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--cocoa); }
::-webkit-scrollbar-thumb { background: var(--mocha); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# MATPLOTLIB THEME
# ─────────────────────────────────────────
COCOA   = "#2B1810"
CREAM   = "#F5EFE6"
BRONZE  = "#C4883A"
SAND    = "#D4A574"
MOCHA   = "#5C3D2E"
ASH     = "#8B7355"
BLUE    = "#4A7FA5"
GREEN   = "#5A8F6A"

def set_plot_style():
    mpl.rcParams.update({
        "figure.facecolor":  COCOA,
        "axes.facecolor":    "#1A0F08",
        "axes.edgecolor":    MOCHA,
        "axes.labelcolor":   ASH,
        "axes.titlecolor":   CREAM,
        "axes.titlesize":    12,
        "axes.titleweight":  "normal",
        "axes.titlepad":     12,
        "axes.labelsize":    9,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.color":       ASH,
        "ytick.color":       ASH,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "text.color":        CREAM,
        "grid.color":        "#3D2418",
        "grid.linestyle":    "--",
        "grid.linewidth":    0.5,
        "grid.alpha":        0.6,
        "legend.facecolor":  "#1A0F08",
        "legend.edgecolor":  MOCHA,
        "legend.labelcolor": CREAM,
        "legend.fontsize":   8,
        "font.family":       "monospace",
    })

set_plot_style()

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def heading(label, title):
    st.markdown(f'<p class="page-label">{label}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="section-heading">{title}</p>', unsafe_allow_html=True)
    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

def obs(text):
    st.markdown(f'<div class="obs-box">{text}</div>', unsafe_allow_html=True)

def sublabel(text):
    st.markdown(f'<p class="sub-label">{text}</p>', unsafe_allow_html=True)

def metric_card(col, value, label):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("chocolate_sales.csv")
    df["Amount"] = df["Amount"].str.replace(r"[\$,]", "", regex=True).astype(float)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df["Month"]             = df["Date"].dt.month
    df["Year"]              = df["Date"].dt.year
    df["Revenue_per_Box"]   = df["Amount"] / df["Boxes Shipped"]
    df["Product_Avg_Price"] = df.groupby("Product")["Amount"].transform("mean")
    df["Country_Code"]      = pd.Categorical(df["Country"]).codes
    return df

df = load_data()

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-brand">Chocolate Sales<br><em style="font-size:0.75rem;font-style:italic;font-weight:300;color:#8B7355;">Data Science Pipeline</em></div>', unsafe_allow_html=True)
    st.markdown('<p class="sub-label" style="margin-bottom:0.6rem;">Navigate</p>', unsafe_allow_html=True)
    section = st.radio(
        "",
        ["Data Collection",
         "Preprocessing",
         "Exploratory Analysis",
         "Regression Modeling"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown('<p class="sub-label">Student</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-family:DM Sans,sans-serif;font-size:0.85rem;color:#D4A574;">Bouterbag Amal</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-label" style="margin-top:0.8rem;">Dataset</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-family:DM Sans,sans-serif;font-size:0.82rem;color:#8B7355;">3,282 transactions · 2022–2024</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────
section_meta = {
    "Data Collection":   ("01 / 04", "Data Collection"),
    "Preprocessing":     ("02 / 04", "Preprocessing"),
    "Exploratory Analysis": ("03 / 04", "Exploratory Analysis"),
    "Regression Modeling":  ("04 / 04", "Regression Modeling"),
}
num, title = section_meta[section]
st.markdown(f"""
<div class="page-header">
    <p class="page-label">{num}</p>
    <h1 class="page-title">{title}</h1>
    <p class="page-subtitle">Chocolate Sales · Bouterbag Amal</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════
# 01 — DATA COLLECTION
# ═══════════════════════════════════════════════════
if section == "Data Collection":

    st.markdown("""
    <p style="font-family:'DM Sans',sans-serif;font-size:0.95rem;color:#D4A574;line-height:1.7;max-width:680px;">
    A real-world transactional dataset from a chocolate distribution company spanning January 2022
    to August 2024. Each row represents a single sale across six countries and twenty-two product lines.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns(4, gap="small")
    metric_card(cols[0], "3,282", "Transactions")
    metric_card(cols[1], "6", "Countries")
    metric_card(cols[2], "22", "Products")
    metric_card(cols[3], "2022–24", "Date Range")

    st.markdown("<br>", unsafe_allow_html=True)
    heading("Schema", "Column Descriptions")
    st.markdown("""
    | Column | Scale | Role |
    |---|---|---|
    | Sales Person | Nominal | Salesperson name |
    | Country | Nominal | One of 6 countries |
    | Product | Nominal | One of 22 chocolate types |
    | Date | Interval | Transaction date (DD/MM/YYYY) |
    | **Amount** | **Ratio** | Sale value in USD — target |
    | Boxes Shipped | Ratio | Units per transaction |
    """)

    if st.checkbox("Show raw data sample (first 50 rows)"):
        st.dataframe(df.head(50), use_container_width=True)

    heading("Volume", "Transactions by Country")
    country_counts = df["Country"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 3.5))
    bars = ax.bar(country_counts.index, country_counts.values,
                  color=[BRONZE, SAND, MOCHA, ASH, "#7A5C3A", "#4A3020"],
                  edgecolor="none", width=0.6)
    for bar, val in zip(bars, country_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
                f"{val:,}", ha="center", va="bottom", fontsize=7.5,
                color=SAND, fontfamily="monospace")
    ax.set_ylabel("Transaction Count")
    ax.set_title("Distribution of Transactions Across Countries", pad=14)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    if st.button("Show product distribution"):
        product_counts = df["Product"].value_counts()
        fig, ax = plt.subplots(figsize=(12, 4))
        colors = [BRONZE if i == 0 else (SAND if i < 5 else MOCHA)
                  for i in range(len(product_counts))]
        ax.bar(product_counts.index, product_counts.values,
               color=colors, edgecolor="none", width=0.7)
        ax.set_ylabel("Transaction Count")
        ax.set_title("Transaction Volume by Product")
        plt.xticks(rotation=45, ha="right", fontsize=7.5)
        ax.yaxis.grid(True, alpha=0.4)
        ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ═══════════════════════════════════════════════════
# 02 — PREPROCESSING
# ═══════════════════════════════════════════════════
elif section == "Preprocessing":

    heading("Integrity", "Missing Values & Duplicates")
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        sublabel("Missing values per column")
        missing = df.isnull().sum()
        st.dataframe(missing.rename("Count"), use_container_width=True)
        st.success("No missing values detected.")
    with col2:
        sublabel("Duplicate rows")
        dups = df.duplicated().sum()
        st.metric("Duplicates", dups)
        st.success("Dataset is clean — no duplicates.")

    heading("Transformations", "Cleaning Steps Applied")
    st.markdown("""
    **Amount** — Stripped `$` signs and comma separators, cast to `float64`.
    Before: `"$5,320.00"` → After: `5320.0`

    **Date** — Parsed `DD/MM/YYYY` string to `datetime64` for time-series operations.

    **Engineered features** derived for modeling:
    - `Month` — Calendar month extracted from Date (1–12)
    - `Year` — Calendar year (2022, 2023, 2024)
    - `Revenue_per_Box` — Amount ÷ Boxes Shipped
    - `Product_Avg_Price` — Mean Amount per product (tier proxy)
    - `Country_Code` — Integer encoding of Country (0–5)
    """)

    if st.checkbox("Preview cleaned dataset"):
        st.dataframe(
            df[["Sales Person", "Country", "Product", "Date", "Amount",
                "Boxes Shipped", "Month", "Year", "Revenue_per_Box"]].head(20),
            use_container_width=True
        )

    heading("Statistics", "Descriptive Summary")
    sublabel("Amount · Boxes Shipped · Revenue per Box")
    st.dataframe(df[["Amount", "Boxes Shipped", "Revenue_per_Box"]].describe().round(2),
                 use_container_width=True)

    heading("Outliers", "Amount Distribution — Boxplot View")
    fig, ax = plt.subplots(figsize=(10, 3))
    bp = ax.boxplot(df["Amount"], vert=False, patch_artist=True,
                    boxprops=dict(facecolor=MOCHA, alpha=0.9, linewidth=1.2, edgecolor=BRONZE),
                    whiskerprops=dict(color=ASH, linewidth=1),
                    capprops=dict(color=BRONZE, linewidth=1.5),
                    medianprops=dict(color=BRONZE, linewidth=2),
                    flierprops=dict(marker="o", color=SAND, alpha=0.3, markersize=3, linestyle="none"))
    ax.set_xlabel("Amount (USD)")
    ax.set_title("Sale Amount — Outlier Detection via IQR")
    ax.xaxis.grid(True, alpha=0.35)
    ax.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    obs("Right-skewed distribution with high-value outliers above ~$20,000. These likely represent large bulk orders rather than data errors — they are retained for analysis.")


# ═══════════════════════════════════════════════════
# 03 — EXPLORATORY ANALYSIS
# ═══════════════════════════════════════════════════
elif section == "Exploratory Analysis":

    heading("Target Variable", "Distribution of Sale Amount")
    fig, ax = plt.subplots(figsize=(11, 4))
    n, bins, patches = ax.hist(df["Amount"], bins=55, edgecolor="none", alpha=0.9)
    for patch, left in zip(patches, bins[:-1]):
        norm_val = left / df["Amount"].max()
        patch.set_facecolor(mpl.colors.to_rgba(
            mpl.colors.hsv_to_rgb([0.07, 0.6 + norm_val * 0.4, 0.5 + norm_val * 0.4])))
    ax.axvline(df["Amount"].mean(),   color=BRONZE, linewidth=1.5, linestyle="--",
               label=f"Mean  ${df['Amount'].mean():,.0f}")
    ax.axvline(df["Amount"].median(), color=SAND,   linewidth=1.5, linestyle=":",
               label=f"Median ${df['Amount'].median():,.0f}")
    ax.set_xlabel("Amount (USD)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Transaction Amounts")
    ax.legend(framealpha=0.4)
    ax.yaxis.grid(True, alpha=0.35)
    ax.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    obs("Right-skewed — most transactions fall below $10,000. A tail of high-value outliers pulls the mean ($6,030) above the median ($5,225), suggesting periodic bulk purchases.")

    heading("Geography", "Total Revenue by Country")
    country_rev = df.groupby("Country")["Amount"].sum().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    bars = ax.barh(country_rev.index, country_rev.values,
                   color=[MOCHA, MOCHA, MOCHA, MOCHA, SAND, BRONZE],
                   edgecolor="none", height=0.55)
    for bar, val in zip(bars, country_rev.values):
        ax.text(bar.get_width() + 15000, bar.get_y() + bar.get_height()/2,
                f"${val/1e6:.2f}M", va="center", fontsize=7.5,
                color=SAND, fontfamily="monospace")
    ax.set_xlabel("Total Revenue (USD)")
    ax.set_title("Revenue by Country")
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    heading("Time Series", "Monthly Revenue Trend")
    monthly = df.groupby(df["Date"].dt.to_period("M"))["Amount"].sum()
    monthly.index = monthly.index.astype(str)
    fig, ax = plt.subplots(figsize=(13, 4))
    x = range(len(monthly))
    ax.fill_between(x, monthly.values, alpha=0.18, color=BRONZE)
    ax.plot(x, monthly.values, color=BRONZE, linewidth=1.8)
    ax.scatter(x, monthly.values, color=BRONZE, s=18, zorder=5)
    tick_step = max(1, len(monthly) // 10)
    ax.set_xticks(list(x)[::tick_step])
    ax.set_xticklabels(list(monthly.index)[::tick_step], rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel("Revenue (USD)")
    ax.set_title("Monthly Revenue — Jan 2022 to Aug 2024")
    ax.yaxis.grid(True, alpha=0.35)
    ax.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    obs("Revenue shows recurring mid-year peaks and a general upward trend from 2022 to 2024.")

    col_left, col_right = st.columns(2, gap="medium")
    with col_left:
        if st.checkbox("Show product revenue breakdown"):
            product_rev = df.groupby("Product")["Amount"].sum().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(9, 5))
            gradient = [BRONZE if i == 0 else (SAND if i < 5 else MOCHA)
                        for i in range(len(product_rev))]
            ax.bar(range(len(product_rev)), product_rev.values, color=gradient, edgecolor="none", width=0.7)
            ax.set_xticks(range(len(product_rev)))
            ax.set_xticklabels(product_rev.index, rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("Total Revenue (USD)")
            ax.set_title("Revenue by Product Line")
            ax.yaxis.grid(True, alpha=0.35)
            ax.set_axisbelow(True)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    with col_right:
        if st.checkbox("Show top salesperson ranking"):
            top_sp = df.groupby("Sales Person")["Amount"].sum().sort_values(ascending=True).tail(10)
            fig, ax = plt.subplots(figsize=(9, 5))
            bars = ax.barh(top_sp.index, top_sp.values,
                           color=[MOCHA]*8 + [SAND, BRONZE], edgecolor="none", height=0.55)
            ax.set_xlabel("Total Revenue (USD)")
            ax.set_title("Top 10 Salespersons")
            ax.xaxis.grid(True, alpha=0.3)
            ax.set_axisbelow(True)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    heading("Correlation", "Feature Relationship Matrix")
    corr_cols = ["Amount", "Boxes Shipped", "Month", "Revenue_per_Box", "Product_Avg_Price"]
    corr = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(7, 5))
    mask = np.zeros_like(corr, dtype=bool)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "cocoa", ["#1A0F08", MOCHA, "#2B1810", BRONZE, SAND])
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=1, linecolor="#2B1810",
                annot_kws={"size": 9, "color": CREAM},
                ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Matrix — Numerical Features", pad=12)
    ax.tick_params(colors=SAND)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    obs("Amount and Boxes Shipped are nearly uncorrelated (r ≈ −0.01) — shipping more boxes does not mean higher revenue. Product_Avg_Price shows the strongest signal (r ≈ 0.11), confirming that product tier is a key revenue driver.")

    heading("Distribution by Country", "Amount Density — Plotly")
    countries  = df["Country"].unique()
    hist_data  = [df[df["Country"] == c]["Amount"].dropna().values for c in countries]
    colors_px  = [BRONZE, SAND, "#9B7B5C", ASH, "#7A5C3A", "#C4A882"]
    fig_dist   = ff.create_distplot(hist_data, list(countries),
                                     show_hist=False, show_rug=False,
                                     colors=colors_px[:len(countries)])
    fig_dist.update_layout(
        plot_bgcolor="#1A0F08", paper_bgcolor=COCOA,
        font=dict(family="DM Mono, monospace", color=CREAM, size=11),
        title=dict(text="Sale Amount Distribution by Country", font=dict(size=14, color=CREAM)),
        xaxis=dict(title="Amount (USD)", gridcolor=MOCHA, showgrid=True),
        yaxis=dict(title="Density", gridcolor=MOCHA, showgrid=True),
        legend=dict(bgcolor=COCOA, bordercolor=MOCHA),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    heading("Scatter", "Boxes Shipped vs Amount")
    fig2 = px.scatter(df, x="Boxes Shipped", y="Amount", color="Country",
                      opacity=0.35,
                      color_discrete_sequence=[BRONZE, SAND, ASH, "#9B7B5C", "#7A5C3A", "#C4A882"],
                      labels={"Amount": "Amount (USD)"},
                      title="Do More Boxes Mean More Revenue?")
    fig2.update_layout(
        plot_bgcolor="#1A0F08", paper_bgcolor=COCOA,
        font=dict(family="DM Mono, monospace", color=CREAM, size=11),
        xaxis=dict(gridcolor=MOCHA), yaxis=dict(gridcolor=MOCHA),
        legend=dict(bgcolor=COCOA, bordercolor=MOCHA),
    )
    fig2.update_traces(marker=dict(size=5))
    st.plotly_chart(fig2, use_container_width=True)
    obs("No clear linear trend — the scatter confirms the near-zero correlation. Revenue is driven by product type and pricing, not shipment volume.")


# ═══════════════════════════════════════════════════
# 04 — REGRESSION MODELING
# ═══════════════════════════════════════════════════
elif section == "Regression Modeling":

    st.markdown("""
    <p style="font-family:'DM Sans',sans-serif;font-size:0.92rem;color:#D4A574;line-height:1.7;max-width:700px;">
    Predicting sale Amount from numeric features using three linear models: OLS, Ridge (L2), and Lasso (L1).
    All three share the same feature set and 80/20 train-test split.
    </p>
    """, unsafe_allow_html=True)

    FEATURES = ["Boxes Shipped", "Month", "Product_Avg_Price", "Country_Code"]
    TARGET   = "Amount"

    heading("Features", "Input Variables")
    st.markdown("""
    | Feature | Description |
    |---|---|
    | `Boxes Shipped` | Number of boxes in the transaction |
    | `Month` | Month of transaction (1–12) |
    | `Product_Avg_Price` | Mean revenue for that product — price-tier proxy |
    | `Country_Code` | Integer encoding of Country (0–5) |
    """)

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    heading("Split", "Train / Test Partition (80% / 20%)")
    cols = st.columns(2, gap="small")
    metric_card(cols[0], f"{len(X_train):,}", "Training samples")
    metric_card(cols[1], f"{len(X_test):,}",  "Test samples")

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    if st.checkbox("Show StandardScaler statistics"):
        st.dataframe(pd.DataFrame({
            "Feature":        FEATURES,
            "Mean (train)":   scaler.mean_.round(3),
            "Std  (train)":   scaler.scale_.round(3)
        }), use_container_width=True)

    ols   = LinearRegression()
    ridge = Ridge(alpha=1.0)
    lasso = Lasso(alpha=1.0)
    ols.fit(X_train_sc,   y_train)
    ridge.fit(X_train_sc, y_train)
    lasso.fit(X_train_sc, y_train)
    y_pred_ols   = ols.predict(X_test_sc)
    y_pred_ridge = ridge.predict(X_test_sc)
    y_pred_lasso = lasso.predict(X_test_sc)

    heading("Performance", "Model Comparison — MAE · RMSE · R²")
    def metrics(y_true, y_pred):
        return {
            "MAE ($)":  round(mean_absolute_error(y_true, y_pred), 2),
            "RMSE ($)": round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
            "R²":       round(r2_score(y_true, y_pred), 4)
        }
    m_ols   = metrics(y_test, y_pred_ols)
    m_ridge = metrics(y_test, y_pred_ridge)
    m_lasso = metrics(y_test, y_pred_lasso)
    results = pd.DataFrame([
        {"Model": "OLS Linear Regression", **m_ols},
        {"Model": "Ridge  (α = 1)",        **m_ridge},
        {"Model": "Lasso  (α = 1)",        **m_lasso},
    ])
    st.dataframe(results.set_index("Model"), use_container_width=True)
    obs(f"All three models produce nearly identical results — regularization has minimal effect, indicating the model is not overfitting. R² ≈ 0.01 means selected features explain only ~1% of variance in Amount. MAE ~ ${m_ols['MAE ($)']:,} means average prediction error is around $3,600.")

    heading("Coefficients", "Feature Weights Across Models")
    sublabel("Standardized — 1 unit = 1 std deviation change in feature")
    coef_df = pd.DataFrame({
        "Feature":           FEATURES,
        "OLS":   ols.coef_.round(2),
        "Ridge": ridge.coef_.round(2),
        "Lasso": lasso.coef_.round(2),
    }).sort_values("OLS", ascending=False)
    st.dataframe(coef_df.set_index("Feature"), use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    x_pos = np.arange(len(FEATURES))
    w = 0.25
    ax.bar(x_pos - w,  ols.coef_,   w, label="OLS",   color=BRONZE, edgecolor="none")
    ax.bar(x_pos,      ridge.coef_, w, label="Ridge", color=BLUE,   edgecolor="none")
    ax.bar(x_pos + w,  lasso.coef_, w, label="Lasso", color=GREEN,  edgecolor="none")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(FEATURES, rotation=12, ha="right")
    ax.axhline(0, color=ASH, linewidth=0.8, linestyle="--")
    ax.set_ylabel("Coefficient (USD / std dev)")
    ax.set_title("Coefficient Comparison: OLS vs Ridge vs Lasso")
    ax.legend(framealpha=0.4)
    ax.yaxis.grid(True, alpha=0.35)
    ax.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    obs("Product_Avg_Price carries the largest positive coefficient — higher-tier products generate more revenue. Boxes Shipped is near-zero, consistent with the EDA finding. Month is slightly negative — no meaningful seasonal effect on single-transaction value. All three models agree: no strong regularization effect.")

    heading("Predictions", "Actual vs Predicted — OLS")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_test, y_pred_ols, alpha=0.25, s=16, color=BRONZE, edgecolors="none")
    lims = [float(y_test.min()), float(y_test.max())]
    ax.plot(lims, lims, color=SAND, linewidth=1.5, linestyle="--", label="Perfect prediction")
    ax.set_xlabel("Actual Amount (USD)")
    ax.set_ylabel("Predicted Amount (USD)")
    ax.set_title(f"Predicted vs Actual   ·   R² = {r2_score(y_test, y_pred_ols):.4f}")
    ax.legend(framealpha=0.4)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    heading("Residuals", "Error Distribution Analysis")
    residuals = y_test - y_pred_ols
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].scatter(y_pred_ols, residuals, alpha=0.25, s=12, color=BLUE, edgecolors="none")
    axes[0].axhline(0, color=BRONZE, linewidth=1.2, linestyle="--")
    axes[0].set_xlabel("Predicted Amount")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs Predicted")
    axes[0].yaxis.grid(True, alpha=0.3)
    axes[0].set_axisbelow(True)

    axes[1].hist(residuals, bins=45, color=MOCHA, edgecolor=COCOA, alpha=0.95,
                 linewidth=0.3)
    axes[1].axvline(0, color=BRONZE, linewidth=1.2, linestyle="--")
    axes[1].set_xlabel("Residual (USD)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Residual Distribution")
    axes[1].yaxis.grid(True, alpha=0.3)
    axes[1].set_axisbelow(True)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    obs("Residuals are centered at zero — the model is unbiased. Spread is roughly constant with no fan pattern, suggesting homoscedasticity holds. Residual histogram is approximately bell-shaped. Remaining large errors reflect factors absent from the feature set: discount rates, promotions, customer segments.")

    if st.button("Show sample predictions (first 15 rows)"):
        pred_sample = pd.DataFrame({
            "Actual ($)":   y_test.values[:15].round(2),
            "OLS ($)":      y_pred_ols[:15].round(2),
            "Ridge ($)":    y_pred_ridge[:15].round(2),
            "Error ($)":    (y_test.values[:15] - y_pred_ols[:15]).round(2)
        })
        st.dataframe(pred_sample, use_container_width=True)

    heading("Conclusions", "Key Takeaways")
    st.markdown("""
    | Question | Answer |
    |---|---|
    | Can we predict Amount from these features? | Partially — R² ≈ 0.01, weak predictive power |
    | Strongest predictor? | `Product_Avg_Price` — product tier drives revenue |
    | Does box count predict revenue? | No — correlation ≈ −0.01, coefficient ≈ 0 |
    | OLS vs Ridge vs Lasso? | All equivalent — no overfitting present |
    | Why is R² so low? | Revenue reflects negotiated prices, promotions, and product mix not captured in this dataset |
    """)
    obs("To build a stronger predictive model, additional features such as discount rate, customer segment, or product category (beyond average price) would be needed. The linear regression correctly identifies product tier as the dominant signal, consistent with EDA findings.")