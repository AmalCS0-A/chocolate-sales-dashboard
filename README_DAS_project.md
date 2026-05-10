# 🍫 Chocolate Sales — Data Science Project

**Student:** Bouterbag Amal  
**Course:** Statistical Modeling for Predictive Analysis  
**Dataset:** Chocolate Sales (3,282 transactions | 6 countries | 22 products | 2022–2024)

---

## 📌 Project Overview

This project applies a complete data science pipeline to a real-world chocolate sales dataset.
The goal is to explore the data, understand what drives sales revenue, and build a predictive model using Linear Regression.

**Target Variable:** `Amount` — the USD revenue generated per transaction.

---

## 🗂️ Repository Structure

```
├── chocolate_sales.csv           # Raw dataset
├── analysis.py                   # Full pipeline: cleaning → EDA → regression
├── app.py                        # Streamlit dashboard (interactive)
├── README.md                     # This file
├── plot_01_amount_distribution.png
├── plot_02_revenue_by_country.png
├── plot_03_revenue_by_product.png
├── plot_04_monthly_trend.png
├── plot_05_scatter_boxes_amount.png
├── plot_06_correlation_heatmap.png
├── plot_07_coefficients.png
├── plot_08_predicted_vs_actual.png
└── plot_09_residuals.png
```

---

## 🚀 How to Run

### Run the full analysis script
```bash
python analysis.py
```

### Launch the interactive Streamlit dashboard
```bash
pip install streamlit scikit-learn matplotlib seaborn plotly pandas numpy
streamlit run app.py
```

---

## 📦 Step 1 — Data Collection

The dataset was sourced from a chocolate distribution company and contains the following columns:

| Column | NOIR Scale | Type | Description |
|--------|-----------|------|-------------|
| Sales Person | Nominal | Qualitative | Name of the salesperson |
| Country | Nominal | Qualitative | One of 6 countries (Australia, UK, USA, India, New Zealand, Canada) |
| Product | Nominal | Qualitative | One of 22 chocolate product types |
| Date | Interval | Quantitative | Transaction date (DD/MM/YYYY) |
| **Amount** | **Ratio** | **Quantitative** | 💰 **Sale value in USD — TARGET variable** |
| Boxes Shipped | Ratio | Quantitative | Number of boxes per transaction |

---

## 🧹 Step 2 — Preprocessing

### Cleaning Steps
1. **Amount column**: Removed `$` signs and commas → converted to `float64`
   - Example: `"$5,320.00"` → `5320.0`
2. **Date column**: Parsed from `DD/MM/YYYY` string to `datetime64`
3. **No missing values** were found (confirmed with `df.isnull().sum()`)
4. **No duplicate rows** were found (confirmed with `df.duplicated().sum()`)

### Feature Engineering (for Regression)
| New Feature | How | Why |
|---|---|---|
| `Month` | `df["Date"].dt.month` | Captures seasonality |
| `Year` | `df["Date"].dt.year` | Captures year-over-year growth |
| `Revenue_per_Box` | `Amount / Boxes Shipped` | Price per box metric |
| `Product_Avg_Price` | Mean Amount grouped by Product | Proxy for product tier / value |
| `Country_Code` | Integer encoding of Country | Makes country usable in regression |

---

## 📊 Step 3 — Exploratory Data Analysis

### Key Findings

**Distribution of Amount:**
- The target variable `Amount` is **right-skewed** — most transactions fall below $10,000, but some high-value deals reach ~$26,000.
- Mean ≈ **$6,030** | Median ≈ **$5,225** — the gap confirms right skew.
- This means the average is pulled up by a few large orders; the "typical" transaction is around $5,000–6,000.

**Revenue by Country:**
- **Australia** leads in total revenue, followed by India and the UK.
- ANOVA testing (p = 0.49) confirmed that the **mean transaction amount does not significantly differ across countries** — the difference in total revenue is explained by volume, not price per transaction.

**Revenue by Product:**
- *Smooth Silky Salty* and *After Nines* generate the highest total revenue.
- Products have very different average prices — this became our most important feature.

**Boxes Shipped vs Amount:**
- Pearson correlation r ≈ **−0.013** (p = 0.451) → **no meaningful linear relationship**.
- Shipping more boxes does NOT predict higher revenue. This is the most important insight: price per box varies so much by product that box count is nearly useless as a predictor alone.

**Monthly Trend:**
- Revenue shows a general **upward trend** from 2022 to 2024 with seasonal fluctuations (peaks in mid-year months).

---

## 🤖 Step 4 — Linear Regression Modeling

### Features Used
```
X = ["Boxes Shipped", "Month", "Product_Avg_Price", "Country_Code"]
y = "Amount"
```

### Methodology
1. **Train/Test split**: 80% training (2,625 rows) / 20% test (657 rows) — `random_state=42`
2. **Feature scaling**: `StandardScaler` fitted on training data only (to avoid data leakage)
3. **Models trained**: OLS Linear Regression, Ridge (α=1), Lasso (α=1)
4. **Evaluation metrics**: MAE, RMSE, R²

### Results

| Model | MAE ($) | RMSE ($) | R² |
|-------|---------|---------|-----|
| OLS Linear Regression | ~$3,614 | ~$4,392 | ~0.010 |
| Ridge (α=1) | ~$3,614 | ~$4,392 | ~0.010 |
| Lasso (α=1) | ~$3,614 | ~$4,392 | ~0.010 |

### Interpretation of Results

**R² ≈ 0.01 — Why is it low?**

An R² of 0.01 means the four features used explain only about 1% of the variance in sale Amount. This is a meaningful finding, not a failure:
- As shown in the EDA, `Boxes Shipped` has essentially **zero correlation** with Amount.
- `Country` does not significantly affect the per-transaction amount (confirmed by ANOVA).
- `Month` has minimal seasonal effect on individual transaction prices.
- The only feature with any predictive signal is `Product_Avg_Price` (r ≈ 0.11).

The real drivers of Amount are likely **negotiated deal prices, customer loyalty discounts, and product-specific pricing** — variables not present in this dataset.

**Coefficients (standardized):**
- `Product_Avg_Price` → **largest positive coefficient** — confirms that product type/tier is the main price driver. Products with higher average prices generate more revenue per transaction.
- `Boxes Shipped` → coefficient ≈ 0 — confirms the EDA finding; boxes and revenue are unrelated.
- `Month` → slightly negative — no strong seasonal effect on individual transaction value.
- `Country_Code` → small positive — countries have marginal differences in transaction amounts.

**OLS vs Ridge vs Lasso:**
All three models produce identical results. This tells us:
1. The model is **not overfitting** — regularization has nothing to correct.
2. The low R² is genuinely due to missing features, not model complexity.

**Residual Analysis:**
- Residuals are centered around **mean ≈ $0** — the model is unbiased.
- The residual distribution is approximately **bell-shaped** — the normality assumption holds reasonably.
- No systematic fan shape in the residuals vs predicted plot — homoscedasticity is approximately met.
- Large residuals exist because Amount depends on factors outside our feature set.

### What would improve the model?
- Adding **product category** (premium vs standard) as a feature
- Adding **salesperson performance** as a feature
- Adding **discount rate** or **deal type** if available
- Using a non-linear model (Random Forest, XGBoost) to capture interaction effects

---

## 📝 Summary Table

| Step | Key Finding |
|------|------------|
| Data Collection | 3,282 transactions, Amount is the target (Ratio scale) |
| Preprocessing | No missing values, Amount cleaned from string to float |
| EDA | Amount is right-skewed; Boxes Shipped ≈ uncorrelated with Amount (r=−0.01) |
| Correlation | Product_Avg_Price has strongest correlation with Amount (r≈0.11) |
| Regression | R²≈0.01 — model is weak but correctly identifies product tier as main driver |
| Best Feature | `Product_Avg_Price` — the product type determines the sale value |
| OLS vs Ridge vs Lasso | All identical — no overfitting, missing features are the main issue |

---

## 🛠️ Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit
plotly
scipy
```

Install all: `pip install pandas numpy matplotlib seaborn scikit-learn streamlit plotly scipy`
