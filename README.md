# 🍬 Product Line Profitability & Margin Performance Analysis
### Nassau Candy Distributor — Data Analytics Portfolio Project

<br>

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?style=flat&logo=plotly&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=flat&logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-2ea44f?style=flat)

---

## 📌 Project Overview

This project delivers a comprehensive **product-line profitability and margin performance analysis** for Nassau Candy Distributor — one of the largest wholesale manufacturers and distributors of specialty confections in the United States.

Using transactional order-level data, the analysis moves beyond surface-level revenue metrics to uncover **which products and divisions truly drive profit**, which carry hidden margin risk, and where strategic intervention — through pricing, cost renegotiation, or portfolio rationalization — can have the greatest financial impact.

The project is delivered as two integrated outputs:
- A **structured EDA notebook** documenting the full analytical workflow
- An **interactive Streamlit dashboard** enabling real-time, filter-driven profitability exploration

---

## 🎯 Business Problem & Objective

In high-volume distribution businesses, **sales figures alone are misleading**. Some products generate strong revenue but operate on thin margins. Others consume disproportionate cost while contributing minimally to total profit. Without granular profitability visibility, decisions around pricing, promotions, and portfolio management remain reactive and intuition-based.

**This analysis aims to:**

- Identify which product lines deliver the highest gross margin
- Determine whether high-sales products are actually profitable
- Quantify profit concentration risk across the product portfolio
- Compare revenue and profit contributions across product divisions
- Flag cost-heavy, margin-poor products for strategic review
- Provide data-driven, actionable recommendations for portfolio and pricing decisions

---

## ✨ Key Features & Highlights

- ✅ **End-to-end analytical pipeline** — from raw data ingestion and cleaning through to business recommendations
- ✅ **Five-module interactive Streamlit dashboard** with real-time filtering by date, division, product, and margin threshold
- ✅ **Dual-level profitability analysis** — product-level rankings and division-level performance diagnostics
- ✅ **Pareto (80/20) concentration analysis** — identifies over-dependency risk in revenue and profit
- ✅ **Strategic quadrant classification** — products mapped as Stars, Volume Traps, Niche Opportunities, or Exit Candidates
- ✅ **Cost structure diagnostics** — flags products with cost ratios or margins breaching risk thresholds
- ✅ **Margin volatility tracking** — monthly standard deviation trend to assess portfolio stability over time
- ✅ **Fully reproducible** — clean, modular, well-commented code with a public dataset

---

## 📂 Repository Structure

```
nassau-candy-profitability/
│
├── assets/                                      # Project logo assets
│   └── logo.png                                 # Company logo for dashboard header
│
├── 20260125_Nassau_Candy_Distributor.ipynb      # Full EDA & analysis notebook
│                                                # Covers data cleaning, KPI calculations,
│                                                # product/division analysis, Pareto, and
│                                                # cost diagnostics with written commentary
│
├── Nassau_Candy_Distributor.csv                 # Source transactional dataset
│                                                # ~10,000+ order-level records across
│                                                # products, divisions, customers, and regions
│
├── streamlit_app.py                             # Interactive Streamlit dashboard
│                                                # Five analytical modules with sidebar
│                                                # filters and Plotly visualizations
│
├── requirements.txt                             # Python package dependencies
│
└── README.md                                    # Project documentation (this file)
```

---

## 🗂️ Dataset Description

| Field | Description |
|---|---|
| `Row ID` | Unique row identifier |
| `Order ID` | Unique order identifier |
| `Order Date` | Date the order was placed |
| `Ship Date` | Date the order was shipped |
| `Ship Mode` | Shipping method used |
| `Customer ID` | Unique customer identifier |
| `Country / Region` | Customer country or region |
| `City` | Customer city |
| `State / Province` | Customer state or province |
| `Postal Code` | Customer postal code |
| `Division` | Product division (Chocolate, Sugar, Other) |
| `Region` | Customer geographic region |
| `Product ID` | Unique product identifier |
| `Product Name` | Full product name |
| `Sales` | Total sales value of the order ($) |
| `Units` | Total units sold in the order |
| `Gross Profit` | Gross profit = Sales − Cost ($) |
| `Cost` | Manufacturing cost of the order ($) |

> **Source:** Dataset provided by [Unified Mentor](https://www.unifiedmentor.com/) as part of the Data Analyst Internship programme. Also accessible via the project's public [Google Drive link](https://drive.google.com/file/d/1c4VDb0Pf7RCgps4aLMiSuLtdaUpU_X49/view).

---

## 🛠️ Tools & Technologies

| Category | Tools |
|---|---|
| **Language** | Python 3.10+ |
| **Data Manipulation** | Pandas, NumPy |
| **Visualisation** | Plotly (Graph Objects & Express) |
| **Dashboard** | Streamlit |
| **Notebook Environment** | Jupyter Notebook |
| **Version Control** | Git & GitHub |

---

## 📊 Key Insights

> *Insights are directional and based on observed patterns in the data. Exact figures are available in the notebook and dashboard.*

- **Profit is highly concentrated** — a small subset of products drives the overwhelming majority of total gross profit, consistent with the Pareto principle. This concentration, while reflecting portfolio strength, introduces significant over-dependency risk.

- **The Chocolate division dominates financially** — it accounts for the vast majority of both total revenue and gross profit, and its profit contribution consistently exceeds its revenue share, signalling strong structural margin efficiency.

- **Revenue and profit contributions are misaligned in the Other division** — despite generating meaningful revenue, the Other division contributes a disproportionately smaller share of gross profit, indicating structural margin constraints likely tied to product mix or cost structure.

- **High-sales products are not always high-margin products** — several products appear commercially significant by volume but operate with thin margins that limit their actual profit contribution.

- **A clear cost-risk product was identified** — at least one product in the portfolio exhibits a cost ratio exceeding 0.90, meaning costs consume more than 90% of its sales revenue, flagging it for immediate cost renegotiation or portfolio review.

- **The Sugar division maintains healthy margins at small scale** — despite contributing minimally to total revenue, its gross margin is comparable to the Chocolate division, suggesting pricing efficiency in a niche segment.

---

## 🖥️ Dashboard Features

The Streamlit dashboard is structured into **five analytical modules**, accessible via the sidebar:

### 🔍 Sidebar Filters (Global)
- **Date Range Selector** — filter all modules by order date window
- **Division Filter** — isolate analysis to a specific product division
- **Product Search** — focus analysis on a single product
- **Margin Risk Threshold Slider** — surface only products meeting a minimum margin threshold

---

### 📋 Module 1 — Overview
- **Gross Margin (%) KPI** with donut visualisation and total sales / profit summary
- **Profit per Unit KPI** with gauge chart and units sold
- **Margin Volatility KPI** with rolling 3-month standard deviation sparkline
- **Revenue vs Profit Trend** — monthly line chart showing divergence between sales and profit over time

### 📦 Module 2 — Product Profitability Overview
- **Product Leaderboard** — ranked horizontal bar chart; switchable by Gross Profit, Gross Margin %, Sales, Units, or Profit per Unit
- **Revenue vs Profit Contribution Chart** — grouped bar comparing each product's revenue share and profit share side-by-side
- **Portfolio Positioning Scatter** — four-quadrant map (Stars / Volume Traps / Niche Opportunities / Exit Candidates) plotting every product by revenue and margin

### 🏢 Module 3 — Division Performance
- **Revenue vs Profit Contribution by Division** — grouped bar chart with percentage labels
- **Profit-Revenue Gap Chart** — diverging bar showing which divisions over- or under-perform relative to their revenue share
- **Gross Margin Quality Bar** — division-level margin comparison
- **Division Financial Diagnostics Table** — full summary with margin, contribution, gap, and performance flag

### 💰 Module 4 — Cost Diagnostics
- **Cost vs Sales Scatter** — product-level scatter with gross margin as colour encoding and a 45° break-even reference line; optional product labels
- **KPI Summary Row** — average margin %, high-risk product count, low-margin product count, average cost ratio
- **Risk-Flagged Products Table** — products with margin < 5% or cost ratio > 0.90, with strategic classification and recommended action
- **Division Risk Overview** — side-by-side table and bar chart of average margin and cost ratio by division

### 📈 Module 5 — Pareto Analysis
- **Pareto Combo Charts** (one each for Gross Profit and Sales) — bar chart of individual contribution overlaid with a cumulative percentage line and an 80% threshold reference line
- **Concentration Summary** — auto-generated statement identifying how many products reach the 80% contribution threshold

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/nassau-candy-profitability.git
cd nassau-candy-profitability
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv

# Activate — macOS / Linux
source venv/bin/activate

# Activate — Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Core dependencies include:**

```
streamlit
pandas
numpy
plotly
```

---

## ▶️ How to Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

The dashboard will open automatically in your default browser at:

```
http://localhost:8501
```

> **Note:** The dataset is loaded directly from the public GitHub repository URL within `streamlit_app.py`, so no manual data setup is required. Ensure you have an active internet connection on first load, or replace the URL with the local path to `Nassau_Candy_Distributor.csv`.

---

## 📸 Screenshots

> *Screenshot placeholders — replace with actual dashboard captures before publishing.*

| Module | Preview |
|---|---|
| Overview — KPI Panel | `[ screenshot: overview_kpis.png ]` |
| Product Leaderboard | `[ screenshot: product_leaderboard.png ]` |
| Portfolio Positioning Scatter | `[ screenshot: portfolio_scatter.png ]` |
| Division Performance | `[ screenshot: division_performance.png ]` |
| Cost Diagnostics Scatter | `[ screenshot: cost_diagnostics.png ]` |
| Pareto Analysis | `[ screenshot: pareto_analysis.png ]` |

---

## 🔮 Future Improvements

- [ ] **Time-series drill-down** — month-over-month and year-over-year margin trend comparison per product
- [ ] **Customer-segment profitability layer** — extend analysis to identify most and least profitable customer segments by region
- [ ] **Automated PDF report export** — generate a one-click summary report from the dashboard for stakeholder distribution
- [ ] **Pricing sensitivity module** — model the margin impact of hypothetical price changes on flagged products
- [ ] **Factory-level cost analysis** — leverage the factory-product correlation table from the PRD to attribute costs to sourcing locations
- [ ] **Deployment** — host the dashboard on Streamlit Community Cloud for public access

---

## 🙏 Acknowledgements

This project was completed as part of the **Data Analyst Internship Programme** at [**Unified Mentor**](https://www.unifiedmentor.com/).

- **Dataset:** Provided by Unified Mentor for the Nassau Candy Distributor analysis project
- **Project Scope & PRD:** Defined by Unified Mentor as part of the structured internship curriculum
- **Nassau Candy Distributor:** [nassaucandy.com](https://www.nassaucandy.com/) — the real-world business context for this analysis

---

## 👤 Author

**[Your Name]**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/your-profile)
[![GitHub](https://img.shields.io/badge/GitHub-Portfolio-181717?style=flat&logo=github&logoColor=white)](https://github.com/your-username)
[![Email](https://img.shields.io/badge/Email-Contact-D14836?style=flat&logo=gmail&logoColor=white)](mailto:your.email@example.com)

> *Replace the placeholders above with your actual name, LinkedIn profile, GitHub username, and email address before publishing.*

---

<div align="center">

*Built with 🍬 and Python — as part of a data analytics portfolio.*

</div>
