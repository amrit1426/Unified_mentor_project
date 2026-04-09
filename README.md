```markdown
# 🍬 Product Line Profitability & Margin Performance Analysis
### Nassau Candy Distributor

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?logo=plotly&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview

Nassau Candy is one of the largest wholesale manufacturers and distributors of specialty and private-label confections in the United States, operating across thousands of SKUs with vertically integrated manufacturing and nationwide distribution.

**The Business Problem:**
In high-volume distribution environments, revenue figures alone are a misleading measure of business health. Products that sell in large volumes may carry thin margins, absorb disproportionate operational costs, or weaken overall portfolio profitability without being immediately visible in top-line metrics.

**Objective:**
This project performs a comprehensive, data-driven profitability and margin performance analysis on Nassau Candy's transactional sales data to:

- Identify which products and divisions truly drive gross profit
- Detect margin-risk items within the product portfolio
- Quantify profit concentration and over-dependency risk
- Diagnose cost structure inefficiencies at the product level
- Deliver actionable recommendations for pricing, sourcing, and portfolio decisions

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Exploratory Data Analysis** | Full EDA on transactional sales data using pandas, numpy, matplotlib, and seaborn |
| **Profitability Analysis** | Product and division-level ranking by gross profit, gross margin, and profit per unit |
| **Margin Analysis** | Margin volatility tracking, profit-revenue gap detection, and division efficiency scoring |
| **Pareto Analysis** | 80/20 concentration analysis for both revenue and profit to identify over-dependency risk |
| **Cost Diagnostics** | Cost-vs-sales scatter analysis, cost ratio flagging, and strategic quadrant classification |
| **Interactive Dashboard** | Multi-module Streamlit dashboard with real-time filters, Plotly charts, and KPI cards |

---

## 🛠️ Tech Stack

### EDA Notebook
| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, and aggregation |
| `numpy` | Numerical operations and metric calculations |
| `matplotlib` | Static visualizations and chart styling |
| `seaborn` | Statistical plots and distribution analysis |

### Streamlit Dashboard
| Library | Purpose |
|---|---|
| `streamlit` | Dashboard framework and UI layout |
| `plotly` | Interactive charts (bar, scatter, Pareto, gauge, donut) |
| `pandas` | Data filtering and KPI aggregation |
| `numpy` | Derived metric and flag calculations |

---

## 📁 Project Structure

```
nassau-candy-profitability/
│
├── assets/
│   └── logo.png                              # Company logo for dashboard header
│
├── 20260125_Nassau_Candy_Distributor.ipynb   # EDA notebook — full exploratory analysis
├── Nassau_Candy_Distributor.csv              # Raw transactional dataset
├── streamlit_app.py                          # Interactive Streamlit dashboard
├── requirements.txt                          # Python dependencies
└── README.md                                 # Project documentation
```

---

## 📊 Dataset Description

The dataset contains **10,194 transactional records** across **18 fields**, where each row represents a single product line item within a customer order.

| Field | Description |
|---|---|
| `Order ID` | Unique order identifier |
| `Order Date` / `Ship Date` | Transaction and shipment dates |
| `Ship Mode` | Shipping method used |
| `Customer ID` | Unique customer identifier |
| `City`, `State/Province`, `Region` | Customer geographic information |
| `Division` | Product division (Chocolate, Sugar, Other) |
| `Product ID` / `Product Name` | Product identifiers |
| `Sales` | Total sales value of the order ($) |
| `Units` | Total units sold |
| `Cost` | Manufacturing cost of the order ($) |
| `Gross Profit` | Sales − Cost ($) |

> 📥 **Dataset Source:** [Nassau Candy Distributor Dataset](https://drive.google.com/file/d/1c4VDb0Pf7RCgps4aLMiSuLtdaUpU_X49/view) — provided by [Unified Mentor](https://www.unifiedmentor.com/)

---

## 🔬 Methodology

### 1. Data Cleaning & Validation
- Removed exact duplicate records
- Parsed and validated `Order Date` and `Ship Date` fields; excluded unparseable rows
- Enforced numeric types on `Sales`, `Cost`, `Units`, and `Gross Profit`
- Removed records with zero or negative values in financial fields
- Cross-validated `Gross Profit` against `Sales − Cost`; excluded inconsistent rows
- Standardized `Product Name` and `Division` labels (whitespace stripped, title-cased)

### 2. Feature Engineering
- **Gross Margin (row-level):** `Gross Profit / Sales`
- **Profit per Unit (row-level):** `Gross Profit / Units`

### 3. KPI Calculations

| KPI | Formula |
|---|---|
| Gross Margin (%) | `(Gross Profit ÷ Sales) × 100` |
| Profit per Unit | `Gross Profit ÷ Units Sold` |
| Revenue Contribution (%) | `Product Sales ÷ Total Sales × 100` |
| Profit Contribution (%) | `Product Profit ÷ Total Profit × 100` |
| Profit-Revenue Gap (%) | `Profit Contribution − Revenue Contribution` |
| Margin Volatility | `σ(Monthly Gross Margin %)` |
| Cost Ratio | `Cost ÷ Sales` |

### 4. Analytical Approach
1. **Product-Level Analysis** — Ranked products by gross profit and gross margin; classified portfolio into strategic quadrants (Stars, Volume Traps, Niche Opportunities, Exit Candidates)
2. **Division-Level Analysis** — Aggregated metrics by division; evaluated revenue-vs-profit imbalance and flagged overperforming and underperforming divisions
3. **Pareto Analysis** — Determined the minimum number of products required to generate 80% of total revenue and total profit
4. **Cost Diagnostics** — Scatter-based cost-vs-sales analysis; flagged products with margin < 5% or cost ratio > 0.90 with actionable recommendations

---

## 📱 Dashboard Features

The Streamlit dashboard is organized into **five independent modules**, accessible from the sidebar.

### Overview
- Global KPI cards: **Gross Margin (%)**, **Profit per Unit**, **Margin Volatility**
- Gross margin donut chart and profit-per-unit gauge
- Rolling 3-month volatility sparkline
- Monthly **Revenue vs. Gross Profit** trend chart

### Product Profitability Overview
- Leaderboard: rank products by any KPI (Gross Margin %, Gross Profit, Sales, Units, Profit per Unit)
- Grouped bar chart: **Revenue vs. Profit Contribution (%)** per product
- Scatter chart: **Portfolio Positioning** — Revenue vs. Gross Margin with quadrant classification

### Division Performance
- Grouped bar: **Revenue vs. Profit Contribution (%)** by division
- Diverging bar: **Profit-Revenue Gap** with overperforming/underperforming flags
- Bar chart: **Gross Margin (%)** by division
- Executive diagnostic table with all division-level KPIs

### Cost Diagnostics
- Interactive **Cost vs. Sales scatter** (color-encoded by margin %, with break-even reference line)
- Summary KPI row: Avg Margin %, High Risk Products, Low Margin Products, Avg Cost Ratio
- **Products Requiring Attention** table with strategic position and recommendations
- Division risk overview table and bar chart

### Pareto Analysis
- Dual-axis Pareto chart (bar + cumulative line) for **Gross Profit** and **Sales**
- 80% threshold reference line with automatic cutoff annotation

### Sidebar Filters (apply across all modules)
- 📅 Date range selector
- 🏭 Division filter
- 🔍 Product search
- 📉 Margin risk threshold slider

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.9 or higher
- pip

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/nassau-candy-profitability.git
cd nassau-candy-profitability
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv

# On macOS/Linux
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Dashboard
```bash
streamlit run streamlit_app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

### 5. Run the EDA Notebook
```bash
jupyter notebook 20260125_Nassau_Candy_Distributor.ipynb
```

> **Note:** The dashboard loads data directly from the public GitHub-hosted CSV. No manual data setup is required.

---

## 💡 Results & Insights

- **Profit is highly concentrated:** A small subset of products drives the majority of both revenue and profit, consistent with the Pareto principle. This introduces over-dependency risk requiring active portfolio management.

- **Chocolate division dominates:** The Chocolate division accounts for approximately **92.9% of revenue** and **95.1% of gross profit**, with a gross margin of ~67.5% and a positive profit-revenue gap — indicating structural over-performance relative to its revenue share.

- **The Other division underperforms:** Despite contributing ~6.8% of revenue, the Other division delivers only ~4.6% of total profit with the lowest gross margin (~44.8%) and a negative profit-revenue gap, signaling structural margin constraints.

- **Kazookles flagged as cost risk:** With a cost ratio of 0.92 and a gross margin of only 7.69%, Kazookles (Other division) was identified as a Volume Trap requiring cost renegotiation or pricing review.

- **High-sales ≠ high-margin:** Several products generating significant revenue operate below the portfolio's average gross margin, representing pricing inefficiency or elevated cost structures that reduce their true financial contribution.

---

## 🔮 Future Improvements

- **Time-series forecasting** — Integrate ARIMA or Prophet models to project margin trends and flag early deterioration
- **Customer-level profitability** — Extend analysis to measure margin contribution per customer segment or region
- **Automated alerting** — Add threshold-based email or Slack alerts when product margins fall below defined risk levels
- **Scenario modeling** — Build an interactive what-if tool allowing pricing and cost simulations within the dashboard
- **Database integration** — Replace CSV ingestion with a live database or data warehouse connection for real-time analytics
- **Unit test coverage** — Add automated tests for KPI calculation functions to ensure data integrity across updates

---

## 🙏 Acknowledgements

- **[Unified Mentor](https://www.unifiedmentor.com/)** — Project scope, mentorship, and dataset provision
- **[Nassau Candy](https://www.nassaucandy.com/)** — Real-world business context

---

*This project was completed as part of the Unified Mentor Data Analyst Internship Program.*
```
