# 🍬 Product Line Profitability & Margin Performance Analysis
### Nassau Candy Distributor

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=flat-square&logo=streamlit)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?style=flat-square&logo=plotly)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=flat-square&logo=pandas)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 Project Overview

In high-volume distribution businesses like Nassau Candy, **sales volume alone is a misleading indicator of financial health**. A product line may generate impressive revenue while quietly eroding overall profitability through high input costs, thin margins, or structural pricing inefficiencies.

This project delivers a comprehensive **product-line profitability and margin performance analysis** using transactional sales data from Nassau Candy Distributor — one of the largest wholesale manufacturers and distributors of specialty confections in the United States. The analysis goes beyond revenue figures to uncover which products and divisions *truly* drive profit, where margin risk is concentrated, and what strategic actions can improve portfolio efficiency.

The findings are surfaced through an interactive **Streamlit dashboard** that enables business stakeholders to explore KPIs, drill down by product or division, and identify cost and margin risk flags — all without writing a single line of code.

---

## 🎯 Key Objectives

- 📊 Identify product lines with the **highest gross margin and profit contribution**
- 🔍 Detect **high-revenue but low-margin** products that create volume traps
- 🏭 Evaluate **division-level financial performance** and revenue-to-profit imbalances
- 📉 Perform **Pareto (80/20) analysis** to quantify profit concentration risk
- 💸 Diagnose **cost structure inefficiencies** and flag products requiring repricing or cost renegotiation
- 📈 Track **margin volatility** over time to identify stability risks
- 🗂️ Build an **interactive analytics dashboard** for business stakeholders

---

## 🗃️ Dataset Description

**File:** `Nassau_Candy_Distributor.csv`  
**Records:** ~10,194 transactions | **Fields:** 18

| Field | Description |
|---|---|
| `Order ID` | Unique order identifier |
| `Order Date` / `Ship Date` | Transaction and shipment timestamps |
| `Ship Mode` | Shipping method used |
| `Customer ID` | Unique customer identifier |
| `Country/Region`, `City`, `State/Province` | Customer location fields |
| `Division` | Product division (Chocolate, Sugar, Other) |
| `Region` | Customer geographic region |
| `Product ID` / `Product Name` | Product identifiers |
| `Sales` | Total revenue value of the order ($) |
| `Units` | Total units sold |
| `Cost` | Manufacturing cost of the order ($) |
| `Gross Profit` | Derived profit = Sales − Cost ($) |

> 📎 Dataset provided by [Unified Mentor](https://www.unifiedmentor.com/) as part of the Data Analyst Internship program.

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| **Data Processing** | Python 3.10+, Pandas, NumPy |
| **EDA & Static Visualization** | Matplotlib, Seaborn |
| **Interactive Visualization** | Plotly (Graph Objects + Express) |
| **Dashboard Framework** | Streamlit |
| **Notebook Environment** | Jupyter Notebook |
| **Version Control** | Git, GitHub |

---

## 🔄 Project Workflow

```
Raw CSV Data
     │
     ▼
┌─────────────────────┐
│   1. Data Cleaning  │  → Remove duplicates, fix dates, validate numerics,
│   & Validation      │    cross-check Gross Profit, standardize labels
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  2. Feature         │  → Gross Margin (%), Profit per Unit,
│     Engineering     │    Revenue & Profit Contribution (%),
│                     │    Profit-Revenue Gap, Margin Volatility
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  3. Exploratory     │  → Distribution analysis, outlier detection,
│     Data Analysis   │    trend identification (EDA Notebook)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  4. Profitability   │  → Product rankings, division diagnostics,
│     & Margin        │    Pareto concentration, cost structure
│     Analysis        │    analysis, strategic quadrant mapping
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  5. Dashboard       │  → Interactive Streamlit app with modular
│     Development     │    filters, KPI cards, and Plotly charts
└─────────────────────┘
```

---

## 💡 Key Insights

> *These insights reflect patterns commonly observed in this analysis. Exact values depend on the filtered dataset view.*

- **📦 High Revenue ≠ High Profit** — Several top-selling products operate on thin margins, contributing significantly to revenue while delivering disproportionately little profit. Volume alone is not a reliable profitability signal.

- **📐 Pareto Principle Holds** — A small subset of products (~20%) is responsible for the majority of both total revenue and gross profit. This concentration creates strategic opportunity but also over-dependency risk.

- **🍫 Chocolate Division Dominates** — The Chocolate division generates the overwhelming majority of both revenue and profit, with gross margins exceeding 67%. Its profit contribution outpaces its revenue share — a strong efficiency signal.

- **⚠️ The "Other" Division Underperforms** — Despite contributing ~6.8% of revenue, the Other division yields a disproportionately lower profit share (~4.6%), reflecting a negative profit-revenue gap and structural margin constraints.

- **💰 Cost-Heavy Products Suppress Margins** — Certain products exhibit cost ratios above 0.90, meaning over 90 cents of every dollar in revenue is consumed by manufacturing cost. These products require urgent repricing or cost renegotiation.

- **🏷️ Low-Volume, Low-Margin Products Carry Tail Risk** — A cluster of products with minimal revenue and sub-5% margins contribute little value to the portfolio and represent rationalization candidates.

---

## 📊 Streamlit Dashboard Features

The dashboard is organized into **5 interactive modules**, accessible via the sidebar:

### 🏠 Overview
- Global KPI cards: **Gross Margin %**, **Profit per Unit**, **Margin Volatility**
- Donut chart for margin visualization
- Gauge chart for profit-per-unit benchmark
- Rolling 3-month volatility sparkline
- Monthly **Revenue vs. Gross Profit trend** line chart

### 📦 Product Profitability Overview
- **Product leaderboard** — ranked by Gross Margin %, Gross Profit, Sales, Units, or Profit per Unit (user-selectable)
- **Revenue vs. Profit Contribution** grouped bar chart
- **Portfolio Positioning Scatter** — maps every product across Revenue vs. Margin with quadrant classification (Stars, Volume Traps, Niche Opportunities, Exit Candidates)

### 🏭 Division Performance
- Revenue vs. Profit Contribution comparison by division
- **Profit-Revenue Gap** diverging bar chart (green = overperforming, red = underperforming)
- Gross Margin % ranking by division
- Executive diagnostic summary table

### 🔬 Cost Diagnostics
- **Cost vs. Sales scatter** with Gross Margin % color encoding and 45-degree break-even reference line
- KPI row: Avg Margin %, High Risk Products, Low Margin Products, Avg Cost Ratio
- **Strategic position classification**: Star / Volume Trap / Niche Opportunity / Exit Candidate
- **Risk flag table** with actionable recommendations (Repricing Review / Cost Renegotiation / Immediate Review)
- Division-level cost risk summary chart

### 📉 Pareto Analysis
- Dual-axis **Pareto combo chart** (bar + cumulative % line) for both Gross Profit and Sales
- Automatic identification of the product count that generates 80% of each metric
- Portfolio concentration percentage callout

### 🎛️ Interactive Filters (Sidebar)
- 📅 Date range selector
- 🏭 Division filter
- 🔎 Product search
- 📐 Margin risk threshold slider

---

## 🖼️ Dashboard Screenshots

> *Screenshots will be added after deployment.*

| Module | Preview |
|---|---|
| Overview — KPIs & Trend | `[Screenshot Placeholder]` |
| Product Profitability Leaderboard | `[Screenshot Placeholder]` |
| Portfolio Positioning Scatter | `[Screenshot Placeholder]` |
| Division Performance | `[Screenshot Placeholder]` |
| Cost Diagnostics | `[Screenshot Placeholder]` |
| Pareto Analysis | `[Screenshot Placeholder]` |

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Product-Line-Profitability-Margin-Performance-Analysis-for-Nassau-Candy-Distributor.git
cd Product-Line-Profitability-Margin-Performance-Analysis-for-Nassau-Candy-Distributor
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
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

### 5. Run the EDA Notebook (Optional)

```bash
jupyter notebook 20260125_Nassau_Candy_Distributor.ipynb
```

---

## 📦 Requirements

Key dependencies from `requirements.txt`:

```
streamlit
pandas
numpy
plotly
matplotlib
seaborn
jupyter
```

> Install all at once with `pip install -r requirements.txt`.

---

## 📁 Repository Structure

```
📦 Product-Line-Profitability-Margin-Performance-Analysis-for-Nassau-Candy-Distributor
 ┣ 📂 assets/
 ┃ ┗ 🖼️ logo.png                               ← Dashboard header logo
 ┣ 📓 20260125_Nassau_Candy_Distributor.ipynb   ← EDA Notebook
 ┣ 📊 Nassau_Candy_Distributor.csv              ← Raw dataset
 ┣ 🐍 streamlit_app.py                          ← Streamlit dashboard
 ┣ 📄 requirements.txt                          ← Python dependencies
 ┗ 📄 README.md                                 ← Project documentation
```

---

## 🔮 Future Improvements

- 🤖 **ML Integration** — Implement margin prediction models (e.g., regression, gradient boosting) to forecast product-level profitability under different pricing or cost scenarios
- 📈 **Demand Forecasting** — Apply time-series models (ARIMA, Prophet) to project revenue and profit trends by product or division
- 🚚 **Supply Chain Optimization** — Incorporate shipping route and lead time data to correlate logistics costs with margin performance
- 🗺️ **Geographic Profitability Mapping** — Visualize profit and margin performance by state/region using choropleth maps
- 🔔 **Automated Alerts** — Add margin threshold alerting for real-time monitoring of at-risk products
- ☁️ **Cloud Deployment** — Deploy dashboard to Streamlit Community Cloud or AWS for stakeholder access without local setup

---

## 🙏 Acknowledgements

- **[Unified Mentor](https://www.unifiedmentor.com/)** — For providing the project scope, dataset, and internship framework
- **[Nassau Candy Distributor](https://www.nassaucandy.com/)** — Dataset originates from their transactional sales records
- **Streamlit & Plotly teams** — For the open-source tools that power the interactive dashboard

---

## 👤 Author

**[Your Name]**  
Data Analyst Intern — Unified Mentor  
📧 [your.email@example.com]  
🔗 [LinkedIn Profile](https://linkedin.com/in/your-profile)  
🐙 [GitHub Profile](https://github.com/your-username)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).  
Feel free to use, adapt, and build on this work with attribution.

---

<div align="center">
  <sub>Built with ❤️ using Python, Plotly, and Streamlit</sub>
</div>
