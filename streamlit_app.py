import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------


st.set_page_config(
    page_title="Nassau Candy Executive Analytics",
    layout="wide"
)
st.markdown(
    "<h1 style='text-align: center;'>Product Line Profitability & Margin Performance Analysis</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h4 style='text-align: center; color: gray;'>Nassau Candy Distributor</h4>",
    unsafe_allow_html=True
)

st.markdown("---")

plt.style.use("dark_background")
sns.set_theme(style="dark")

def style_chart(fig, ax):
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    return fig

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/amrit1426/Unified_mentor_project/refs/heads/main/Nassau_Candy_Distributor.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")

    # Match EDA logic exactly
    df["Gross Margin"] = df["Gross Profit"] / df["Sales"]
    df["Profit per Unit"] = df["Gross Profit"] / df["Units"]

    return df

df = load_data()

# ===================================================
# GLOBAL KPI SUMMARY
# ===================================================

total_sales = df["Sales"].sum()
total_cost = df["Cost"].sum()
total_profit = df["Gross Profit"].sum()
gross_margin = (total_profit / total_sales) * 100

# st.markdown("## Summary")

k1, k2, k3, k4 = st.columns(4)

k1.metric("Total Sales", f"${total_sales:,.0f}")
k2.metric("Total Gross Profit", f"${total_profit:,.0f}")
k3.metric("Gross Margin %", f"{gross_margin:.2f}%")
k4.metric("Total Cost", f"${total_cost:,.0f}")

st.divider()



# ---------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------
st.sidebar.title("Filters")

division_filter = st.sidebar.multiselect(
    "Division",
    options=df["Division"].unique(),
    default=df["Division"].unique()
)

margin_threshold = st.sidebar.slider(
    "Margin Risk Threshold (%)",
    0, 100, 20
)
min_date = df["Order Date"].min()
max_date = df["Order Date"].max()

date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# ---------------------------------------------------
# SAFE FILTERING (Fixes unintended filtering issue)
# ---------------------------------------------------
filtered_df = df.copy()

# Apply date filter only if user changed range
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])

    if (start_date != min_date) or (end_date != max_date):
        filtered_df = filtered_df[
            (filtered_df["Order Date"].notna()) &
            (filtered_df["Order Date"] >= start_date) &
            (filtered_df["Order Date"] <= end_date)
        ]

# Apply division filter only if subset selected
if len(division_filter) < len(df["Division"].unique()):
    filtered_df = filtered_df[
        filtered_df["Division"].isin(division_filter)
    ]

# ---------------------------------------------------
# TABS
# ---------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Product Profitability",
    "Division Performance",
    "Cost Diagnostics",
    "Pareto Analysis"
])

# ===================================================
# TAB 1 – PRODUCT PROFITABILITY
# ===================================================
with tab1:

    st.header("Product Profitability Overview")

    # Match EDA aggregation exactly
    product_summary = (
        filtered_df
        .groupby("Product Name")
        .agg({
            "Sales": "sum",
            "Gross Profit": "sum",
            "Units": "sum",
            "Gross Margin": "mean"
        })
        .reset_index()
    )

    # -------- Leaderboard Controls --------
    col1, col2 = st.columns(2)
    with col1:
        metric_choice = st.selectbox(
        "Leaderboard Ranking Metric",
        ["Gross Profit", "Gross Margin", "Sales", "Units"]
    )

    with col2:
        leaderboard_size = st.slider(
            "Number of Products",
            5, 40, 15
        )

    leaderboard = (
        product_summary
        .sort_values(metric_choice, ascending=False)
        .head(leaderboard_size)
    )

    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(
        data=leaderboard,
        y="Product Name",
        x=metric_choice,
        ax=ax
    )
    ax.set_title(f"Top Products by {metric_choice}")
    ax.set_xlabel(metric_choice)
    ax.set_ylabel("")
    fig = style_chart(fig, ax)
    st.pyplot(fig)

    # -------- Profit Contribution --------
    st.subheader("Profit Contribution (%)")

    # Aggregate & sort
    contribution = (
        product_summary
        .sort_values("Gross Profit", ascending=False)
        .head(leaderboard_size)
        .copy()
    )

    # Calculate percentage contribution
    total_profit_products = product_summary["Gross Profit"].sum()
    contribution["Contribution %"] = (
        contribution["Gross Profit"] / total_profit_products
    )

    # Wrap long product names
    def wrap_labels(label, width=25):
        import textwrap
        return "\n".join(textwrap.wrap(label, width))

    contribution["Wrapped Name"] = contribution["Product Name"].apply(wrap_labels)

    # Plot
    fig, ax = plt.subplots(figsize=(11,7))

    bars = ax.barh(
        contribution["Wrapped Name"],
        contribution["Contribution %"]
    )
    ax.invert_yaxis()
    # ax.set_title("Product Profit Contribution")
    ax.set_xlabel("Contribution % of Total Profit")
    ax.set_ylabel("")

    # 🔹 Reduce y-axis tick font size
    ax.tick_params(axis='y', labelsize=8)

    # 🔹 Annotate with bar color for visibility
    for bar in bars:
        width = bar.get_width()
        bar_color = bar.get_facecolor()

        ax.text(
            width + 0.005,
            bar.get_y() + bar.get_height()/2,
            f"{width:.2%}",
            va="center",
            fontsize=9,
            color="white"
        )

    ax.set_xlim(0, contribution["Contribution %"].max() * 1.15)

    fig = style_chart(fig, ax)
    st.pyplot(fig)

# ===================================================
# TAB 2 – DIVISION PERFORMANCE (EXECUTIVE VERSION)
# ===================================================
with tab2:

    st.header("Division Performance Dashboard")

    # --------------------------------------------------
    # Aggregate Division-Level Metrics
    # --------------------------------------------------
    division_metrics = (
        filtered_df
        .groupby('Division')
        .agg({
            'Sales': 'sum',
            'Gross Profit': 'sum',
            'Cost': 'sum'
        })
        .reset_index()
    )

    # Core KPIs
    division_metrics['Gross Margin'] = (
        division_metrics['Gross Profit'] /
        division_metrics['Sales']
    )

    total_revenue = division_metrics['Sales'].sum()
    total_profit = division_metrics['Gross Profit'].sum()

    division_metrics['Revenue_Contribution_%'] = (
        division_metrics['Sales'] / total_revenue * 100
    )

    division_metrics['Profit_Contribution_%'] = (
        division_metrics['Gross Profit'] / total_profit * 100
    )

    division_metrics['Profit_Revenue_Gap'] = (
        division_metrics['Profit_Contribution_%'] -
        division_metrics['Revenue_Contribution_%']
    )

    # Performance flag
    division_metrics['Performance Flag'] = np.where(
        division_metrics['Profit_Revenue_Gap'] > 0,
        "Overperforming",
        "Underperforming"
    )

    # Consistent sorting by Revenue
    division_metrics = division_metrics.sort_values(
        'Sales', ascending=False
    ).reset_index(drop=True)

    divisions = division_metrics['Division']
    x = np.arange(len(divisions))

    # ===================================================
    # 1️⃣ Revenue vs Profit (Absolute Comparison)
    # ===================================================
    st.subheader("Revenue vs Profit Comparision by Division")

    fig1, ax1 = plt.subplots(figsize=(10,5))

    bar_width = 0.35

    bars1 = ax1.bar(
        x,
        division_metrics['Sales'],
        width=bar_width,
        label='Revenue ($)',
        color="#5364B1"
    )

    bars2 = ax1.bar(
        x + bar_width,
        division_metrics['Gross Profit'],
        width=bar_width,
        label='Gross Profit ($)',
        color="#3A9E85"
    )

    ax1.set_xticks(x + bar_width / 2)
    ax1.set_xticklabels(divisions, fontweight='bold')
    ax1.set_ylabel("Amount ($)")
    # ax1.set_title("Revenue vs Gross Profit by Division", fontweight='bold')
    ax1.legend()

    fig1 = style_chart(fig1, ax1)
    st.pyplot(fig1)

    # ===================================================
    # 2️⃣ Gross Margin by Division
    # ===================================================
    st.subheader("Gross Margin Distribution by Division")
    # Sort divisions by Gross Margin (High → Low)
    sorted_margin = division_metrics.sort_values(
        'Gross Margin', ascending=False
    )

    fig2, ax2 = plt.subplots(figsize=(9,5))

    bars = ax2.bar(
        sorted_margin['Division'],
        sorted_margin['Gross Margin'],
        color="#4C72B0"
    )

    ax2.set_ylabel("Gross Margin (%)")
    # ax2.set_title("Division-Level Margin Performance", fontweight='bold')

    # Annotate margin %
    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.005,
            f"{height:.2%}",
            ha='center',
            fontsize=10,
            color="white",
            fontweight="bold"
        )

    fig2 = style_chart(fig2, ax2)
    st.pyplot(fig2)



    # ===================================================
    # 3️⃣ Revenue vs Profit Contribution (%)
    # ===================================================
    st.subheader("Revenue vs Profit Contribution (%)")

    fig3, ax3 = plt.subplots(figsize=(10,5))

    bars1 = ax3.bar(
        x,
        division_metrics['Revenue_Contribution_%'],
        width=bar_width,
        label='Revenue Contribution (%)',
        color="#5364B1"
    )

    bars2 = ax3.bar(
        x + bar_width,
        division_metrics['Profit_Contribution_%'],
        width=bar_width,
        label='Profit Contribution (%)',
        color="#3A9E85"
    )
    # Annotate Revenue bars
    for bar in bars1:
        height = bar.get_height()
        bar_color = bar.get_facecolor()

        ax3.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.5,
            f"{height:.1f}%",
            ha='center',
            fontsize=9,
            color="white"
        )

    # Annotate Profit bars
    for bar in bars2:
        height = bar.get_height()
        bar_color = bar.get_facecolor()

        ax3.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.5,
            f"{height:.1f}%",
            ha='center',
            fontsize=9,
            color="white"
        )
    ax3.set_xticks(x + bar_width / 2)
    ax3.set_xticklabels(divisions, fontweight='bold')
    ax3.set_ylabel("Contribution (%)")
    # ax3.set_title("Revenue vs Profit Share by Division", fontweight='bold')
    ax3.legend()

    fig3 = style_chart(fig3, ax3)
    st.pyplot(fig3)

    # ===================================================
    # 4️⃣ Profit-Revenue Gap (Efficiency Indicator)
    # ===================================================
    st.subheader("Profit-Revenue Gap (Efficiency Indicator)")

    fig4, ax4 = plt.subplots(figsize=(9,4))

    colors = [
        "#3A9E85" if gap > 0 else "#D9534F"
        for gap in division_metrics['Profit_Revenue_Gap']
    ]

    bars = ax4.bar(
        divisions,
        division_metrics['Profit_Revenue_Gap'],
        color=colors
    )

    ax4.axhline(0, linewidth=1)
    ax4.set_ylabel("Gap (%)")
    # ax4.set_title("Profit Contribution Minus Revenue Contribution", fontweight='bold')

    for bar in bars:
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width()/2,
            height + (0.3 if height > 0 else -0.8),
            f"{height:.2f}%",
            ha='center',
            fontsize=9,
            color="white"
        )

    fig4 = style_chart(fig4, ax4)
    st.pyplot(fig4)

    # ===================================================
    # 5️⃣ Executive Diagnostic Table
    # ===================================================
    st.subheader("Division Financial Diagnostics")

    st.dataframe(
        division_metrics[
            [
                "Division",
                "Gross Margin",
                "Revenue_Contribution_%",
                "Profit_Contribution_%",
                "Profit_Revenue_Gap",
                "Performance Flag"
            ]
        ].style.format({
            "Gross Margin": "{:.2%}",
            "Revenue_Contribution_%": "{:.2f}%",
            "Profit_Contribution_%": "{:.2f}%",
            "Profit_Revenue_Gap": "{:.2f}%"
        })
    )


# ===================================================
# TAB 3 – COST DIAGNOSTICS
# ===================================================
with tab3:

    st.header("Cost Structure Diagnostics")
    st.markdown(
        "<p style='color:white;'>"
        "Evaluate product-level cost efficiency, margin risk, and strategic positioning."
        "</p>",
        unsafe_allow_html=True
    )

    st.divider()

    # ---------------------------------------------------
    # 1️⃣ FILTER
    # ---------------------------------------------------
    colf1, colf2 = st.columns([1,1])

    with colf1:
        selected_division = st.selectbox(
            "Select Division",
            ["All"] + list(filtered_df["Division"].unique())
        )

    with colf2:
        show_labels = st.checkbox("Show Product Labels", value=True)

    if selected_division != "All":
        df_diag = filtered_df[filtered_df["Division"] == selected_division]
    else:
        df_diag = filtered_df.copy()

    # ---------------------------------------------------
    # 2️⃣ PRODUCT AGGREGATION
    # ---------------------------------------------------
    product_cost = (
        df_diag
        .groupby(["Division", "Product Name"])
        .agg({
            "Sales": "sum",
            "Cost": "sum",
            "Gross Margin": "mean"
        })
        .reset_index()
    )

    product_cost["Cost Ratio"] = product_cost["Cost"] / product_cost["Sales"]
    product_cost["Margin %"] = product_cost["Gross Margin"] * 100

    # ---------------------------------------------------
    # 3️⃣ SCATTER PLOT (Cost vs Sales)
    # ---------------------------------------------------
    st.subheader("Cost vs Sales by Product")

    fig, ax = plt.subplots(figsize=(10,6))

    scatter = ax.scatter(
        product_cost["Cost"],
        product_cost["Sales"],
        c=product_cost["Margin %"],
        cmap="coolwarm",
        alpha=0.85
    )

    # 45-degree reference line
    max_val = max(product_cost["Cost"].max(), product_cost["Sales"].max())
    ax.plot([0, max_val], [0, max_val], linestyle="--")

    # Optional labels
    if show_labels:
        for i, row in product_cost.iterrows():
            ax.text(
                row["Cost"],
                row["Sales"],
                row["Product Name"],
                fontsize=8,
                color="white"
            )

    ax.set_xlabel("Cost", color="white")
    ax.set_ylabel("Sales", color="white")
    ax.set_title("Cost vs Sales (Color = Margin %)", color="white")
    ax.tick_params(colors='white')

    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    cbar = fig.colorbar(scatter)
    cbar.ax.tick_params(colors='white')

    st.pyplot(fig)

    # ---------------------------------------------------
    # 4️⃣ KPI SUMMARY ROW
    # ---------------------------------------------------
    st.divider()

    k1, k2, k3, k4 = st.columns(4)

    high_risk_count = (product_cost["Margin %"] < 0).sum()
    low_margin_count = ((product_cost["Margin %"] >= 0) & (product_cost["Margin %"] < 5)).sum()

    k1.metric("Avg Margin %", f"{product_cost['Margin %'].mean():.2f}%")
    k2.metric("High Risk Products", high_risk_count)
    k3.metric("Low Margin Products", low_margin_count)
    k4.metric("Avg Cost Ratio", f"{product_cost['Cost Ratio'].mean():.2f}")

    # ---------------------------------------------------
    # 5️⃣ QUADRANT CLASSIFICATION
    # ---------------------------------------------------
    sales_median = product_cost["Sales"].median()
    margin_median = product_cost["Margin %"].median()

    def classify(row):
        if row["Sales"] >= sales_median and row["Margin %"] >= margin_median:
            return "Star"
        elif row["Sales"] >= sales_median and row["Margin %"] < margin_median:
            return "Volume Trap"
        elif row["Sales"] < sales_median and row["Margin %"] >= margin_median:
            return "Niche Opportunity"
        else:
            return "Exit Candidate"

    product_cost["Strategic Position"] = product_cost.apply(classify, axis=1)

    # ---------------------------------------------------
    # 6️⃣ RISK FLAGS
    # ---------------------------------------------------
    risk = product_cost[
        (product_cost["Margin %"] < 5) |
        (product_cost["Cost Ratio"] > 0.9)
    ].copy()

    risk["Recommendation"] = np.where(
        risk["Margin %"] < 0,
        "Immediate Review",
        np.where(
            risk["Cost Ratio"] > 0.9,
            "Cost Renegotiation",
            "Repricing Review"
        )
    )

    risk = risk.sort_values("Margin %")

    st.divider()
    st.subheader("Products Requiring Attention")
    st.dataframe(
        risk[[
            "Division",
            "Product Name",
            "Sales",
            "Cost",
            "Margin %",
            "Cost Ratio",
            "Strategic Position",
            "Recommendation"
        ]]
    )

    # ---------------------------------------------------
    # 7️⃣ DIVISION SUMMARY
    # ---------------------------------------------------
    st.divider()
    st.subheader("Division Risk Overview")

    colA, colB = st.columns(2)

    with colA:
        division_summary = (
            product_cost
            .groupby("Division")
            .agg({
                "Margin %": "mean",
                "Cost Ratio": "mean"
            })
            .sort_values("Margin %")
        )
        st.dataframe(division_summary)

    with colB:
        fig2, ax2 = plt.subplots(figsize=(6,4))
        division_summary["Margin %"].plot(kind="barh", ax=ax2)

        ax2.set_xlabel("Average Margin %", color="white")
        ax2.tick_params(colors='white')

        fig2.patch.set_facecolor("#0E1117")
        ax2.set_facecolor("#0E1117")

        st.pyplot(fig2)


# ===================================================
# TAB 4 – PARETO (Line Only)
# ===================================================
with tab4:

    st.header("Profit & Revenue Concentration Analysis")

    for metric in ["Gross Profit", "Sales"]:

        st.subheader(f"80% Contribution – {metric}")

        pareto = (
            filtered_df
            .groupby("Product Name")[metric]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )

        pareto["Cumulative %"] = (
            pareto[metric].cumsum() /
            pareto[metric].sum()
        )

        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(
            np.arange(1, len(pareto)+1),
            pareto["Cumulative %"],
            linewidth=2
        )
        ax.axhline(0.8, linestyle="--")
        ax.set_title(f"Cumulative Curve – {metric}")
        ax.set_xlabel("Product Rank")
        ax.set_ylabel("Cumulative %")
        fig = style_chart(fig, ax)
        st.pyplot(fig)

        cutoff = np.argmax(pareto["Cumulative %"] >= 0.8) + 1
        st.success(f"{cutoff} products generate 80% of {metric}")

