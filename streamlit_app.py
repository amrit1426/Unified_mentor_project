import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import base64
import os

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Nassau Candy Product Line Profitability & Margin Performance Analysis",
    page_icon="Active",
    layout="wide"
)


# ---------------------------------------------------
# HEADER LOGOS
# ---------------------------------------------------


def get_base64_image(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

um_base64 = get_base64_image("assets/um.png")
nas_base64 = get_base64_image("assets/nas.png")

st.markdown(
    f"""
    <div style="
        background-color: #ffffff;
        border-bottom: 1px solid #e5e7eb;
        padding: 25px 0;
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 8vw;
    ">
        <img src="data:image/png;base64,{um_base64}" width="220">
        <img src="data:image/png;base64,{nas_base64}" width="260">
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------
# MAIN TITLE
# ---------------------------------------------------

st.markdown("## Product Line Profitability & Margin Performance Analysis")
st.divider()
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
# LOAD & CLEAN DATA WITH LOGGING
# ---------------------------------------------------

@st.cache_data
def load_data():
    # Load CSV
    df = pd.read_csv(
        "https://raw.githubusercontent.com/amrit1426/Unified_mentor_project/refs/heads/main/Nassau_Candy_Distributor.csv"
    )
    original_rows = len(df)

    # Drop exact duplicate rows
    df = df.drop_duplicates()
    duplicates_dropped = original_rows - len(df)

    # Convert dates
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d-%m-%Y', errors='coerce')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d-%m-%Y', errors='coerce')
    df = df.dropna(subset=['Order Date', 'Ship Date'])
    dates_dropped = original_rows - duplicates_dropped - len(df)

    # Enforce numeric types
    numeric_cols = ['Sales', 'Cost', 'Units', 'Gross Profit']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Remove rows with zero or negative Sales, Cost, or Units
    pre_filter_rows = len(df)
    df = df[(df['Sales'] > 0) & (df['Cost'] > 0) & (df['Units'] > 0)]
    invalid_numeric_dropped = pre_filter_rows - len(df)

    # Validate Gross Profit (remove inconsistent rows)
    pre_profit_check = len(df)
    df = df[df['Gross Profit'].round(2) == (df['Sales'] - df['Cost']).round(2)]
    profit_mismatch_dropped = pre_profit_check - len(df)

    # Standardize text labels
    df['Division'] = df['Division'].str.strip().str.title()
    df['Product Name'] = df['Product Name'].str.strip().str.title()

    # Reset index
    df = df.reset_index(drop=True)

    # # Logging summary
    # st.markdown("### Data Cleaning Summary")
    # st.markdown(f"- Original rows: {original_rows}")
    # st.markdown(f"- Duplicate rows dropped: {duplicates_dropped}")
    # st.markdown(f"- Invalid/missing dates dropped: {dates_dropped}")
    # st.markdown(f"- Zero/negative Sales, Cost, or Units dropped: {invalid_numeric_dropped}")
    # st.markdown(f"- Gross Profit mismatches dropped: {profit_mismatch_dropped}")
    # st.markdown(f"- Final cleaned rows: {len(df)}")

    return df

# Load and display cleaned dataset
df = load_data()



# ---------------------------------------------------
# PROFITABILITY METRICS CALCULATION
# ---------------------------------------------------

def add_row_level_features(df):
    df = df.copy()
    df["Gross Margin"] = df["Gross Profit"] / df["Sales"]
    df["Profit per Unit"] = df["Gross Profit"] / df["Units"]
    return df

df = add_row_level_features(df)


# ---------------------------------------------------
# SIDEBAR NAVIGATION (Modules)
# ---------------------------------------------------

with st.sidebar.expander("Modules", expanded=False):  # Keep modules separate
    selected_module = st.radio(
        "Select Module",
        [
            "Overview", 
            "Product Profitability Overview",
            "Division Performance",
            "Cost Diagnostics",
            "Pareto Analysis"
        ],
        label_visibility="collapsed"
    )


# ---------------------------------------------------
# FILTERS (Sidebar)
# ---------------------------------------------------

with st.sidebar.expander("Filters", expanded=True):  # Move filters to sidebar
    min_date = df["Order Date"].min()
    max_date = df["Order Date"].max()



    # Product Search
    product_list = sorted(df["Product Name"].unique())
    selected_product = st.selectbox(
        "Product Search",
        options=["All Products"] + product_list
    )
    # Division Filter
    division_list = sorted(df["Division"].unique())
    selected_division = st.selectbox(
        "Division", 
        options=["All Divisions"] + division_list
    )

    # Margin Threshold
    margin_threshold = st.slider(
        "Margin Risk Threshold (%)",
        0, 100, 0
    )

    # Date Range
    date_range = st.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )


# ---------------------------------------------------
# SAFE FILTERING
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

# Apply division filter only if specific division selected
if selected_division != "All Divisions":
    filtered_df = filtered_df[
        filtered_df["Division"] == selected_division
    ]
# Apply margin threshold filter
if margin_threshold > 0:
    filtered_df = filtered_df[
        (filtered_df["Gross Margin"] * 100) >= margin_threshold
    ]

# Apply product filter only if specific product selected
if selected_product != "All Products":
    filtered_df = filtered_df[
        filtered_df["Product Name"] == selected_product
    ]

# Add Empty Data Protection
if filtered_df.empty:
    st.warning("No data available for selected filters.")
    st.stop()


# ---------------------------------------------------
# KPI CALCULATIONS
# ---------------------------------------------------

def calculate_kpis(filtered_df):

    # -----------------------------
    # Global KPIs
    # -----------------------------
    total_sales = filtered_df["Sales"].sum()
    total_profit = filtered_df["Gross Profit"].sum()
    total_units = filtered_df["Units"].sum()

    gross_margin = total_profit / total_sales if total_sales else 0
    profit_per_unit = total_profit / total_units if total_units else 0

    # -----------------------------
    # Product Summary Calculations
    # -----------------------------
    product_summary = (
        filtered_df.groupby("Product Name")
        .agg({
            "Sales": "sum",
            "Gross Profit": "sum",
            "Units": "sum"
        })
        .reset_index()
    )

    product_summary["Gross Margin (%)"] = (
        product_summary["Gross Profit"] / product_summary["Sales"] * 100
    ).fillna(0)

    product_summary["Profit per Unit"] = (
        product_summary["Gross Profit"] / product_summary["Units"]
    ).fillna(0)
    if total_sales:
        product_summary["Revenue Contribution (%)"] = (
            product_summary["Sales"] / total_sales * 100
        )
    else:
        product_summary["Revenue Contribution (%)"] = 0

    if total_profit:
        product_summary["Profit Contribution (%)"] = (
            product_summary["Gross Profit"] / total_profit * 100
        )
    else:
        product_summary["Profit Contribution (%)"] = 0


    # -----------------------------
    # Division-Level Aggregation
    # -----------------------------
    division_summary = (
        filtered_df
        .groupby("Division")
        .agg({
            "Sales": "sum",
            "Gross Profit": "sum",
            "Cost": "sum"
        })
        .reset_index()
    )

    division_summary["Gross Margin (%)"] = np.where(
        division_summary["Sales"] > 0,
        division_summary["Gross Profit"] / division_summary["Sales"] * 100,
        0
    )

    if total_sales:
        division_summary["Revenue Contribution (%)"] = (
            division_summary["Sales"] / total_sales * 100
        )
    else:
        division_summary["Revenue Contribution (%)"] = 0

    if total_profit:
        division_summary["Profit Contribution (%)"] = (
            division_summary["Gross Profit"] / total_profit * 100
        )
    else:
        division_summary["Profit Contribution (%)"] = 0

    division_summary["Profit-Revenue Gap (%)"] = (
        division_summary["Profit Contribution (%)"] -
        division_summary["Revenue Contribution (%)"]
    )

    division_summary["Performance Flag"] = np.where(
        division_summary["Profit-Revenue Gap (%)"] > 0,
        "Overperforming",
        np.where(
            division_summary["Profit-Revenue Gap (%)"] < 0,
            "Underperforming",
            "Neutral"
        )
    )
    # -----------------------------
    # Margin Volatility
    # -----------------------------
    monthly_df = (
        filtered_df
        .set_index("Order Date")
        .resample("ME")
        .agg({"Sales": "sum", "Gross Profit": "sum"})
        .reset_index()
    )

    monthly_df = monthly_df[monthly_df["Sales"] > 0]

    monthly_df["Margin %"] = (
        monthly_df["Gross Profit"] / monthly_df["Sales"] * 100
    )

    if len(monthly_df) > 1:
        margin_volatility = monthly_df["Margin %"].std()
    else:
        margin_volatility = 0

    monthly_df["Rolling Volatility"] = (
        monthly_df["Margin %"].rolling(window=3).std()
    )
    monthly_df["Rolling Volatility"] = (
        monthly_df["Rolling Volatility"]
        .fillna(0)
    )
    return {
        "total_sales": total_sales,
        "total_profit": total_profit,
        "total_units": total_units,
        "gross_margin": gross_margin,
        "profit_per_unit": profit_per_unit,
        "product_summary": product_summary,
        "division_summary": division_summary,
        "margin_volatility": margin_volatility,
        "monthly_df": monthly_df
    }


# ---------------------------------------------------
# KEY PERFORMANCE INDICATORS & TRENDS (OVERVIEW)
# ---------------------------------------------------

if selected_module == "Overview":
    kpi_data = calculate_kpis(filtered_df)
    monthly_df = kpi_data["monthly_df"]
    margin_volatility = kpi_data["margin_volatility"]
    gross_margin = kpi_data["gross_margin"]*100
    total_sales = kpi_data["total_sales"]
    total_profit = kpi_data["total_profit"]
    profit_per_unit = kpi_data["profit_per_unit"]
    total_units = kpi_data["total_units"]

    st.markdown("### Key Performance Indicators")

    # ===================================================
    # ===================================================
    # ROW 1 – KPI REPRESENTATIONS
    # ===================================================

    kpi1, spacer1, kpi2, spacer2, kpi3 = st.columns([1, 0.14, 1, 0.1, 1])

    # ---------------------------------------------------
    # KPI 1 – GROSS MARGIN (%)
    # ---------------------------------------------------

    with kpi1:

        # Title WITH VALUE
        st.markdown(f"**Gross Margin: {gross_margin:.1f}%**")
        left, right = st.columns([1, 1.2])

        # LEFT SIDE TEXT
        with left:
            st.markdown(f"**Total Sales**  \n${total_sales:,.0f}")
            st.markdown(f"**Total Gross Profit**  \n${total_profit:,.0f}")

        # RIGHT SIDE DONUT
        with right:

            fig_margin = go.Figure(go.Pie(
                values=[gross_margin, 100 - gross_margin],
                hole=0.75,
                textinfo="none"
            ))

            fig_margin.update_layout(
                height=140,
                margin=dict(t=0, b=0, l=0, r=0),
                showlegend=False,
                annotations=[dict(
                    text=f"<b>{gross_margin:.1f}%</b>",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=14)
                )],
                font=dict(size=9)
            )

            st.plotly_chart(fig_margin, width='stretch')


    # ---------------------------------------------------
    # KPI 2 – PROFIT PER UNIT
    # ---------------------------------------------------

    with kpi2:

        # Title top-left
        st.markdown(f"**Profit per Unit: ${profit_per_unit:,.2f}**")

        # Units Sold below title
        st.markdown(f"**Units Sold**  \n{total_units:,.0f}")

        # gauge_max = profit_per_unit * 1.5 if profit_per_unit > 0 else 10
        gauge_max = max(profit_per_unit * 1.3, 3)

        fig_ppu = go.Figure(go.Indicator(
            mode="gauge+number",
            value=profit_per_unit,
            number={'font': {'size': 14}},
            gauge={
                'axis': {'range': [0, gauge_max]},
                'bar': {'color': '#4C72B0', 'thickness': 0.6}
            }
        ))

        fig_ppu.update_layout(
            height=80,
            margin=dict(t=0, b=0, l=0, r=0),
            font=dict(size=9)
        )

        st.plotly_chart(fig_ppu, width='stretch')


    # ---------------------------------------------------
    # KPI 3 – MARGIN VOLATILITY
    # ---------------------------------------------------



    with kpi3:

        st.markdown("**Margin Volatility (Std Dev)**")

        st.metric(
            label="",
            value=f"{margin_volatility:.2f}%",
            help="Standard deviation of monthly margin %. Lower indicates more stable profitability."
        )

        # -----------------------------
        # Mini Rolling Volatility Chart
        # -----------------------------

        fig_vol = go.Figure()

        fig_vol.add_trace(go.Scatter(
            x=monthly_df["Order Date"],
            y=monthly_df["Rolling Volatility"],
            mode="lines",
            line=dict(width=2),
            showlegend=False
        ))

        fig_vol.update_layout(
            height=80,
            margin=dict(t=10, b=10, l=10, r=10),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig_vol, width='stretch')

    # ===================================================
    # ROW 2 – REVENUE VS PROFIT TREND (Margin Drivers)
    # ===================================================

    # Monthly aggregation
    monthly_trend = (
        filtered_df
            .set_index("Order Date")
            .resample("ME")
            .agg({
                "Sales": "sum",
                "Gross Profit": "sum"
            })
            .reset_index()
    )

    fig_driver = go.Figure()

    fig_driver.add_trace(go.Scatter(
        x=monthly_trend["Order Date"],
        y=monthly_trend["Sales"],
        mode="lines+markers",
        name="Revenue"
    ))

    fig_driver.add_trace(go.Scatter(
        x=monthly_trend["Order Date"],
        y=monthly_trend["Gross Profit"],
        mode="lines+markers",
        name="Gross Profit"
    ))

    fig_driver.update_layout(
        title="Revenue vs Profit Trend (Margin Drivers)",
        height=300,
        hovermode="x unified",
        yaxis_title="Amount ($)",
        xaxis_title=None
    )

    st.plotly_chart(fig_driver, width='stretch')


# ===================================================
# TAB 1 – PRODUCT PROFITABILITY OVERVIEW
# ===================================================

elif selected_module == "Product Profitability Overview":

    st.markdown("### Product Profitability Overview")
    # st.caption("Tip: Hover over charts to view exact values.")
    # ---------------------------------------------
    # Get KPI Data (Modular Architecture)
    # ---------------------------------------------
    kpi_data = calculate_kpis(filtered_df)
    product_summary = kpi_data["product_summary"]
    total_products = product_summary.shape[0]
    avg_margin = product_summary["Gross Margin (%)"].mean()
    top_product = product_summary.sort_values("Gross Profit", ascending=False)["Product Name"].iloc[0]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Products", total_products)

    with col2:
        st.metric("Avg Product Margin", f"{avg_margin:.2f}%")

    with col3:
        st.metric("Top Profit Product", " ")
        st.write(top_product)
    st.divider()

    # ---------------------------------------------
    # Leaderboard Controls
    # ---------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        metric_choice = st.selectbox(
            "Select Leaderboard Ranking Metric",
            ["Gross Margin (%)", "Gross Profit", "Sales", "Units", "Profit per Unit"]
        )

    with col2:
        leaderboard_size = st.slider(
            "Select Number of Products",
            5, 40, 15
        )

    leaderboard = (
        product_summary
        .sort_values(metric_choice, ascending=False)
        .head(leaderboard_size)
    )

    # ===================================================
    # 1️⃣ Leaderboard – Horizontal Bar
    # ===================================================

    fig_leaderboard = go.Figure()

    fig_leaderboard.add_trace(go.Bar(
        x=leaderboard[metric_choice],
        y=leaderboard["Product Name"],
        orientation="h"
    ))

    fig_leaderboard.update_layout(
        title=f"Product-level {metric_choice} Leaderboard",
        height=400,
        yaxis=dict(autorange="reversed"),
        xaxis_title=metric_choice,
        margin=dict(l=10, r=10, t=40, b=10)
    )

    st.plotly_chart(fig_leaderboard, width="stretch")

    # ===================================================
    # 2️⃣ Revenue & Profit Contribution – Grouped Bar
    # ===================================================

    contribution = (
        product_summary
        .sort_values("Gross Profit", ascending=False)
        .head(leaderboard_size)
    )

    fig_contribution = go.Figure()

    fig_contribution.add_trace(go.Bar(
        x=contribution["Revenue Contribution (%)"],
        y=contribution["Product Name"],
        name="Revenue %",
        orientation="h"
    ))

    fig_contribution.add_trace(go.Bar(
        x=contribution["Profit Contribution (%)"],
        y=contribution["Product Name"],
        name="Profit %",
        orientation="h"
    ))

    fig_contribution.update_layout(
        title="Revenue vs Profit Contribution (%)",
        barmode="group",
        height=400,
        yaxis=dict(autorange="reversed"),
        xaxis_title="Contribution (%)",
        margin=dict(l=10, r=10, t=40, b=10)
    )

    st.plotly_chart(fig_contribution, width="stretch")

    # ==============================================================
    # 3️⃣ Product Portfolio Positioning/Map Revenue vs Margin Scatter
    # ==============================================================

    product_summary = product_summary.copy()

    avg_sales = product_summary["Sales"].mean()
    avg_margin = product_summary["Gross Margin (%)"].mean()

    # Assign Quadrant Category
    product_summary["Quadrant"] = "Other"

    product_summary.loc[
        (product_summary["Sales"] >= avg_sales) &
        (product_summary["Gross Margin (%)"] >= avg_margin),
        "Quadrant"
    ] = "High Revenue / High Margin"

    product_summary.loc[
        (product_summary["Sales"] >= avg_sales) &
        (product_summary["Gross Margin (%)"] < avg_margin),
        "Quadrant"
    ] = "High Revenue / Low Margin"

    product_summary.loc[
        (product_summary["Sales"] < avg_sales) &
        (product_summary["Gross Margin (%)"] >= avg_margin),
        "Quadrant"
    ] = "Low Revenue / High Margin"

    product_summary.loc[
        (product_summary["Sales"] < avg_sales) &
        (product_summary["Gross Margin (%)"] < avg_margin),
        "Quadrant"
    ] = "Low Revenue / Low Margin"


    fig_scatter = px.scatter(
        product_summary,
        x="Sales",
        y="Gross Margin (%)",
        color="Quadrant",
        hover_name="Product Name",
        height=450
    )

    fig_scatter.add_vline(x=avg_sales, line_dash="dash", line_width=1)
    fig_scatter.add_hline(y=avg_margin, line_dash="dash", line_width=1)

    fig_scatter.update_layout(
        title="Product Portfolio Positioning: Revenue vs Margin",
        xaxis_title="Total Revenue ($)",
        yaxis_title="Gross Margin (%)",
        legend_title="Portfolio Position",
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig_scatter, width="stretch")


# ===================================================
# TAB 2 – DIVISION PERFORMANCE
# ===================================================

elif selected_module == "Division Performance":

    st.markdown("### Division Performance Overview")
    # ---------------------------------------------
    # KPI Data
    # ---------------------------------------------
    kpi_data = calculate_kpis(filtered_df)
    division_summary = kpi_data["division_summary"]

    division_summary = division_summary.sort_values(
        "Revenue Contribution (%)",
        ascending=False
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Top Division (Revenue)", division_summary.iloc[0]["Division"])

    with col2:
        st.metric("Top Revenue Share", f"{division_summary.iloc[0]['Revenue Contribution (%)']:.1f}%")

    with col3:
        st.metric("Top Profit Share", f"{division_summary.iloc[0]['Profit Contribution (%)']:.1f}%")
    
    st.divider()


    # ===================================================
    # 1️⃣ Revenue vs Profit Contribution (Strategic View)
    # ===================================================

    fig_contribution = go.Figure()

    fig_contribution.add_trace(go.Bar(
        x=division_summary["Division"],
        y=division_summary["Revenue Contribution (%)"],
        name="Revenue %"
    ))

    fig_contribution.add_trace(go.Bar(
        x=division_summary["Division"],
        y=division_summary["Profit Contribution (%)"],
        name="Profit %"
    ))

    fig_contribution.update_traces(
        texttemplate="%{y:.1f}%",
        textposition="outside",
        cliponaxis=False
    )

    fig_contribution.update_layout(
        title="Revenue vs Profit Contribution (%)",
        barmode="group",
        height=420,
        yaxis_title="Contribution (%)",
        hovermode="x unified"
    )

    st.plotly_chart(fig_contribution, width="stretch")

    # ===================================================
    # 2️⃣ Profit-Revenue Gap (Efficiency Signal)
    # ===================================================

    fig_gap = px.bar(
        division_summary,
        x="Division",
        y="Profit-Revenue Gap (%)",
        color="Performance Flag",
        color_discrete_map={
            "Overperforming": "green",
            "Underperforming": "red",
            "Neutral": "gray"
        }
    )

    fig_gap.add_hline(y=0)

    fig_gap.update_traces(
        text=division_summary["Profit-Revenue Gap (%)"]
            .round(2)
            .astype(str) + "%",
        textposition="outside",
        cliponaxis=False
    )

    fig_gap.update_layout(
        title="Profit-Revenue Gap (Efficiency Indicator)",
        height=400,
        yaxis_title="Gap (%)"
    )

    st.plotly_chart(fig_gap, width="stretch")

    # ===================================================
    # 3️⃣ Gross Margin Quality
    # ===================================================

    sorted_margin = division_summary.sort_values(
        "Gross Margin (%)",
        ascending=False
    )

    fig_margin = px.bar(
        sorted_margin,
        x="Division",
        y="Gross Margin (%)"
    )

    fig_margin.update_traces(
        text=sorted_margin["Gross Margin (%)"]
            .round(2)
            .astype(str) + "%",
        textposition="outside",
        cliponaxis=False
    )

    fig_margin.update_layout(
        title="Gross Margin (%) by Division",
        height=400,
        yaxis_title="Gross Margin (%)"
    )

    st.plotly_chart(fig_margin, width="stretch")

    # ===================================================
    # Executive Diagnostic Table
    # ===================================================

    st.markdown("### Division Financial Diagnostics")

    st.dataframe(
        division_summary[
            [
                "Division",
                "Gross Margin (%)",
                "Revenue Contribution (%)",
                "Profit Contribution (%)",
                "Profit-Revenue Gap (%)",
                "Performance Flag"
            ]
        ].style.format({
            "Gross Margin (%)": "{:.2f}%",
            "Revenue Contribution (%)": "{:.2f}%",
            "Profit Contribution (%)": "{:.2f}%",
            "Profit-Revenue Gap (%)": "{:.2f}%"
        })
    )


    

# ===================================================
# TAB 3 – COST DIAGNOSTICS
# ===================================================

elif selected_module == "Cost Diagnostics":

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

elif selected_module == "Pareto Analysis":

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

