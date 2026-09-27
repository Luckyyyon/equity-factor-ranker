import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as gg
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.universe_loader import load_universe_data, DEFAULT_UNIVERSE
from src.factor_library import compute_raw_factors
from src.cross_sectional import standardize_factors
from src.composite_scorer import compute_composite_scores, DEFAULT_FACTOR_WEIGHTS
from src.backtest_dispersion import analyze_decile_dispersion

# Page Config
st.set_page_config(
    page_title="Quantitative Equity Research | Institutional Factor Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark institutional custom CSS styling
st.markdown("""
<style>
    /* Dark institutional layout styling */
    .stApp {
        background-color: #0b0e14;
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #141b2d;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .metric-title {
        font-size: 0.82rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 4px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #141b2d;
        padding: 6px 12px;
        border-radius: 10px;
        border: 1px solid #1e293b;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        white-space: pre;
        border-radius: 6px;
        color: #94a3b8;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563eb !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# App Title & Header
st.markdown('<div class="main-header">Institutional Equity Screening & Multi-Factor Ranking Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Cross-Sectional Factor Normalization, Sector-Neutral Z-Scores & Long/Short Decile Dispersion Analytics</div>', unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("Model Configuration")

# Universe Selector
universe_input = st.sidebar.text_area(
    "Equity Universe Tickers (Comma separated)",
    value=", ".join(DEFAULT_UNIVERSE),
    height=120
)
tickers = [t.strip().upper() for t in universe_input.split(",") if t.strip()]

# Factor Normalization Settings
st.sidebar.subheader("Standardization Settings")
sector_neutral = st.sidebar.checkbox("Sector-Neutral Z-Scores", value=True, help="Compute Z-Scores relative to sector peers rather than whole universe")
winsorize_bounds = st.sidebar.slider("Winsorization Tails (%)", 0.0, 5.0, (2.5, 97.5), step=0.5)

# Factor Weight Sliders
st.sidebar.subheader("Category Factor Weights")
w_value = st.sidebar.slider("Value Factor Weight", 0.0, 1.0, 0.35, step=0.05)
w_quality = st.sidebar.slider("Quality Factor Weight", 0.0, 1.0, 0.35, step=0.05)
w_momentum = st.sidebar.slider("Momentum Factor Weight", 0.0, 1.0, 0.30, step=0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("Sub-Factor Weights")

with st.sidebar.expander("Value Factors"):
    w_ebitda_ev = st.slider("EBITDA / EV", 0.0, 1.0, 0.40, step=0.05)
    w_fcf_yield = st.slider("FCF Yield", 0.0, 1.0, 0.40, step=0.05)
    w_earn_yield = st.slider("Earnings Yield (1/PE)", 0.0, 1.0, 0.20, step=0.05)

with st.sidebar.expander("Quality Factors"):
    w_roic = st.slider("ROIC", 0.0, 1.0, 0.40, step=0.05)
    w_op_margin = st.slider("Operating Margin", 0.0, 1.0, 0.35, step=0.05)
    w_fin_health = st.slider("Financial Health (-NetDebt/EBITDA)", 0.0, 1.0, 0.25, step=0.05)

with st.sidebar.expander("Momentum Factors"):
    w_mom_12m1m = st.slider("12M - 1M Momentum", 0.0, 1.0, 0.60, step=0.05)
    w_mom_6m = st.slider("6M Momentum", 0.0, 1.0, 0.40, step=0.05)

# Load Data Button / Cache
@st.cache_data(ttl=1800, show_spinner="Fetching fundamental and market pricing data...")
def get_cached_universe_data(ticker_list):
    return load_universe_data(ticker_list)

raw_universe_df, prices_dict = get_cached_universe_data(tuple(tickers))

if raw_universe_df.empty:
    st.error("Failed to load universe data or no valid data returned for specified tickers.")
    st.stop()

# Compute raw factors
raw_factors = compute_raw_factors(raw_universe_df)

# Standardize factors
factor_columns = [
    'Value_EBITDA_EV', 'Value_FCF_Yield', 'Value_Earnings_Yield',
    'Quality_ROIC', 'Quality_Op_Margin', 'Quality_Fin_Health',
    'Mom_12M_1M', 'Mom_6M'
]

std_factors = standardize_factors(
    raw_factors,
    factor_cols=factor_columns,
    sector_neutral=sector_neutral,
    lower_q=winsorize_bounds[0] / 100.0,
    upper_q=winsorize_bounds[1] / 100.0
)

# Custom Weights dict
user_weights = {
    'Value': w_value,
    'Quality': w_quality,
    'Momentum': w_momentum,
    'Value_EBITDA_EV': w_ebitda_ev,
    'Value_FCF_Yield': w_fcf_yield,
    'Value_Earnings_Yield': w_earn_yield,
    'Quality_ROIC': w_roic,
    'Quality_Op_Margin': w_op_margin,
    'Quality_Fin_Health': w_fin_health,
    'Mom_12M_1M': w_mom_12m1m,
    'Mom_6M': w_mom_6m
}

# Calculate composite ranking
ranked_df = compute_composite_scores(std_factors, user_weights)

# Main Dashboard Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Composite Factor Screener",
    "Factor Heatmaps & Z-Scores",
    "Long/Short Decile Dispersion",
    "Factor Model Specs"
])

# TAB 1: Composite Factor Screener
with tab1:
    st.markdown("### Universe Ranking & Composite Score Card")
    
    # Top Metrics Bar
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Active Universe</div>
            <div class="metric-value">{len(ranked_df)} Stocks</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        top_stock = ranked_df.index[0] if len(ranked_df) > 0 else "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Top Ranked Stock</div>
            <div class="metric-value">{top_stock}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        bottom_stock = ranked_df.index[-1] if len(ranked_df) > 0 else "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Lowest Ranked Stock</div>
            <div class="metric-value">{bottom_stock}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        avg_z = ranked_df['Composite_Z'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Universe Mean Z-Score</div>
            <div class="metric-value">{avg_z:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Sector Filter & Search
    c_sec, c_search = st.columns([1, 2])
    sectors = ["All Sectors"] + sorted(list(ranked_df['Sector'].dropna().unique()))
    selected_sector = c_sec.selectbox("Filter Sector", sectors)
    search_query = c_search.text_input("Search Ticker or Company Name", "")

    filtered_df = ranked_df.copy()
    if selected_sector != "All Sectors":
        filtered_df = filtered_df[filtered_df['Sector'] == selected_sector]
    if search_query.strip():
        q = search_query.strip().lower()
        filtered_df = filtered_df[
            filtered_df.index.str.lower().str.contains(q) | 
            filtered_df['Name'].str.lower().str.contains(q)
        ]

    # Data Table View
    display_cols = [
        'Name', 'Sector', 'Percentile_Rank', 'Composite_Z',
        'Value_Composite_Z', 'Quality_Composite_Z', 'Momentum_Composite_Z'
    ]
    
    st.dataframe(
        filtered_df[display_cols].style.background_gradient(
            cmap='Blues', subset=['Percentile_Rank', 'Composite_Z']
        ).format({
            'Percentile_Rank': '{:.1f}%',
            'Composite_Z': '{:.2f}',
            'Value_Composite_Z': '{:.2f}',
            'Quality_Composite_Z': '{:.2f}',
            'Momentum_Composite_Z': '{:.2f}'
        }),
        use_container_width=True,
        height=500
    )

    # Top/Bottom Decile Breakdown Chart
    st.markdown("### Top Decile vs Bottom Decile Factor Profile")
    n_top = max(1, int(len(ranked_df) * 0.2))
    top_decile = ranked_df.iloc[:n_top]
    bottom_decile = ranked_df.iloc[-n_top:]

    profile_df = pd.DataFrame({
        'Top 20% Basket': [
            top_decile['Value_Composite_Z'].mean(),
            top_decile['Quality_Composite_Z'].mean(),
            top_decile['Momentum_Composite_Z'].mean()
        ],
        'Bottom 20% Basket': [
            bottom_decile['Value_Composite_Z'].mean(),
            bottom_decile['Quality_Composite_Z'].mean(),
            bottom_decile['Momentum_Composite_Z'].mean()
        ]
    }, index=['Value Factor Z', 'Quality Factor Z', 'Momentum Factor Z'])

    fig_profile = px.bar(
        profile_df, 
        barmode='group',
        labels={'value': 'Mean Z-Score', 'index': 'Factor Category'},
        color_discrete_sequence=['#3b82f6', '#ef4444'],
        title="Factor Exposure Comparison (Top Basket vs. Bottom Basket)"
    )
    fig_profile.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig_profile, use_container_width=True)


# TAB 2: Factor Heatmaps & Cross-Sectional Z-Scores
with tab2:
    st.markdown("### Cross-Sectional Z-Score Factor Breakdown")

    # Heatmap of Z-Scores
    z_score_cols = [
        'Z_Value_EBITDA_EV', 'Z_Value_FCF_Yield', 'Z_Value_Earnings_Yield',
        'Z_Quality_ROIC', 'Z_Quality_Op_Margin', 'Z_Quality_Fin_Health',
        'Z_Mom_12M_1M', 'Z_Mom_6M'
    ]

    heatmap_data = ranked_df[z_score_cols].copy()
    heatmap_data.columns = [c.replace('Z_', '') for c in z_score_cols]

    fig_heat = px.imshow(
        heatmap_data,
        labels=dict(x="Factor Metric", y="Ticker", color="Z-Score"),
        x=heatmap_data.columns,
        y=heatmap_data.index,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title="Cross-Sectional Z-Score Heatmap (Normalized & Sector Neutral)"
    )
    fig_heat.update_layout(template="plotly_dark", height=650)
    st.plotly_chart(fig_heat, use_container_width=True)

    # Individual Factor Scatter Analysis
    st.markdown("### Factor vs. Factor Multi-Dimensional Scatter")
    c_x, c_y = st.columns(2)
    x_factor = c_x.selectbox("X-Axis Factor Z-Score", ['Value_Composite_Z', 'Quality_Composite_Z', 'Momentum_Composite_Z'], index=0)
    y_factor = c_y.selectbox("Y-Axis Factor Z-Score", ['Value_Composite_Z', 'Quality_Composite_Z', 'Momentum_Composite_Z'], index=1)

    fig_scatter = px.scatter(
        ranked_df,
        x=x_factor,
        y=y_factor,
        color="Sector",
        size="Percentile_Rank",
        hover_name="Name",
        text=ranked_df.index,
        title=f"{x_factor} vs. {y_factor} Universe Scatter"
    )
    fig_scatter.update_traces(textposition='top center')
    fig_scatter.update_layout(template="plotly_dark", height=550)
    st.plotly_chart(fig_scatter, use_container_width=True)


# TAB 3: Long/Short Decile Dispersion
with tab3:
    st.markdown("### Historical Backtest Decile Dispersion & Spread Performance")
    
    col_pct, _ = st.columns([1, 2])
    decile_slice = col_pct.slider("Basket Cutoff Percentile", 0.10, 0.30, 0.20, step=0.05, help="Top/Bottom percentage basket size (e.g. 0.20 = Top 20% vs Bottom 20%)")

    summary_stats, ts_returns = analyze_decile_dispersion(ranked_df, prices_dict, decile_pct=decile_slice)

    if not ts_returns.empty:
        # Performance Chart
        fig_disp = px.line(
            ts_returns,
            x=ts_returns.index,
            y=ts_returns.columns,
            title="Cumulative Return Spread Performance: Top Basket vs Bottom Basket vs Equal-Weighted Benchmark",
            labels={'value': 'Growth of $1.00', 'index': 'Date'},
            color_discrete_map={
                'Top Basket (Long)': '#22c55e',
                'Bottom Basket (Short)': '#ef4444',
                'Universe Benchmark (EQW)': '#94a3b8',
                'Long/Short Spread': '#3b82f6'
            }
        )
        fig_disp.update_layout(template="plotly_dark", height=500, hovermode="x unified")
        st.plotly_chart(fig_disp, use_container_width=True)

        # Performance Summary Table
        st.markdown("### Basket Risk & Return Performance Metrics")
        st.dataframe(
            summary_stats.style.format({
                'Total Return': '{:.2%}',
                'Annualized Return': '{:.2%}',
                'Annualized Volatility': '{:.2%}',
                'Sharpe Ratio (Rf=3%)': '{:.2f}',
                'Max Drawdown': '{:.2%}'
            }).background_gradient(cmap='Greens', subset=['Total Return', 'Sharpe Ratio (Rf=3%)']),
            use_container_width=True
        )
    else:
        st.warning("Insufficient historical price data to run decile dispersion backtest.")


# TAB 4: Factor Model Specs
with tab4:
    st.markdown("""
    ### Quantitative Factor Model Specifications

    #### 1. Factor Definitions & Directionality
    - **Value Factors:**
      - **EBITDA / EV**: Operating cash generation before depreciation relative to Enterprise Value. Higher is cheaper.
      - **FCF Yield**: Free Cash Flow divided by Market Cap. Higher indicates stronger cash return.
      - **Earnings Yield**: Inverted P/E ratio ($1 / PE$). Higher indicates cheaper valuation.
    - **Quality Factors:**
      - **ROIC**: Return on Invested Capital ($NOPAT / Enterprise Value$). Measures capital efficiency.
      - **Operating Margin**: Operating Profitability relative to Revenue.
      - **Financial Health**: Inverted Net Debt / EBITDA ($-1 \times NetDebt / EBITDA$). Higher score indicates lower debt leverage.
    - **Momentum Factors:**
      - **12M - 1M Momentum**: Price Return from $t-252$ to $t-21$ trading days (skipping trailing 1 month to eliminate short-term reversal bias).
      - **6M Momentum**: Trailing 126 trading day cumulative price performance.

    #### 2. Cross-Sectional Normalization & Winsorization
    - **Winsorization**: Extremes capped at top and bottom tails (e.g. 2.5% and 97.5%) to eliminate single-stock outlier distortion.
    - **Sector Neutral Z-Scores**: Computed per sector $s$:
      $$Z_{i, s} = \\frac{X_{i, s} - \\mu_{s}}{\\sigma_{s}}$$
      Eliminates structural sector biases (e.g., tech always having higher P/E than utilities).
    - **Composite Percentile Ranking**: Weighted sum of category Z-scores scaled to a percentile range $[0, 100]$.
    """)

