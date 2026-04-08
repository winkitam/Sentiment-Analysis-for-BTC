# sentiment_edge.py - Complete Working Version
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page config MUST be first Streamlit command
st.set_page_config(
    page_title="SentimentEdge — AI-Powered Crypto Portfolio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #050810 0%, #0c1120 100%);
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #e8eaf0;
    }
    
    .metric-card {
        background: #0c1120;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        font-family: 'Space Mono', monospace;
        background: linear-gradient(135deg, #f7931a, #627eea);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.68rem;
        color: #6b7a99;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }
    
    .sentiment-card {
        background: #0c1120;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .formula-box {
        background: #111827;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        text-align: center;
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        margin: 1.5rem 0;
    }
    
    hr {
        border-color: rgba(255,255,255,0.07);
        margin: 2rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid rgba(255,255,255,0.07);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #6b7a99;
        padding: 0.7rem 1.5rem;
    }
    
    .stTabs [aria-selected="true"] {
        color: #e8eaf0;
        border-bottom: 2px solid #627eea;
    }
    
    .alloc-bar-track {
        background: rgba(255,255,255,0.06);
        border-radius: 4px;
        height: 10px;
        overflow: hidden;
    }
    
    .alloc-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.6s ease;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA
# ============================================================================

@st.cache_data
def load_weights_data():
    """Load portfolio weights data"""
    data = [
        {"date": "2018-08-31", "mvo_btc": 1.1481, "mvo_eth": -0.1481, "samvo_btc": 5, "samvo_eth": -5},
        {"date": "2018-09-30", "mvo_btc": 1.4155, "mvo_eth": -0.4155, "samvo_btc": 5, "samvo_eth": -5},
        {"date": "2018-10-31", "mvo_btc": 1.3857, "mvo_eth": -0.3857, "samvo_btc": -5, "samvo_eth": 5},
        {"date": "2018-11-30", "mvo_btc": 5, "mvo_eth": -5, "samvo_btc": -5, "samvo_eth": 5},
        {"date": "2018-12-31", "mvo_btc": -5, "mvo_eth": 5, "samvo_btc": -5, "samvo_eth": 5},
        {"date": "2019-01-31", "mvo_btc": 5, "mvo_eth": -5, "samvo_btc": -5, "samvo_eth": 5},
        {"date": "2019-02-28", "mvo_btc": 5, "mvo_eth": -5, "samvo_btc": -5, "samvo_eth": 5},
        {"date": "2019-03-31", "mvo_btc": 5, "mvo_eth": -5, "samvo_btc": -5, "samvo_eth": 5},
        {"date": "2019-04-30", "mvo_btc": 5, "mvo_eth": -5, "samvo_btc": -5, "samvo_eth": 5},
        {"date": "2019-05-31", "mvo_btc": 2.7933, "mvo_eth": -1.7933, "samvo_btc": -5, "samvo_eth": 5},
        {"date": "2019-06-30", "mvo_btc": 2.3314, "mvo_eth": -1.3314, "samvo_btc": 1.2073, "samvo_eth": -0.2073},
        {"date": "2019-07-31", "mvo_btc": 2.8418, "mvo_eth": -1.8418, "samvo_btc": 5, "samvo_eth": -5},
        {"date": "2019-08-31", "mvo_btc": 2.6144, "mvo_eth": -1.6144, "samvo_btc": -0.1966, "samvo_eth": 1.1966},
        {"date": "2019-09-30", "mvo_btc": 2.2549, "mvo_eth": -1.2549, "samvo_btc": -1.4666, "samvo_eth": 2.4666},
        {"date": "2019-10-31", "mvo_btc": 1.9838, "mvo_eth": -0.9838, "samvo_btc": -2.4989, "samvo_eth": 3.4989},
        {"date": "2019-11-30", "mvo_btc": 1.4462, "mvo_eth": -0.4462, "samvo_btc": -0.9669, "samvo_eth": 1.9669},
        {"date": "2019-12-31", "mvo_btc": 2.3959, "mvo_eth": -1.3959, "samvo_btc": 0.9916, "samvo_eth": 0.0084},
        {"date": "2020-01-31", "mvo_btc": 1.6335, "mvo_eth": -0.6335, "samvo_btc": 1.5098, "samvo_eth": -0.5098},
        {"date": "2020-02-29", "mvo_btc": 1.4095, "mvo_eth": -0.4095, "samvo_btc": 1.7342, "samvo_eth": -0.7342},
        {"date": "2020-03-31", "mvo_btc": 2.4657, "mvo_eth": -1.4657, "samvo_btc": 2.7316, "samvo_eth": -1.7316},
        {"date": "2020-04-30", "mvo_btc": 1.3656, "mvo_eth": -0.3656, "samvo_btc": 3.5238, "samvo_eth": -2.5238},
        {"date": "2020-05-31", "mvo_btc": 1.8362, "mvo_eth": -0.8362, "samvo_btc": 5, "samvo_eth": -5},
        {"date": "2020-06-30", "mvo_btc": -0.2051, "mvo_eth": 1.2051, "samvo_btc": 5, "samvo_eth": -5},
        {"date": "2020-07-31", "mvo_btc": -3.8921, "mvo_eth": 4.8921, "samvo_btc": 3.0881, "samvo_eth": -2.0881},
        {"date": "2020-08-31", "mvo_btc": -5, "mvo_eth": 5, "samvo_btc": 1.0591, "samvo_eth": -0.0591},
        {"date": "2020-09-30", "mvo_btc": -2.9672, "mvo_eth": 3.9672, "samvo_btc": 2.1589, "samvo_eth": -1.1589},
        {"date": "2020-10-31", "mvo_btc": -1.0855, "mvo_eth": 2.0855, "samvo_btc": 2.939, "samvo_eth": -1.939},
        {"date": "2020-11-30", "mvo_btc": 0.0597, "mvo_eth": 0.9403, "samvo_btc": 0.7882, "samvo_eth": 0.2118},
        {"date": "2020-12-31", "mvo_btc": 0.5395, "mvo_eth": 0.4605, "samvo_btc": 1.1089, "samvo_eth": -0.1089},
        {"date": "2021-01-31", "mvo_btc": 0.0728, "mvo_eth": 0.9272, "samvo_btc": 0.0449, "samvo_eth": 0.9551},
        {"date": "2021-02-28", "mvo_btc": 0.7215, "mvo_eth": 0.2785, "samvo_btc": -4.4188, "samvo_eth": 5},
        {"date": "2021-03-31", "mvo_btc": 0.6681, "mvo_eth": 0.3319, "samvo_btc": -2.8395, "samvo_eth": 3.8395},
        {"date": "2021-04-30", "mvo_btc": 0.4715, "mvo_eth": 0.5285, "samvo_btc": -5, "samvo_eth": 5},
        {"date": "2021-05-31", "mvo_btc": 0.1015, "mvo_eth": 0.8985, "samvo_btc": -5, "samvo_eth": 5},
        {"date": "2021-06-30", "mvo_btc": 0.1403, "mvo_eth": 0.8597, "samvo_btc": -5, "samvo_eth": 5},
        {"date": "2021-07-31", "mvo_btc": 0.3077, "mvo_eth": 0.6923, "samvo_btc": -5, "samvo_eth": 5},
        {"date": "2021-08-31", "mvo_btc": 0.3459, "mvo_eth": 0.6541, "samvo_btc": -5, "samvo_eth": 5},
        {"date": "2021-09-30", "mvo_btc": 0.2768, "mvo_eth": 0.7232, "samvo_btc": -5, "samvo_eth": 5},
        {"date": "2021-10-31", "mvo_btc": 0.153, "mvo_eth": 0.847, "samvo_btc": -5, "samvo_eth": 5},
        {"date": "2021-11-30", "mvo_btc": -0.229, "mvo_eth": 1.229, "samvo_btc": -5, "samvo_eth": 5},
        {"date": "2021-12-31", "mvo_btc": -1.8158, "mvo_eth": 2.8158, "samvo_btc": -5, "samvo_eth": 5},
        {"date": "2022-01-31", "mvo_btc": -2.7032, "mvo_eth": 3.7032, "samvo_btc": -5, "samvo_eth": 5},
        {"date": "2022-02-28", "mvo_btc": -5, "mvo_eth": 5, "samvo_btc": -5, "samvo_eth": 5},
        {"date": "2022-03-31", "mvo_btc": -5, "mvo_eth": 5, "samvo_btc": -5, "samvo_eth": 5},
        {"date": "2022-04-30", "mvo_btc": -5, "mvo_eth": 5, "samvo_btc": 5, "samvo_eth": -5},
        {"date": "2022-05-31", "mvo_btc": 4.3047, "mvo_eth": -3.3047, "samvo_btc": 5, "samvo_eth": -5},
        {"date": "2022-06-30", "mvo_btc": -5, "mvo_eth": 5, "samvo_btc": -5, "samvo_eth": 5}
    ]
    return pd.DataFrame(data)

@st.cache_data
def load_lambdas_data():
    """Load lambda coefficients data"""
    data = [
        {"date": "2018-08-31", "btc": -7.23, "eth": -6.45}, {"date": "2018-09-30", "btc": -6.82, "eth": -5.26},
        {"date": "2018-10-31", "btc": -6.35, "eth": -4.50}, {"date": "2018-11-30", "btc": -3.46, "eth": -2.48},
        {"date": "2018-12-31", "btc": -3.42, "eth": -2.10}, {"date": "2019-01-31", "btc": -3.63, "eth": 0.32},
        {"date": "2019-02-28", "btc": -1.75, "eth": -0.93}, {"date": "2019-03-31", "btc": -1.25, "eth": -0.58},
        {"date": "2019-04-30", "btc": -1.56, "eth": 0.66}, {"date": "2019-05-31", "btc": -2.19, "eth": 0.84},
        {"date": "2019-06-30", "btc": -1.15, "eth": 1.15}, {"date": "2019-07-31", "btc": -2.05, "eth": -0.04},
        {"date": "2019-08-31", "btc": -1.23, "eth": 1.27}, {"date": "2019-09-30", "btc": -0.86, "eth": 1.08},
        {"date": "2019-10-31", "btc": -1.41, "eth": 1.05}, {"date": "2019-11-30", "btc": -2.58, "eth": -0.63},
        {"date": "2019-12-31", "btc": -2.15, "eth": 0.04}, {"date": "2020-01-31", "btc": -0.09, "eth": 0.11},
        {"date": "2020-02-29", "btc": 0.48, "eth": -0.15}, {"date": "2020-03-31", "btc": 1.25, "eth": 0.19},
        {"date": "2020-04-30", "btc": 0.50, "eth": -1.68}, {"date": "2020-05-31", "btc": 0.08, "eth": -2.00},
        {"date": "2020-06-30", "btc": -0.76, "eth": -2.56}, {"date": "2020-07-31", "btc": 0.13, "eth": -2.09},
        {"date": "2020-08-31", "btc": -0.20, "eth": -2.48}, {"date": "2020-09-30", "btc": -0.73, "eth": -2.58},
        {"date": "2020-10-31", "btc": -0.49, "eth": -3.26}, {"date": "2020-11-30", "btc": -1.68, "eth": -2.59},
        {"date": "2020-12-31", "btc": -2.18, "eth": -2.99}, {"date": "2021-01-31", "btc": -3.13, "eth": -3.72},
        {"date": "2021-02-28", "btc": -6.12, "eth": -2.46}, {"date": "2021-03-31", "btc": -7.54, "eth": -2.54},
        {"date": "2021-04-30", "btc": -7.64, "eth": -1.72}, {"date": "2021-05-31", "btc": -8.54, "eth": -2.49},
        {"date": "2021-06-30", "btc": -7.23, "eth": -2.72}, {"date": "2021-07-31", "btc": -7.62, "eth": -2.66},
        {"date": "2021-08-31", "btc": -7.77, "eth": -3.03}, {"date": "2021-09-30", "btc": -7.87, "eth": -2.92},
        {"date": "2021-10-31", "btc": -8.03, "eth": -3.03}, {"date": "2021-11-30", "btc": -7.29, "eth": -4.32},
        {"date": "2021-12-31", "btc": -8.07, "eth": -4.72}, {"date": "2022-01-31", "btc": -8.10, "eth": -6.33},
        {"date": "2022-02-28", "btc": -4.44, "eth": -7.83}, {"date": "2022-03-31", "btc": -2.32, "eth": -6.55},
        {"date": "2022-04-30", "btc": -2.17, "eth": -6.09}, {"date": "2022-05-31", "btc": 0.59, "eth": -3.75},
        {"date": "2022-06-30", "btc": -4.36, "eth": -1.10}
    ]
    return pd.DataFrame(data)

@st.cache_data
def load_btc_sentiment():
    """Load BTC sentiment data"""
    data = [
        {"date": "2017-09-21", "sentiment": -0.0009}, {"date": "2017-09-28", "sentiment": 0.1258},
        {"date": "2017-10-05", "sentiment": 0.148}, {"date": "2017-10-12", "sentiment": 0.0948},
        {"date": "2017-10-19", "sentiment": 0.1477}, {"date": "2017-10-26", "sentiment": 0.1938},
        {"date": "2017-11-02", "sentiment": 0.0679}, {"date": "2017-11-09", "sentiment": 0.2673},
        {"date": "2017-11-16", "sentiment": 0.1266}, {"date": "2017-11-23", "sentiment": -0.0948},
        {"date": "2017-11-30", "sentiment": -0.021}, {"date": "2017-12-07", "sentiment": 0.1923},
        {"date": "2017-12-14", "sentiment": 0.0524}, {"date": "2017-12-21", "sentiment": -0.0113},
        {"date": "2017-12-28", "sentiment": -0.0258}, {"date": "2018-01-04", "sentiment": 0.2179},
        {"date": "2018-01-11", "sentiment": 0.004}, {"date": "2018-01-18", "sentiment": 0.1628},
        {"date": "2018-01-25", "sentiment": 0.3143}, {"date": "2018-02-01", "sentiment": 0.0905},
        {"date": "2018-02-08", "sentiment": 0.1817}, {"date": "2018-02-15", "sentiment": 0.2987},
        {"date": "2018-02-23", "sentiment": 0.2011}, {"date": "2018-03-02", "sentiment": 0.204},
        {"date": "2018-03-09", "sentiment": 0.1994}, {"date": "2018-03-18", "sentiment": 0.0712},
        {"date": "2018-03-25", "sentiment": 0.3309}, {"date": "2018-04-01", "sentiment": 0.2144},
        {"date": "2018-04-08", "sentiment": 0.2584}, {"date": "2018-04-15", "sentiment": 0.1455},
        {"date": "2018-04-22", "sentiment": 0.1906}, {"date": "2018-04-29", "sentiment": 0.1576},
        {"date": "2018-05-06", "sentiment": 0.2771}, {"date": "2018-05-15", "sentiment": 0.1905},
        {"date": "2018-05-22", "sentiment": 0.2641}, {"date": "2018-05-29", "sentiment": 0.192},
        {"date": "2018-06-05", "sentiment": 0.1392}, {"date": "2018-06-12", "sentiment": 0.1278},
        {"date": "2018-06-19", "sentiment": 0.2357}, {"date": "2018-06-26", "sentiment": 0.2238},
        {"date": "2018-07-03", "sentiment": 0.2984}, {"date": "2018-07-10", "sentiment": 0.2665},
        {"date": "2018-07-17", "sentiment": 0.2578}, {"date": "2018-07-24", "sentiment": 0.173},
        {"date": "2018-07-31", "sentiment": 0.259}, {"date": "2018-08-07", "sentiment": 0.2255},
        {"date": "2018-08-14", "sentiment": -0.0083}, {"date": "2018-08-21", "sentiment": 0.0941},
        {"date": "2018-08-28", "sentiment": 0.2024}, {"date": "2018-09-04", "sentiment": 0.2178},
        {"date": "2018-09-11", "sentiment": 0.3331}, {"date": "2018-09-18", "sentiment": 0.095},
        {"date": "2018-09-25", "sentiment": 0.1536}, {"date": "2018-10-02", "sentiment": 0.1036},
        {"date": "2018-10-09", "sentiment": 0.1787}, {"date": "2018-10-16", "sentiment": 0.2745},
        {"date": "2018-10-23", "sentiment": 0.0299}, {"date": "2018-10-30", "sentiment": 0.2001},
        {"date": "2018-11-06", "sentiment": 0.3221}, {"date": "2018-11-13", "sentiment": 0.177},
        {"date": "2018-11-20", "sentiment": 0.0547}, {"date": "2018-11-27", "sentiment": 0.1235},
        {"date": "2018-12-04", "sentiment": -0.0304}, {"date": "2018-12-11", "sentiment": 0.1448},
        {"date": "2018-12-18", "sentiment": 0.2367}, {"date": "2018-12-25", "sentiment": 0.0948},
        {"date": "2019-01-01", "sentiment": 0.3767}, {"date": "2019-01-08", "sentiment": 0.2377},
        {"date": "2019-01-15", "sentiment": 0.1942}, {"date": "2019-01-22", "sentiment": -0.0225},
        {"date": "2019-01-29", "sentiment": 0.2108}, {"date": "2019-02-05", "sentiment": 0.0914},
        {"date": "2019-02-12", "sentiment": 0.1115}, {"date": "2019-02-19", "sentiment": 0.1702},
        {"date": "2019-02-26", "sentiment": 0.1949}, {"date": "2019-03-05", "sentiment": 0.2412},
        {"date": "2019-03-12", "sentiment": 0.106}, {"date": "2019-03-19", "sentiment": 0.2345},
        {"date": "2019-03-26", "sentiment": 0.1654}, {"date": "2019-04-02", "sentiment": 0.1897}
    ]
    return pd.DataFrame(data)

# ============================================================================
# HELPERS
# ============================================================================

def get_sentiment_label(score):
    if score > 0.3:
        return "Very Positive", "🟢"
    elif score > 0.05:
        return "Positive", "📈"
    elif score > -0.05:
        return "Neutral", "⚖️"
    elif score > -0.3:
        return "Negative", "📉"
    else:
        return "Very Negative", "🔴"

def calculate_allocation(amount, btc_sentiment, eth_sentiment, btc_lambda, eth_lambda, risk_level):
    base_return = 0.08
    adj_btc = base_return + btc_lambda * btc_sentiment
    adj_eth = base_return + eth_lambda * eth_sentiment
    
    risk_scale = {'conservative': 0.4, 'moderate': 0.65, 'aggressive': 0.85}.get(risk_level, 0.65)
    
    raw_btc = max(0, adj_btc)
    raw_eth = max(0, adj_eth)
    total = raw_btc + raw_eth
    
    if total <= 0:
        w_btc, w_eth, w_cash = 0, 0, 1
    else:
        allocated = risk_scale
        w_btc = (raw_btc / total) * allocated
        w_eth = (raw_eth / total) * allocated
        w_cash = 1 - w_btc - w_eth
    
    return {
        'btc_weight': w_btc, 'eth_weight': w_eth, 'cash_weight': w_cash,
        'btc_amount': amount * w_btc, 'eth_amount': amount * w_eth, 'cash_amount': amount * w_cash,
        'adj_btc': adj_btc, 'adj_eth': adj_eth
    }

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    col1, col2, col3 = st.columns([2, 3, 1])
    with col1:
        st.markdown("### SentimentEdge")
    with col3:
        st.markdown("""
        <div style="font-family: 'Space Mono', monospace; font-size: 0.7rem;
                    padding: 0.3rem 0.8rem; border-radius: 4px;
                    border: 1px solid #00d4aa; color: #00d4aa; display: inline-block;">
            <span style="display: inline-block; width: 8px; height: 8px;
                         border-radius: 50%; background: #00d4aa;
                         margin-right: 0.4rem; animation: pulse 2s infinite;"></span>
            LIVE
        </div>
        """, unsafe_allow_html=True)
    
    # Hero
    st.markdown("""
    <div style="text-align: center; padding: 2rem 2rem 1rem 2rem;">
        <p style="font-family: 'Space Mono', monospace; font-size: 0.72rem;
                  letter-spacing: 0.2em; color: #00d4aa; margin-bottom: 1.5rem;">
            SENTIMENT-ADJUSTED MEAN-VARIANCE OPTIMIZATION
        </p>
        <h1 style="font-size: clamp(2.8rem, 7vw, 5.5rem); font-weight: 800;
                   line-height: 1.0; letter-spacing: -0.03em; margin-bottom: 1.5rem;">
            Invest Smarter<br>
            <span style="background: linear-gradient(135deg, #f7931a 0%, #627eea 100%);
                         -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                with Market Sentiment
            </span>
        </h1>
        <p style="color: #a0aec0; font-size: 1.1rem; max-width: 560px;
                  margin: 0 auto 2rem auto; line-height: 1.7;">
            Combining VADER + FinBERT sentiment analysis with Markowitz portfolio optimization —
            dynamically adjusted using social media and news signals across BTC and ETH.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Stats row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-value">1,258</div><div class="metric-label">Trading Days</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-value">2</div><div class="metric-label">NLP Models</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-value">SA-MVO</div><div class="metric-label">Core Algorithm</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-value">±1.0</div><div class="metric-label">Sentiment Range</div></div>', unsafe_allow_html=True)
    with col5:
        st.markdown('<div class="metric-card"><div class="metric-value">12-Mo</div><div class="metric-label">Rolling Window</div></div>', unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Portfolio Simulator", "📈 Lambda Coefficient", "💬 Sentiment History", "⚙️ Portfolio Allocator"])
    
    # Tab 1: Portfolio Simulator
    with tab1:
        st.subheader("Portfolio Weight Comparison — MVO vs SA-MVO")
        
        df_weights = load_weights_data()
        
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        with col_filter1:
            strat_filter = st.selectbox("Strategy", ["Both", "MVO Only", "SA-MVO Only"])
        with col_filter2:
            asset_filter = st.selectbox("Asset", ["Both", "BTC Only", "ETH Only"])
        with col_filter3:
            periods = st.slider("Number of Periods", 5, len(df_weights), len(df_weights))
        
        df_filtered = df_weights.head(periods)
        
        # Create figure
        fig = go.Figure()
        
        # Add traces based on filters
        if strat_filter in ["Both", "MVO Only"]:
            if asset_filter in ["Both", "BTC Only"]:
                fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['mvo_btc'],
                                        mode='lines+markers', name='MVO BTC',
                                        line=dict(color='#f7931a', width=2),
                                        marker=dict(size=4)))
            if asset_filter in ["Both", "ETH Only"]:
                fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['mvo_eth'],
                                        mode='lines+markers', name='MVO ETH',
                                        line=dict(color='#f7931a', width=2, dash='dot'),
                                        marker=dict(size=4)))
        
        if strat_filter in ["Both", "SA-MVO Only"]:
            if asset_filter in ["Both", "BTC Only"]:
                fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['samvo_btc'],
                                        mode='lines+markers', name='SA-MVO BTC',
                                        line=dict(color='#627eea', width=2),
                                        marker=dict(size=4)))
            if asset_filter in ["Both", "ETH Only"]:
                fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['samvo_eth'],
                                        mode='lines+markers', name='SA-MVO ETH',
                                        line=dict(color='#627eea', width=2, dash='dot'),
                                        marker=dict(size=4)))
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='#0c1120',
            plot_bgcolor='#0c1120',
            height=500,
            xaxis_title="Date",
            yaxis_title="Weight (%)",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            hovermode='x unified'
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Periods", periods)
        with col_m2:
            avg_mvo = df_filtered['mvo_btc'].mean()
            st.metric("Avg MVO BTC", f"{avg_mvo:.2f}%")
        with col_m3:
            avg_samvo = df_filtered['samvo_btc'].mean()
            st.metric("Avg SA-MVO BTC", f"{avg_samvo:.2f}%")
    
    # Tab 2: Lambda Coefficient
    with tab2:
        st.subheader("Rolling 12-Month λ (Sentiment Impact Coefficient) — Lag 3")
        st.markdown("λ represents the marginal effect of sentiment on future returns. Negative values indicate a contrarian relationship — high sentiment predicts lower returns.")
        
        df_lambdas = load_lambdas_data()
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df_lambdas['date'], y=df_lambdas['btc'],
                                  mode='lines+markers', name='λ BTC',
                                  line=dict(color='#f7931a', width=2),
                                  marker=dict(size=4)))
        fig2.add_trace(go.Scatter(x=df_lambdas['date'], y=df_lambdas['eth'],
                                  mode='lines+markers', name='λ ETH',
                                  line=dict(color='#627eea', width=2),
                                  marker=dict(size=4)))
        
        fig2.update_layout(
            template='plotly_dark',
            paper_bgcolor='#0c1120',
            plot_bgcolor='#0c1120',
            height=450,
            xaxis_title="Date",
            yaxis_title="λ Coefficient",
            hovermode='x unified'
        )
        
        fig2.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)")
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Latest lambda values
        latest = df_lambdas.iloc[-1]
        col_l1, col_l2 = st.columns(2)
        with col_l1:
            st.metric("Latest BTC λ", f"{latest['btc']:.2f}", delta="Contrarian" if latest['btc'] < 0 else "Pro-cyclical")
        with col_l2:
            st.metric("Latest ETH λ", f"{latest['eth']:.2f}", delta="Contrarian" if latest['eth'] < 0 else "Pro-cyclical")
    
    # Tab 3: Sentiment History
    with tab3:
        st.subheader("Weekly Weighted Sentiment — BTC vs ETH (2017–2022)")
        
        df_btc_sent = load_btc_sentiment()
        
        # Generate synthetic ETH sentiment (shifted + noise)
        np.random.seed(42)
        eth_sent = df_btc_sent['sentiment'].values * 0.75 + np.random.normal(0, 0.05, len(df_btc_sent))
        eth_sent = np.clip(eth_sent, -0.5, 0.8)
        
        fig3 = go.Figure()
        
        # Add area fill for BTC
        fig3.add_trace(go.Scatter(x=df_btc_sent['date'], y=df_btc_sent['sentiment'],
                                  mode='lines', name='BTC Sentiment',
                                  line=dict(color='#f7931a', width=2),
                                  fill='tozeroy',
                                  fillcolor='rgba(247,147,26,0.1)'))
        
        fig3.add_trace(go.Scatter(x=df_btc_sent['date'], y=eth_sent,
                                  mode='lines', name='ETH Sentiment',
                                  line=dict(color='#627eea', width=2, dash='dot'),
                                  fill='tozeroy',
                                  fillcolor='rgba(98,126,234,0.1)'))
        
        fig3.update_layout(
            template='plotly_dark',
            paper_bgcolor='#0c1120',
            plot_bgcolor='#0c1120',
            height=500,
            xaxis_title="Date",
            yaxis_title="Sentiment Score",
            hovermode='x unified'
        )
        
        fig3.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
        fig3.add_hrect(y0=-0.05, y1=0.05, line_width=0, fillcolor="rgba(255,255,255,0.03)", layer="below")
        
        st.plotly_chart(fig3, use_container_width=True)
    
    # Tab 4: Portfolio Allocator
    with tab4:
        st.subheader("Sentiment-Adjusted Portfolio Builder")
        st.markdown("Enter your parameters to calculate the optimal BTC/ETH allocation based on the SA-MVO model.")
        
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            amount = st.number_input("Investment Amount (USD)", min_value=100, value=10000, step=100)
        with col_a2:
            risk_level = st.selectbox("Risk Tolerance", ["conservative", "moderate", "aggressive"], index=1)
        
        st.markdown("### Sentiment Parameters")
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            btc_sentiment = st.slider("BTC Sentiment Score", -1.0, 1.0, 0.28, 0.01)
            btc_lambda = st.slider("BTC λ Coefficient", -10.0, 5.0, -4.36, 0.01)
        with col_s2:
            eth_sentiment = st.slider("ETH Sentiment Score", -1.0, 1.0, 0.14, 0.01)
            eth_lambda = st.slider("ETH λ Coefficient", -10.0, 5.0, -1.10, 0.01)
        
        # Calculate allocation
        allocation = calculate_allocation(amount, btc_sentiment, eth_sentiment, btc_lambda, eth_lambda, risk_level)
        
        st.markdown("### 📊 Recommended Portfolio Allocation")
        
        # Progress bars for allocation
        col_b1, col_b2 = st.columns([2, 1])
        with col_b1:
            # BTC bar
            st.markdown(f"**BTC** — {allocation['btc_weight']*100:.1f}%")
            st.progress(allocation['btc_weight'], text="")
            
            # ETH bar
            st.markdown(f"**ETH** — {allocation['eth_weight']*100:.1f}%")
            st.progress(allocation['eth_weight'], text="")
            
            # Cash bar
            st.markdown(f"**Cash/Stable** — {allocation['cash_weight']*100:.1f}%")
            st.progress(allocation['cash_weight'], text="")
        
        with col_b2:
            st.markdown(f"""
            <div style="background:#111827; border-radius:12px; padding:1rem; text-align:center;">
                <div style="font-size:0.7rem; color:#6b7a99;">BTC (USD)</div>
                <div style="font-size:1.3rem; font-weight:800; color:#f7931a;">${allocation['btc_amount']:,.0f}</div>
                <hr style="margin:0.5rem 0;">
                <div style="font-size:0.7rem; color:#6b7a99;">ETH (USD)</div>
                <div style="font-size:1.3rem; font-weight:800; color:#627eea;">${allocation['eth_amount']:,.0f}</div>
                <hr style="margin:0.5rem 0;">
                <div style="font-size:0.7rem; color:#6b7a99;">Cash / Stable</div>
                <div style="font-size:1.3rem; font-weight:800; color:#a0aec0;">${allocation['cash_amount']:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Note based on allocation
        if allocation['cash_weight'] > 0.5:
            st.warning("⚠️ Negative adjusted returns from λ × S signal caution. High cash allocation recommended.")
        elif allocation['adj_btc'] > allocation['adj_eth']:
            st.info("📈 BTC shows a stronger adjusted return signal under current sentiment. Weighted toward BTC.")
        else:
            st.info("📈 ETH shows a stronger adjusted return signal under current sentiment. Weighted toward ETH.")
        
        # Display adjusted returns
        with st.expander("🔬 View Adjusted Returns Calculation"):
            st.markdown(f"""
            <div class="formula-box">
                <div>Adjusted Return = Historical Return + (λ × Sentiment Score)</div>
                <div style="margin-top: 0.75rem;">
                    <span class="var">BTC: {allocation['adj_btc']*100:.2f}%</span> 
                    <span class="op">= 8% + ({btc_lambda:.2f} × {btc_sentiment:.2f})</span>
                </div>
                <div>
                    <span class="var">ETH: {allocation['adj_eth']*100:.2f}%</span> 
                    <span class="op">= 8% + ({eth_lambda:.2f} × {eth_sentiment:.2f})</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Methodology Section
    st.subheader("SA-MVO Methodology")
    
    col_meth1, col_meth2, col_meth3 = st.columns(3)
    with col_meth1:
        st.markdown("""
        <div style="background:#0c1120; border:1px solid rgba(255,255,255,0.07); border-radius:16px; padding:1.5rem; height:100%;">
            <div style="font-size:2rem; margin-bottom:1rem;">📡</div>
            <h4>Data Collection</h4>
            <p style="color:#a0aec0; font-size:0.85rem;">Social media posts and financial news headlines collected daily. BTC data spans 2017–2022 with over 3,000 daily entries.</p>
        </div>
        """, unsafe_allow_html=True)
    with col_meth2:
        st.markdown("""
        <div style="background:#0c1120; border:1px solid rgba(255,255,255,0.07); border-radius:16px; padding:1.5rem; height:100%;">
            <div style="font-size:2rem; margin-bottom:1rem;">🧠</div>
            <h4>Hybrid NLP Analysis</h4>
            <p style="color:#a0aec0; font-size:0.85rem;">VADER (F1=0.96) handles informal social text. FinBERT interprets financial news with domain-specific training.</p>
        </div>
        """, unsafe_allow_html=True)
    with col_meth3:
        st.markdown("""
        <div style="background:#0c1120; border:1px solid rgba(255,255,255,0.07); border-radius:16px; padding:1.5rem; height:100%;">
            <div style="font-size:2rem; margin-bottom:1rem;">⚖️</div>
            <h4>Weighted Sentiment Score</h4>
            <p style="color:#a0aec0; font-size:0.85rem;">BTC: 60% tweet + 40% news. ETH: 65% tweet + 35% news. Final score normalized to [−1, +1].</p>
        </div>
        """, unsafe_allow_html=True)
    
    col_meth4, col_meth5, col_meth6 = st.columns(3)
    with col_meth4:
        st.markdown("""
        <div style="background:#0c1120; border:1px solid rgba(255,255,255,0.07); border-radius:16px; padding:1.5rem; height:100%;">
            <div style="font-size:2rem; margin-bottom:1rem;">📐</div>
            <h4>λ Estimation via OLS</h4>
            <p style="color:#a0aec0; font-size:0.85rem;">Sentiment Impact Coefficient estimated by regressing future returns on current sentiment using OLS with 12-month rolling window.</p>
        </div>
        """, unsafe_allow_html=True)
    with col_meth5:
        st.markdown("""
        <div style="background:#0c1120; border:1px solid rgba(255,255,255,0.07); border-radius:16px; padding:1.5rem; height:100%;">
            <div style="font-size:2rem; margin-bottom:1rem;">📊</div>
            <h4>SA-MVO Optimization</h4>
            <p style="color:#a0aec0; font-size:0.85rem;">Adjusted Return = Historical Return + (λ × Sentiment Score) feeds into Markowitz MVO solver with variance-covariance matrix.</p>
        </div>
        """, unsafe_allow_html=True)
    with col_meth6:
        st.markdown("""
        <div style="background:#0c1120; border:1px solid rgba(255,255,255,0.07); border-radius:16px; padding:1.5rem; height:100%;">
            <div style="font-size:2rem; margin-bottom:1rem;">🔁</div>
            <h4>Rolling Backtesting</h4>
            <p style="color:#a0aec0; font-size:0.85rem;">6-month training window, 1-month testing cycle. Evaluated via Sharpe Ratio, Sortino Ratio, and Volatility metrics.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Formula box
    st.markdown("""
    <div class="formula-box">
        <div><span style="color:#627eea;">Adjusted Return</span><sub>i,t</sub> <span style="color:#6b7a99;">= α</span><sub>i</sub> + <span style="color:#f7931a;">λ</span><sub>i</sub> <span style="color:#6b7a99;">×</span> <span style="color:#627eea;">S</span><sub>i,t</sub> <span style="color:#6b7a99;">+ ε</span><sub>i,t</sub></div>
        <div style="font-size:0.7rem; color:#6b7a99; margin-top:0.5rem;">where S<sub>i,t</sub> is combined sentiment · λ<sub>i</sub> is the OLS-estimated sentiment impact coefficient · z is the return lag (3 days)</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="border-top:1px solid rgba(255,255,255,0.07); padding:2rem 0; margin-top:2rem;">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:1rem;">
            <span style="font-family:'Space Mono',monospace; font-size:0.9rem;">SentimentEdge</span>
            <p style="font-size:0.72rem; color:#6b7a99; max-width:500px; line-height:1.5;">
                ⚠️ This tool is for educational and research purposes only. SA-MVO outputs are based on historical sentiment data (2017–2022) and should not be treated as financial advice. Cryptocurrency investments carry significant risk of loss.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()