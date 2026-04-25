"""
================================================
Influential News Classification & Ranking System
Phase 4 — Streamlit App  |  Main Entry Point
================================================
"""

import streamlit as st

# ── PAGE CONFIG (must be first Streamlit call) ──────────────────────────
st.set_page_config(
    page_title="NewsIQ — Influential News Classifier",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── GLOBAL CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=IBM+Plex+Mono:wght@400;600&family=Source+Serif+4:wght@300;400;600&display=swap');

/* ── Root Variables ── */
:root {
    --bg:          #0d0d0d;
    --surface:     #161616;
    --surface2:    #1e1e1e;
    --border:      #2a2a2a;
    --accent:      #e8c547;
    --accent2:     #ff6b35;
    --text:        #e8e8e8;
    --text-muted:  #888;
    --high:        #4ade80;
    --medium:      #facc15;
    --low:         #f87171;
    --font-display:'Playfair Display', serif;
    --font-body:   'Source Serif 4', serif;
    --font-mono:   'IBM Plex Mono', monospace;
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Hide Streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Metric Cards ── */
[data-testid="stMetric"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    padding: 1rem !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--font-mono) !important;
    color: var(--accent) !important;
    font-size: 1.8rem !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--accent) !important;
    color: #0d0d0d !important;
    font-family: var(--font-mono) !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 0.6rem 2rem !important;
    letter-spacing: 0.05em !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #f0d060 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(232,197,71,0.3) !important;
}

/* ── Text inputs & text areas ── */
.stTextInput input, .stTextArea textarea, .stSelectbox select {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    font-family: var(--font-body) !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(232,197,71,0.2) !important;
}

/* ── Progress bars ── */
.stProgress > div > div {
    background: var(--accent) !important;
    border-radius: 2px !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.1em !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.8rem 1.5rem !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* ── Dividers ── */
hr { border-color: var(--border) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* ── Custom Card ── */
.news-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 6px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}
.news-card:hover { border-left-color: var(--accent2); }

.badge-high   { background:#052e16; color:var(--high);   padding:2px 10px; border-radius:20px; font-family:var(--font-mono); font-size:0.75rem; }
.badge-medium { background:#422006; color:var(--medium); padding:2px 10px; border-radius:20px; font-family:var(--font-mono); font-size:0.75rem; }
.badge-low    { background:#450a0a; color:var(--low);    padding:2px 10px; border-radius:20px; font-family:var(--font-mono); font-size:0.75rem; }

.score-bar-wrap { background:var(--surface2); border-radius:3px; height:8px; margin-top:6px; }
.score-bar      { height:8px; border-radius:3px; background:linear-gradient(90deg,var(--accent),var(--accent2)); }
</style>
""", unsafe_allow_html=True)


# ── SIDEBAR ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 2rem 0;'>
        <div style='font-family:var(--font-mono);color:var(--accent);
                    font-size:0.7rem;letter-spacing:0.2em;margin-bottom:4px;'>
            FINAL YEAR PROJECT
        </div>
        <div style='font-family:"Playfair Display",serif;font-size:1.5rem;
                    font-weight:900;line-height:1.1;'>
            NewsIQ
        </div>
        <div style='font-family:var(--font-mono);color:var(--text-muted);
                    font-size:0.7rem;margin-top:4px;'>
            Influential News Classifier
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠  Home", "🔍  Classify Article", "🏆  Top Articles", "📊  Analytics", "ℹ️  About"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-family:var(--font-mono);font-size:0.7rem;color:var(--text-muted);'>
        <div style='margin-bottom:6px;'>POWERED BY</div>
        <div style='color:var(--text);'>🤖 BERT (bert-base-uncased)</div>
        <div style='color:var(--text);'>⚡ XGBoost Influence Scorer</div>
        <div style='color:var(--text);'>📰 Google News + GDELT</div>
    </div>
    """, unsafe_allow_html=True)

# ── PAGE ROUTING ─────────────────────────────────────────────────────────
if   "Home"     in page: from pages import home;      home.show()
elif "Classify" in page: from pages import classify;  classify.show()
elif "Top"      in page: from pages import ranking;   ranking.show()
elif "Analytics"in page: from pages import analytics; analytics.show()
elif "About"    in page: from pages import about;     about.show()
