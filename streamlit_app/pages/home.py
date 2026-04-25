"""pages/home.py — Landing / Home page"""
import streamlit as st
from utils.model_loader import load_models, load_dataset

def show():
    # ── Hero ────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='padding: 3rem 0 2rem 0;'>
        <div style='font-family:"IBM Plex Mono",monospace;color:#e8c547;
                    font-size:0.75rem;letter-spacing:0.25em;margin-bottom:12px;'>
            NLP · DEEP LEARNING · REAL-TIME CLASSIFICATION
        </div>
        <h1 style='font-family:"Playfair Display",serif;font-size:3.5rem;
                   font-weight:900;line-height:1.05;margin:0;color:#e8e8e8;'>
            Influential News<br>
            <span style='color:#e8c547;'>Classification</span> &amp;
            <span style='color:#ff6b35;'>Ranking</span>
        </h1>
        <p style='font-family:"Source Serif 4",serif;font-size:1.15rem;
                  color:#888;margin-top:1.2rem;max-width:600px;line-height:1.7;'>
            Paste any news article and instantly discover its category,
            influence level, and how it ranks among the most impactful
            news of the past six months.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Stats Row ────────────────────────────────────────────────────────
    df = load_dataset()
    models = load_models()

    total_arts  = len(df) if df is not None else "—"
    num_cats    = df['category'].nunique() if df is not None else "—"
    high_inf    = len(df[df['influence_label']=='High']) if df is not None else "—"
    model_used  = "BERT" if models.get('bert_ok') else "XGBoost"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📰 Articles",        total_arts)
    c2.metric("🗂️ Categories",      num_cats)
    c3.metric("🔥 High Influence",   high_inf)
    c4.metric("🤖 Model",            model_used)

    st.markdown("---")

    # ── Feature Cards ────────────────────────────────────────────────────
    st.markdown("""
    <h2 style='font-family:"Playfair Display",serif;font-size:1.6rem;
               font-weight:700;margin-bottom:1.5rem;'>What Can You Do?</h2>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    cards = [
        ("🔍", "Classify Article",
         "Paste any news article to predict its category and influence score using BERT.",
         "#e8c547"),
        ("🏆", "Top Articles",
         "Browse the most influential articles ranked by category from our dataset.",
         "#ff6b35"),
        ("📊", "Analytics",
         "Explore charts showing sentiment, category distributions, and influence trends.",
         "#4ade80"),
    ]
    for col, (icon, title, desc, color) in zip([col1, col2, col3], cards):
        col.markdown(f"""
        <div style='background:#161616;border:1px solid #2a2a2a;
                    border-top:3px solid {color};border-radius:6px;
                    padding:1.5rem;height:180px;'>
            <div style='font-size:2rem;margin-bottom:8px;'>{icon}</div>
            <div style='font-family:"Playfair Display",serif;font-size:1.1rem;
                        font-weight:700;color:#e8e8e8;margin-bottom:8px;'>{title}</div>
            <div style='font-family:"Source Serif 4",serif;font-size:0.9rem;
                        color:#888;line-height:1.5;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # ── Pipeline Overview ────────────────────────────────────────────────
    st.markdown("""
    <h2 style='font-family:"Playfair Display",serif;font-size:1.6rem;
               font-weight:700;margin-bottom:1.5rem;'>How It Works</h2>
    """, unsafe_allow_html=True)

    steps = [
        ("01", "Data Collection",    "Google News scraping + GDELT influence signals (6 months)"),
        ("02", "Preprocessing",      "Text cleaning, lemmatization, TF-IDF, sentiment scoring"),
        ("03", "BERT Classification","Fine-tuned bert-base-uncased for category prediction"),
        ("04", "Influence Scoring",  "XGBoost model predicts High / Medium / Low influence"),
        ("05", "Ranking",            "Articles ranked by weighted influence score within category"),
    ]
    for num, title, desc in steps:
        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:1.5rem;
                    padding:1rem 0;border-bottom:1px solid #1e1e1e;'>
            <div style='font-family:"IBM Plex Mono",monospace;font-size:1.5rem;
                        font-weight:600;color:#e8c547;min-width:40px;'>{num}</div>
            <div>
                <div style='font-family:"Playfair Display",serif;font-weight:700;
                            font-size:1rem;color:#e8e8e8;'>{title}</div>
                <div style='font-family:"Source Serif 4",serif;font-size:0.88rem;
                            color:#888;margin-top:2px;'>{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Quick Start CTA ──────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈 Use the sidebar to navigate. Start with **🔍 Classify Article** to test the model!")