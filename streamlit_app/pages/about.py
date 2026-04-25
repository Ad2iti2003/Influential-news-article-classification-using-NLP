"""pages/about.py — About / Project Info Page"""
import streamlit as st
import json, os

RESULTS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             'results', 'model_results.json')

def show():
    st.markdown("""
    <h1 style='font-family:"Playfair Display",serif;font-size:2.2rem;
               font-weight:900;margin-bottom:4px;'>ℹ️ About This Project</h1>
    <p style='color:#888;font-family:"Source Serif 4",serif;font-size:1rem;
              margin-bottom:2rem;'>
        Final Year Project — NLP & Deep Learning
    </p>
    """, unsafe_allow_html=True)

    # ── Project Overview ────────────────────────────────────────────────
    st.markdown("""
    <div style='background:#161616;border:1px solid #2a2a2a;
                border-top:3px solid #e8c547;border-radius:6px;
                padding:2rem;margin-bottom:2rem;'>
        <h2 style='font-family:"Playfair Display",serif;font-size:1.4rem;
                   font-weight:700;color:#e8e8e8;margin-top:0;'>
            Influential News Classification & Ranking System
        </h2>
        <p style='font-family:"Source Serif 4",serif;font-size:0.95rem;
                  color:#aaa;line-height:1.8;'>
            This system automatically classifies news articles by category and
            predicts their <strong style='color:#e8c547;'>influence level</strong>
            using state-of-the-art NLP techniques. It scrapes real news from
            Google News, enriches the data with GDELT engagement signals, and
            uses a fine-tuned BERT model for classification.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Pipeline Phases ─────────────────────────────────────────────────
    st.markdown("""
    <h2 style='font-family:"Playfair Display",serif;font-size:1.4rem;
               font-weight:700;margin-bottom:1rem;'>Project Phases</h2>
    """, unsafe_allow_html=True)

    phases = [
        ("Phase 1", "Data Collection",
         "Google News scraping (last 6 months) + GDELT engagement signals for 7 categories.",
         "#e8c547"),
        ("Phase 2", "Preprocessing & Feature Engineering",
         "Text cleaning, lemmatization (spaCy), TF-IDF extraction, sentiment scoring (TextBlob), label encoding.",
         "#ff6b35"),
        ("Phase 3", "Model Training",
         "Baseline: Logistic Regression, Random Forest, XGBoost. Advanced: BERT fine-tuning. Influence: XGBoost scorer.",
         "#4ade80"),
        ("Phase 4", "Streamlit Application",
         "Full web app with article classifier, ranked article browser, analytics dashboard.",
         "#60a5fa"),
    ]
    for phase, title, desc, color in phases:
        st.markdown(f"""
        <div style='display:flex;gap:1.5rem;padding:1rem 0;
                    border-bottom:1px solid #1e1e1e;'>
            <div style='min-width:90px;font-family:"IBM Plex Mono",monospace;
                        font-size:0.75rem;color:{color};font-weight:600;
                        padding-top:2px;'>{phase}</div>
            <div>
                <div style='font-family:"Playfair Display",serif;font-weight:700;
                            color:#e8e8e8;font-size:1rem;margin-bottom:4px;'>{title}</div>
                <div style='font-family:"Source Serif 4",serif;font-size:0.88rem;
                            color:#888;line-height:1.6;'>{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tech Stack ──────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <h2 style='font-family:"Playfair Display",serif;font-size:1.3rem;
                   font-weight:700;margin-bottom:1rem;'>🛠️ Tech Stack</h2>
        """, unsafe_allow_html=True)
        tech = [
            ("Data",      "GNews, GDELT API, BeautifulSoup"),
            ("NLP",       "spaCy, NLTK, TextBlob"),
            ("ML",        "scikit-learn, XGBoost"),
            ("DL",        "PyTorch, HuggingFace Transformers"),
            ("Model",     "BERT (bert-base-uncased)"),
            ("Frontend",  "Streamlit, Plotly"),
        ]
        for category, tools in tech:
            st.markdown(f"""
            <div style='display:flex;gap:12px;padding:8px 0;
                        border-bottom:1px solid #1a1a1a;'>
                <div style='font-family:"IBM Plex Mono",monospace;font-size:0.75rem;
                            color:#e8c547;min-width:80px;'>{category}</div>
                <div style='font-family:"Source Serif 4",serif;font-size:0.88rem;
                            color:#aaa;'>{tools}</div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <h2 style='font-family:"Playfair Display",serif;font-size:1.3rem;
                   font-weight:700;margin-bottom:1rem;'>📊 Model Results</h2>
        """, unsafe_allow_html=True)

        # Try to load saved results
        results = None
        if os.path.exists(RESULTS_PATH):
            with open(RESULTS_PATH) as f:
                results = json.load(f)

        if results:
            for model_name, metrics in results.get('models', {}).items():
                is_best = model_name == results.get('best_model', '')
                border  = '#e8c547' if is_best else '#2a2a2a'
                label   = ' 🏆 BEST' if is_best else ''
                st.markdown(f"""
                <div style='background:#1e1e1e;border:1px solid {border};
                            border-radius:5px;padding:10px 14px;margin-bottom:8px;'>
                    <div style='font-family:"IBM Plex Mono",monospace;font-size:0.78rem;
                                color:#e8c547;margin-bottom:4px;'>{model_name}{label}</div>
                    <div style='display:flex;gap:20px;font-family:"IBM Plex Mono",monospace;
                                font-size:0.82rem;color:#aaa;'>
                        <span>ACC: <strong style='color:#e8e8e8;'>
                            {metrics.get("accuracy",0):.4f}</strong></span>
                        <span>F1: <strong style='color:#e8e8e8;'>
                            {metrics.get("f1_score",0):.4f}</strong></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Model results will appear here after Phase 3 training.")

    st.markdown("---")

    # ── Influence Formula ───────────────────────────────────────────────
    st.markdown("""
    <h2 style='font-family:"Playfair Display",serif;font-size:1.3rem;
               font-weight:700;margin-bottom:1rem;'>📐 Influence Score Formula</h2>
    <div style='background:#161616;border:1px solid #2a2a2a;border-radius:6px;
                padding:1.5rem;font-family:"IBM Plex Mono",monospace;'>
        <div style='font-size:1.1rem;color:#e8c547;margin-bottom:1rem;text-align:center;'>
            Score = 0.4×S + 0.3×C + 0.2×L + 0.1×R
        </div>
        <div style='font-size:0.82rem;color:#888;line-height:2;'>
            <span style='color:#e8e8e8;'>S</span> = Shares / Engagement signal (GDELT hits, normalized 0–1)<br>
            <span style='color:#e8e8e8;'>C</span> = Content tone strength (|sentiment| normalized 0–1)<br>
            <span style='color:#e8e8e8;'>L</span> = Length score (word count normalized 0–1)<br>
            <span style='color:#e8e8e8;'>R</span> = Recency score (1.0 = today, 0.0 = 6 months ago)<br>
            <br>
            <span style='color:#ff6b35;'>High</span> = Score &gt; 0.66 &nbsp;|&nbsp;
            <span style='color:#facc15;'>Medium</span> = 0.33–0.66 &nbsp;|&nbsp;
            <span style='color:#f87171;'>Low</span> = &lt; 0.33
        </div>
    </div>
    """, unsafe_allow_html=True)
