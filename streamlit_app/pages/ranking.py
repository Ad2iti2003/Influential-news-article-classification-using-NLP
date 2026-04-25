"""pages/ranking.py — Top Influential Articles Ranking Page"""
import streamlit as st
import pandas as pd
from utils.model_loader import load_dataset
from utils.predictor import INFLUENCE_EMOJI, CATEGORY_EMOJI

MEDALS = {1: "🥇", 2: "🥈", 3: "🥉"}

def show():
    st.markdown("""
    <h1 style='font-family:"Playfair Display",serif;font-size:2.2rem;
               font-weight:900;margin-bottom:4px;'>🏆 Top Influential Articles</h1>
    <p style='color:#888;font-family:"Source Serif 4",serif;font-size:1rem;
              margin-bottom:2rem;'>
        Most influential news articles ranked by influence score.
    </p>
    """, unsafe_allow_html=True)

    df = load_dataset()

    if df is None or 'influence_score' not in df.columns:
        st.warning("⚠️ Dataset not found. Please ensure Phase 2 data is loaded.")
        st.info("Upload your `articles_cleaned.csv` to the `data/processed/` folder.")
        return

    # ── Filters ────────────────────────────────────────────────────────
    col_f1, col_f2, col_f3 = st.columns([2, 2, 1])

    categories = ['All Categories'] + sorted(df['category'].unique().tolist())
    sel_cat = col_f1.selectbox("Category", categories)

    influence_opts = ['All', 'High', 'Medium', 'Low']
    sel_inf = col_f2.selectbox("Influence Level", influence_opts)

    top_n = col_f3.selectbox("Show Top", [10, 20, 50], index=0)

    # ── Filter ─────────────────────────────────────────────────────────
    filtered = df.copy()
    if sel_cat != 'All Categories':
        filtered = filtered[filtered['category'] == sel_cat]
    if sel_inf != 'All':
        filtered = filtered[filtered['influence_label'] == sel_inf]

    filtered = filtered.sort_values('influence_score', ascending=False).head(top_n)

    # ── Summary bar ────────────────────────────────────────────────────
    st.markdown(f"""
    <div style='font-family:"IBM Plex Mono",monospace;font-size:0.75rem;
                color:#888;margin-bottom:1.5rem;'>
        SHOWING {len(filtered)} ARTICLES
        {f"IN {sel_cat.upper()}" if sel_cat != "All Categories" else ""}
        {f"· {sel_inf.upper()} INFLUENCE" if sel_inf != "All" else ""}
    </div>
    """, unsafe_allow_html=True)

    if filtered.empty:
        st.info("No articles match the selected filters.")
        return

    # ── Article Cards ──────────────────────────────────────────────────
    for rank, (_, row) in enumerate(filtered.iterrows(), 1):
        cat      = str(row.get('category', 'unknown'))
        cat_icon = CATEGORY_EMOJI.get(cat, '📰')
        inf_lbl  = str(row.get('influence_label', 'Low'))
        inf_icon = INFLUENCE_EMOJI.get(inf_lbl, '')
        score    = float(row.get('influence_score', 0))
        medal    = MEDALS.get(rank, f"#{rank}")
        pct      = int(score * 100)

        inf_color = ("#4ade80" if inf_lbl=="High"
                     else "#facc15" if inf_lbl=="Medium" else "#f87171")

        title   = str(row.get('title', 'Untitled'))[:120]
        source  = str(row.get('source', 'Unknown'))
        pub     = str(row.get('published', ''))[:10]
        excerpt = str(row.get('content', ''))[:180].strip()
        if len(str(row.get('content', ''))) > 180:
            excerpt += '...'

        st.markdown(f"""
        <div class='news-card'>
            <div style='display:flex;justify-content:space-between;
                        align-items:flex-start;gap:1rem;'>
                <div style='flex:1;'>
                    <div style='display:flex;align-items:center;gap:10px;
                                margin-bottom:8px;'>
                        <span style='font-family:"IBM Plex Mono",monospace;
                                     font-size:1.1rem;'>{medal}</span>
                        <span style='font-family:"IBM Plex Mono",monospace;
                                     font-size:0.7rem;color:#888;
                                     background:#1e1e1e;padding:2px 8px;
                                     border-radius:3px;'>
                            {cat_icon} {cat.upper()}
                        </span>
                        <span class='badge-{inf_lbl.lower()}'>
                            {inf_icon} {inf_lbl}
                        </span>
                    </div>
                    <div style='font-family:"Playfair Display",serif;
                                font-size:1.05rem;font-weight:700;
                                color:#e8e8e8;margin-bottom:6px;
                                line-height:1.3;'>
                        {title}
                    </div>
                    <div style='font-family:"Source Serif 4",serif;
                                font-size:0.88rem;color:#666;
                                line-height:1.5;margin-bottom:10px;'>
                        {excerpt}
                    </div>
                    <div style='font-family:"IBM Plex Mono",monospace;
                                font-size:0.72rem;color:#555;'>
                        {source} · {pub}
                    </div>
                </div>
                <div style='text-align:center;min-width:80px;'>
                    <div style='font-family:"IBM Plex Mono",monospace;
                                font-size:1.6rem;font-weight:600;
                                color:{inf_color};'>{score:.3f}</div>
                    <div style='font-family:"IBM Plex Mono",monospace;
                                font-size:0.65rem;color:#555;
                                margin-bottom:8px;'>SCORE</div>
                    <div class='score-bar-wrap' style='width:70px;margin:0 auto;'>
                        <div class='score-bar' style='width:{pct}%;'></div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Download Button ─────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    csv = filtered[['title','category','influence_label','influence_score',
                    'source','published']].to_csv(index=False)
    st.download_button(
        label="⬇️  Download Rankings as CSV",
        data=csv,
        file_name=f"top_articles_{sel_cat.replace(' ','_')}.csv",
        mime="text/csv",
    )
