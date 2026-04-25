"""pages/classify.py — Article Classification Page"""
import streamlit as st
import time
import plotly.graph_objects as go
from utils.predictor import predict, INFLUENCE_EMOJI, CATEGORY_EMOJI
from utils.model_loader import load_models

# ── Sample Articles ──────────────────────────────────────────────────────
SAMPLES = {
    "🏀 Sports"      : ("NBA Finals: Record-Breaking Performance Stuns Crowd",
                        "In a historic NBA Finals game, star player delivered a record-breaking 67-point performance, leading his team to a dramatic comeback victory. The game drew over 40 million viewers, making it the most-watched NBA game in a decade. Social media erupted with over 5 million posts within hours of the final buzzer."),
    "💻 Technology"  : ("Apple Unveils Next-Generation AI Chip That Outperforms All Competitors",
                        "Apple has announced a breakthrough AI chip that delivers 3x the performance of current competitors at half the power consumption. The chip will power the next iPhone and MacBook lineup. Industry analysts predict this could shift the smartphone market significantly, with Apple stocks rising 8% following the announcement."),
    "💼 Business"    : ("Global Markets Surge as Fed Signals Rate Cut Ahead",
                        "Global financial markets surged on Wednesday after the Federal Reserve signaled it may cut interest rates in the upcoming quarter. The S&P 500 gained 2.3%, its biggest single-day gain this year. Investors reacted positively to comments from Fed Chairman indicating that inflation has cooled sufficiently to justify easing monetary policy."),
    "🏥 Health"      : ("Scientists Discover Breakthrough Treatment for Alzheimer's Disease",
                        "Researchers at MIT have discovered a new treatment approach that successfully reversed Alzheimer's symptoms in clinical trials. The treatment, using targeted gene therapy, showed 78% effectiveness in early-stage patients. The discovery is being hailed as the most significant advance in Alzheimer's research in 30 years."),
}

def gauge_chart(score: float, label: str):
    color = "#4ade80" if label=="High" else ("#facc15" if label=="Medium" else "#f87171")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        number={'suffix': '%', 'font': {'size': 28, 'color': color,
                                         'family': 'IBM Plex Mono'}},
        gauge={
            'axis'     : {'range': [0,100], 'tickcolor': '#444',
                          'tickfont': {'color': '#666', 'size': 10}},
            'bar'      : {'color': color, 'thickness': 0.25},
            'bgcolor'  : '#1e1e1e',
            'bordercolor': '#2a2a2a',
            'steps'    : [
                {'range': [0,  33], 'color': '#1a0a0a'},
                {'range': [33, 66], 'color': '#1a1500'},
                {'range': [66,100], 'color': '#0a1a0a'},
            ],
            'threshold': {'line':{'color':color,'width':3},
                          'thickness':0.8,'value': score*100},
        },
        title={'text': f"Influence Score", 'font': {'color':'#888', 'size':12,
                                                      'family':'IBM Plex Mono'}},
    ))
    fig.update_layout(
        height=220, margin=dict(t=30,b=0,l=20,r=20),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='#e8e8e8',
    )
    return fig


def prob_bar_chart(all_probs: dict):
    if not all_probs:
        return None
    cats   = list(all_probs.keys())
    probs  = [all_probs[c]*100 for c in cats]
    colors = ['#e8c547' if p==max(probs) else '#2a2a2a' for p in probs]
    fig = go.Figure(go.Bar(
        x=probs, y=cats, orientation='h',
        marker_color=colors, marker_line_width=0,
        text=[f'{p:.1f}%' for p in probs],
        textposition='outside',
        textfont=dict(color='#888', size=11, family='IBM Plex Mono'),
    ))
    fig.update_layout(
        height=max(180, len(cats)*40),
        margin=dict(t=10,b=10,l=10,r=60),
        xaxis=dict(range=[0,110], showgrid=False, showticklabels=False,
                   zeroline=False),
        yaxis=dict(tickfont=dict(color='#e8e8e8', size=11,
                                  family='IBM Plex Mono')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig


# ── MAIN ─────────────────────────────────────────────────────────────────
def show():
    st.markdown("""
    <h1 style='font-family:"Playfair Display",serif;font-size:2.2rem;
               font-weight:900;margin-bottom:4px;'>🔍 Classify Article</h1>
    <p style='color:#888;font-family:"Source Serif 4",serif;font-size:1rem;
              margin-bottom:2rem;'>
        Paste any news article to predict its category and influence level.
    </p>
    """, unsafe_allow_html=True)

    models = load_models()
    model_tag = "🤖 BERT" if models.get('bert_ok') else "⚡ XGBoost"
    st.markdown(f"""
    <div style='font-family:"IBM Plex Mono",monospace;font-size:0.75rem;
                color:#e8c547;margin-bottom:1.5rem;'>
        ACTIVE MODEL: {model_tag}
    </div>
    """, unsafe_allow_html=True)

    # ── Load Sample ────────────────────────────────────────────────────
    st.markdown("**Try a sample article:**")
    sample_cols = st.columns(len(SAMPLES))
    for col, (label, (stitle, scontent)) in zip(sample_cols, SAMPLES.items()):
        if col.button(label, use_container_width=True):
            st.session_state['sample_title']   = stitle
            st.session_state['sample_content'] = scontent

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Input Form ─────────────────────────────────────────────────────
    title = st.text_input(
        "Article Title",
        value=st.session_state.get('sample_title', ''),
        placeholder="Enter the news article headline...",
    )
    content = st.text_area(
        "Article Content",
        value=st.session_state.get('sample_content', ''),
        placeholder="Paste the full article text here...",
        height=200,
    )

    col_btn, col_clear, _ = st.columns([2, 1, 5])
    classify_btn = col_btn.button("⚡ Classify Now", use_container_width=True)
    if col_clear.button("✖ Clear", use_container_width=True):
        st.session_state.pop('sample_title',   None)
        st.session_state.pop('sample_content', None)
        st.rerun()

    # ── Prediction ─────────────────────────────────────────────────────
    if classify_btn:
        if not title.strip() and not content.strip():
            st.warning("Please enter a title or content to classify.")
            return

        with st.spinner("Analyzing article..."):
            time.sleep(0.3)   # UX breathing room
            result = predict(title, content)

        st.markdown("---")
        st.markdown("""
        <h2 style='font-family:"Playfair Display",serif;font-size:1.5rem;
                   font-weight:700;margin-bottom:1.5rem;'>📊 Results</h2>
        """, unsafe_allow_html=True)

        # ── Top Result Cards ───────────────────────────────────────────
        cat      = result['category']
        cat_icon = CATEGORY_EMOJI.get(cat, '📰')
        inf_lbl  = result['influence_label']
        inf_icon = INFLUENCE_EMOJI[inf_lbl]
        badge_cls= f"badge-{inf_lbl.lower()}"

        r1, r2, r3 = st.columns(3)
        r1.metric("Category",
                  f"{cat_icon} {cat.title()}",
                  f"Confidence: {result['category_conf']:.1%}")
        r2.metric("Influence Label",
                  f"{inf_icon} {inf_lbl}",
                  f"Score: {result['influence_score']:.4f}")
        r3.metric("Word Count",
                  f"{result['word_count']:,}",
                  f"Sentiment: {result['sentiment_pol']:+.3f}")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Charts ─────────────────────────────────────────────────────
        left, right = st.columns([1, 1])

        with left:
            st.markdown("""<div style='font-family:"IBM Plex Mono",monospace;
                font-size:0.75rem;color:#888;letter-spacing:0.1em;
                margin-bottom:8px;'>INFLUENCE GAUGE</div>""",
                unsafe_allow_html=True)
            st.plotly_chart(
                gauge_chart(result['influence_score'], inf_lbl),
                use_container_width=True, config={'displayModeBar': False}
            )

            # Influence breakdown
            st.markdown("""<div style='font-family:"IBM Plex Mono",monospace;
                font-size:0.75rem;color:#888;letter-spacing:0.1em;
                margin:12px 0 8px 0;'>ARTICLE SIGNALS</div>""",
                unsafe_allow_html=True)
            signals = {
                'Sentiment Strength'  : abs(result['sentiment_pol']),
                'Subjectivity'        : result['sentiment_subj'],
                'Length Score'        : min(result['word_count']/1000, 1.0),
                'Influence Score'     : result['influence_score'],
            }
            for sig_name, sig_val in signals.items():
                pct = int(sig_val * 100)
                st.markdown(f"""
                <div style='margin-bottom:10px;'>
                    <div style='display:flex;justify-content:space-between;
                                font-family:"IBM Plex Mono",monospace;
                                font-size:0.75rem;color:#888;margin-bottom:3px;'>
                        <span>{sig_name}</span><span>{pct}%</span>
                    </div>
                    <div class='score-bar-wrap'>
                        <div class='score-bar' style='width:{pct}%;'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with right:
            st.markdown("""<div style='font-family:"IBM Plex Mono",monospace;
                font-size:0.75rem;color:#888;letter-spacing:0.1em;
                margin-bottom:8px;'>CATEGORY PROBABILITIES</div>""",
                unsafe_allow_html=True)
            if result['all_probs']:
                prob_fig = prob_bar_chart(result['all_probs'])
                if prob_fig:
                    st.plotly_chart(prob_fig, use_container_width=True,
                                    config={'displayModeBar': False})

            # Summary box
            st.markdown(f"""
            <div style='background:#161616;border:1px solid #2a2a2a;
                        border-left:3px solid #e8c547;border-radius:6px;
                        padding:1.2rem 1.5rem;margin-top:1rem;'>
                <div style='font-family:"IBM Plex Mono",monospace;
                            font-size:0.7rem;color:#888;letter-spacing:0.15em;
                            margin-bottom:10px;'>ANALYSIS SUMMARY</div>
                <div style='font-family:"Source Serif 4",serif;
                            font-size:0.95rem;color:#e8e8e8;line-height:1.7;'>
                    This article is classified as
                    <strong style='color:#e8c547;'>{cat.upper()}</strong>
                    with <strong>{result['category_conf']:.1%}</strong> confidence.<br>
                    It has <strong style='color:{"#4ade80" if inf_lbl=="High" else "#facc15" if inf_lbl=="Medium" else "#f87171"};'>
                    {inf_lbl.upper()} influence</strong>
                    (score: <strong>{result['influence_score']:.4f}</strong>).<br>
                    Sentiment is
                    <strong>{"Positive" if result["sentiment_pol"]>0.05
                              else "Negative" if result["sentiment_pol"]<-0.05
                              else "Neutral"}</strong>
                    ({result["sentiment_pol"]:+.3f}).
                </div>
            </div>
            """, unsafe_allow_html=True)
