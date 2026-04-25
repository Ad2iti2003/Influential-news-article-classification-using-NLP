"""pages/analytics.py — Analytics & Visualization Dashboard"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.model_loader import load_dataset, load_models

PLOTLY_THEME = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font_color='#e8e8e8',
    font_family='IBM Plex Mono',
    colorway=['#e8c547','#ff6b35','#4ade80','#60a5fa','#c084fc','#f87171'],
)

def styled_fig(fig, height=380):
    fig.update_layout(height=height, margin=dict(t=30,b=30,l=10,r=10),
                      **PLOTLY_THEME)
    fig.update_xaxes(gridcolor='#1e1e1e', zerolinecolor='#2a2a2a')
    fig.update_yaxes(gridcolor='#1e1e1e', zerolinecolor='#2a2a2a')
    return fig


def show():
    st.markdown("""
    <h1 style='font-family:"Playfair Display",serif;font-size:2.2rem;
               font-weight:900;margin-bottom:4px;'>📊 Analytics Dashboard</h1>
    <p style='color:#888;font-family:"Source Serif 4",serif;font-size:1rem;
              margin-bottom:2rem;'>
        Explore patterns in influence, sentiment, and category distribution.
    </p>
    """, unsafe_allow_html=True)

    df = load_dataset()

    if df is None or len(df) == 0:
        st.warning("⚠️ No dataset found. Please ensure Phase 2 data is loaded.")
        return

    # ── Ensure required columns ─────────────────────────────────────────
    for col in ['sentiment_polarity','sentiment_subjectivity',
                'word_count','title_length']:
        if col not in df.columns:
            df[col] = 0

    # ── KPI Row ─────────────────────────────────────────────────────────
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Total Articles",  f"{len(df):,}")
    k2.metric("Categories",      df['category'].nunique())
    k3.metric("Avg Influence",   f"{df['influence_score'].mean():.3f}")
    k4.metric("High Influence",  f"{(df['influence_label']=='High').sum():,}")
    k5.metric("Avg Word Count",  f"{df['word_count'].mean():.0f}")

    st.markdown("---")

    # ── Row 1 ────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Articles by Category**")
        cat_counts = df['category'].value_counts().reset_index()
        cat_counts.columns = ['Category','Count']
        fig = px.bar(cat_counts, x='Category', y='Count',
                     color='Count', color_continuous_scale='YlOrRd')
        fig.update_traces(marker_line_width=0)
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(styled_fig(fig), use_container_width=True,
                        config={'displayModeBar': False})

    with col2:
        st.markdown("**Influence Label Distribution**")
        inf_counts = df['influence_label'].value_counts().reset_index()
        inf_counts.columns = ['Label','Count']
        color_map = {'High':'#4ade80','Medium':'#facc15','Low':'#f87171'}
        fig = px.pie(inf_counts, names='Label', values='Count',
                     color='Label', color_discrete_map=color_map,
                     hole=0.45)
        fig.update_traces(textfont_size=12,
                          marker=dict(line=dict(color='#0d0d0d', width=2)))
        st.plotly_chart(styled_fig(fig), use_container_width=True,
                        config={'displayModeBar': False})

    # ── Row 2 ────────────────────────────────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Influence Score Distribution**")
        fig = px.histogram(df, x='influence_score', nbins=40,
                           color_discrete_sequence=['#e8c547'])
        fig.update_traces(marker_line_width=0, opacity=0.85)
        fig.add_vline(x=df['influence_score'].mean(), line_dash='dash',
                      line_color='#ff6b35', annotation_text='Mean',
                      annotation_font_color='#ff6b35')
        st.plotly_chart(styled_fig(fig), use_container_width=True,
                        config={'displayModeBar': False})

    with col4:
        st.markdown("**Avg Influence Score by Category**")
        avg_inf = df.groupby('category')['influence_score'].mean().sort_values(ascending=True)
        fig = px.bar(x=avg_inf.values, y=avg_inf.index,
                     orientation='h', color=avg_inf.values,
                     color_continuous_scale='YlOrRd')
        fig.update_traces(marker_line_width=0)
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(styled_fig(fig), use_container_width=True,
                        config={'displayModeBar': False})

    st.markdown("---")

    # ── Row 3 ────────────────────────────────────────────────────────────
    col5, col6 = st.columns(2)

    with col5:
        st.markdown("**Sentiment Polarity vs Influence Score**")
        sample = df.sample(min(500, len(df)), random_state=42)
        color_map2 = {'High':'#4ade80','Medium':'#facc15','Low':'#f87171'}
        fig = px.scatter(sample, x='sentiment_polarity', y='influence_score',
                         color='influence_label', color_discrete_map=color_map2,
                         opacity=0.6, size_max=6)
        fig.update_traces(marker=dict(size=5))
        fig.add_vline(x=0, line_dash='dot', line_color='#333')
        st.plotly_chart(styled_fig(fig), use_container_width=True,
                        config={'displayModeBar': False})

    with col6:
        st.markdown("**Word Count Distribution by Influence**")
        fig = go.Figure()
        colors = {'High':'#4ade80','Medium':'#facc15','Low':'#f87171'}
        for label in ['High','Medium','Low']:
            sub = df[df['influence_label']==label]['word_count'].clip(0,2000)
            fig.add_trace(go.Box(
                y=sub, name=label,
                marker_color=colors[label],
                line_color=colors[label],
                fillcolor=colors[label].replace(')', ',0.15)').replace('rgb','rgba')
                           if 'rgb' in colors[label] else colors[label],
                boxmean=True,
            ))
        st.plotly_chart(styled_fig(fig), use_container_width=True,
                        config={'displayModeBar': False})

    st.markdown("---")

    # ── Row 4: Heatmap ───────────────────────────────────────────────────
    st.markdown("**Influence Level × Category Heatmap**")
    heat = df.groupby(['category','influence_label']).size().unstack(fill_value=0)
    for lbl in ['High','Medium','Low']:
        if lbl not in heat.columns:
            heat[lbl] = 0
    heat = heat[['High','Medium','Low']]

    fig = px.imshow(heat, color_continuous_scale='YlOrRd',
                    text_auto=True, aspect='auto')
    fig.update_traces(textfont_size=13)
    fig.update_layout(height=300, **PLOTLY_THEME,
                      margin=dict(t=20,b=20,l=10,r=10))
    st.plotly_chart(fig, use_container_width=True,
                    config={'displayModeBar': False})

    # ── Raw Data Table ───────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("📋 View Raw Data Sample"):
        cols_show = ['title','category','influence_label','influence_score',
                     'sentiment_polarity','word_count','source']
        cols_show = [c for c in cols_show if c in df.columns]
        st.dataframe(
            df[cols_show].head(50).style.background_gradient(
                subset=['influence_score'], cmap='YlOrRd'),
            use_container_width=True, height=350
        )
