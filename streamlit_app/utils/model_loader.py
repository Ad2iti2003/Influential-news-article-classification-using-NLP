"""
utils/model_loader.py
Loads and caches all Phase 3 models.
Called by every page in the app.
"""

import streamlit as st
import pickle, os, json
import numpy as np

# ── Try importing heavy libraries gracefully ─────────────────────────────
try:
    import torch
    from transformers import BertTokenizer, BertForSequenceClassification
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR   = os.path.join(BASE_DIR, "data", "processed")

BERT_PATH     = os.path.join(MODELS_DIR, "bert_news_classifier")
TFIDF_PATH    = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
ENCODER_PATH  = os.path.join(MODELS_DIR, "category_encoder.pkl")
INF_PATH      = os.path.join(MODELS_DIR, "influence_scorer.pkl")
XGB_PATH      = os.path.join(MODELS_DIR, "xgboost.pkl")
DATA_PATH     = os.path.join(DATA_DIR,   "articles_cleaned.csv")


@st.cache_resource(show_spinner=False)
def load_models():
    """Load all models once and cache them."""
    models = {}

    # ── Category Encoder ────────────────────────────────────────────────
    if os.path.exists(ENCODER_PATH):
        with open(ENCODER_PATH, 'rb') as f:
            models['encoder'] = pickle.load(f)
        models['categories'] = list(models['encoder'].classes_)
    else:
        models['categories'] = ['business','entertainment','health',
                                 'science','sports','technology']

    # ── TF-IDF Vectorizer ───────────────────────────────────────────────
    if os.path.exists(TFIDF_PATH):
        with open(TFIDF_PATH, 'rb') as f:
            models['tfidf'] = pickle.load(f)

    # ── BERT ─────────────────────────────────────────────────────────────
    if BERT_AVAILABLE and os.path.exists(BERT_PATH):
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            models['tokenizer'] = BertTokenizer.from_pretrained(BERT_PATH)
            models['bert']      = BertForSequenceClassification.from_pretrained(BERT_PATH)
            models['bert'].to(device).eval()
            models['device']    = device
            models['bert_ok']   = True
        except Exception:
            models['bert_ok'] = False
    else:
        models['bert_ok'] = False

    # ── Influence Scorer ─────────────────────────────────────────────────
    if os.path.exists(INF_PATH):
        with open(INF_PATH, 'rb') as f:
            models['inf_scorer'] = pickle.load(f)

    # ── XGBoost baseline ────────────────────────────────────────────────
    if XGB_AVAILABLE and os.path.exists(XGB_PATH):
        with open(XGB_PATH, 'rb') as f:
            models['xgb'] = pickle.load(f)

    return models


@st.cache_data(show_spinner=False)
def load_dataset():
    """Load the cleaned articles dataset."""
    import pandas as pd
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None


def models_ready():
    """Check if minimum required models are available."""
    m = load_models()
    return ('tfidf' in m and 'encoder' in m)
