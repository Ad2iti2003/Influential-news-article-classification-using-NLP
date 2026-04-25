"""
utils/predictor.py
Core prediction logic — used by classify & ranking pages.
"""

import re, string, numpy as np
from utils.model_loader import load_models

STOP_WORDS = {
    'the','a','an','is','in','it','of','and','to','was','for','on',
    'are','with','as','at','be','by','from','or','that','this','have',
    'had','not','but','what','all','were','when','we','there','been',
    'has','would','their','he','she','they','said','say','says','also',
    'could','one','two','new','like','year','time','way'
}

INFLUENCE_LABELS = {0: 'Low', 1: 'Medium', 2: 'High'}
INFLUENCE_EMOJI  = {'High': '🔥', 'Medium': '⚡', 'Low': '💤'}
CATEGORY_EMOJI   = {
    'sports': '⚽', 'technology': '💻', 'business': '💼',
    'entertainment': '🎬', 'health': '🏥', 'science': '🔬',
    'politics': '🏛️', 'world': '🌍'
}


def clean_for_predict(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return ' '.join(tokens)


def get_sentiment(text: str):
    try:
        from textblob import TextBlob
        blob = TextBlob(text[:500])
        return blob.sentiment.polarity, blob.sentiment.subjectivity
    except:
        return 0.0, 0.0


def predict(title: str, content: str) -> dict:
    """
    Full prediction pipeline.
    Returns category, confidence, influence score & label.
    """
    models = load_models()
    clean  = clean_for_predict(title + ' ' + content)

    # ── 1. Category Prediction ─────────────────────────────────────────
    if models.get('bert_ok'):
        # BERT path
        import torch
        tok = models['tokenizer']
        enc = tok(clean[:512], max_length=256, padding='max_length',
                  truncation=True, return_tensors='pt')
        with torch.no_grad():
            out = models['bert'](
                input_ids      = enc['input_ids'].to(models['device']),
                attention_mask = enc['attention_mask'].to(models['device'])
            )
        probs      = torch.softmax(out.logits, dim=1).cpu().numpy()[0]
        cat_id     = int(probs.argmax())
        cat_conf   = float(probs.max())
        all_probs  = {models['categories'][i]: float(probs[i])
                      for i in range(len(models['categories']))}
    elif 'tfidf' in models and 'encoder' in models:
        # TF-IDF + XGBoost fallback
        vec    = models['tfidf'].transform([clean])
        clf    = models.get('xgb') or models.get('encoder')
        if hasattr(clf, 'predict_proba'):
            probs    = clf.predict_proba(vec)[0]
            cat_id   = int(probs.argmax())
            cat_conf = float(probs.max())
            all_probs = {models['categories'][i]: float(probs[i])
                         for i in range(min(len(models['categories']), len(probs)))}
        else:
            cat_id   = 0
            cat_conf = 0.5
            all_probs = {}
        cat_id = int(models['encoder'].transform(
            [models['encoder'].classes_[cat_id]])[0])
    else:
        cat_id, cat_conf, all_probs = 0, 0.5, {}

    category = models['categories'][cat_id] if cat_id < len(models['categories']) else 'unknown'

    # ── 2. Influence Score ─────────────────────────────────────────────
    pol, subj = get_sentiment(content)
    features  = np.array([[
        len(title),
        len(content),
        len(clean.split()),
        pol,
        subj,
        abs(pol),
        cat_id
    ]])

    if 'inf_scorer' in models:
        inf_proba    = models['inf_scorer'].predict_proba(features)[0]
        inf_label_id = int(inf_proba.argmax())
        inf_score    = float(inf_proba[2]*1.0 + inf_proba[1]*0.5 + inf_proba[0]*0.1)
        inf_score    = round(min(inf_score, 1.0), 4)
    else:
        # Heuristic fallback
        inf_score    = round(min(abs(pol)*0.4 + min(len(content)/5000, 0.6), 1.0), 4)
        inf_label_id = 2 if inf_score > 0.66 else (1 if inf_score > 0.33 else 0)

    inf_label = INFLUENCE_LABELS[inf_label_id]

    return {
        'category'        : category,
        'category_conf'   : cat_conf,
        'all_probs'       : all_probs,
        'influence_score' : inf_score,
        'influence_label' : inf_label,
        'sentiment_pol'   : round(pol, 4),
        'sentiment_subj'  : round(subj, 4),
        'word_count'      : len(clean.split()),
    }
