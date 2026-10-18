"""
Lightweight Flask backend for Vercel deployment.
(see README in this folder for why this file is separate from the root backend.py)
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import re, os, json, math, traceback

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, 'model_artifacts')

app = Flask(__name__)
CORS(app)

STOPWORDS = set(
    "a an the and or of to in on at for with is are was were be been am i me my "
    "we our you your it its this that these those as by from so then than there "
    "their they them he she his her will would can could should do does did have "
    "has had just very really also".split()
)


def clean_text(s):
    s = '' if pd.isna(s) else str(s)
    s = s.lower()
    s = re.sub(r'http\S+|www\.\S+', ' ', s)
    s = s.replace("won't", "will not").replace("can't", "cannot")
    s = re.sub(r"n't\b", " not", s)
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def remove_stopwords(s):
    return ' '.join(w for w in s.split() if w not in STOPWORDS)


def ngrams(text):
    words = text.split()
    toks = list(words)
    toks += [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    return toks


# --- Everything that can fail (file I/O, JSON parsing) is wrapped so a bad
# path or missing file never crashes the whole import. If it fails, the app
# still starts; /api/debug and /api/health report what went wrong, and
# /api/analyze & /api/metrics return a clear 503 instead of the server dying. ---
VOCAB = VOCAB_IDX = IDF = W1 = b1 = W2 = b2 = metrics = None
LOAD_ERROR = None
try:
    MODEL_DIR_LISTING = os.listdir(MODEL_DIR) if os.path.isdir(MODEL_DIR) else None
    with open(os.path.join(MODEL_DIR, 'trained_model.json')) as f:
        _trained = json.load(f)
    VOCAB = _trained['vocab']
    VOCAB_IDX = {t: i for i, t in enumerate(VOCAB)}
    IDF = np.array(_trained['idf'])
    W1 = np.array(_trained['W1']); b1 = np.array(_trained['b1'])
    W2 = np.array(_trained['W2']); b2 = np.array(_trained['b2'])
    with open(os.path.join(MODEL_DIR, 'metrics.json')) as f:
        metrics = json.load(f)
except Exception:
    LOAD_ERROR = traceback.format_exc()
    MODEL_DIR_LISTING = None


def tfidf_transform(docs):
    X = np.zeros((len(docs), len(VOCAB)))
    for i, d in enumerate(docs):
        counts = {}
        for t in ngrams(d):
            j = VOCAB_IDX.get(t)
            if j is not None:
                counts[j] = counts.get(j, 0) + 1
        for j, c in counts.items():
            X[i, j] = (1 + math.log(c)) * IDF[j]
        norm = np.linalg.norm(X[i])
        if norm > 0:
            X[i] /= norm
    return X


def predict_proba(texts):
    X = tfidf_transform(texts)
    z1 = X @ W1 + b1
    a1 = np.maximum(0, z1)
    z2 = a1 @ W2 + b2
    z2 = np.clip(z2, -30, 30)
    return (1 / (1 + np.exp(-z2))).ravel()


CATEGORIES = ['Delivery','Billing & Refunds','Product Quality','Customer Service','App & Website','Account & Login','Cancellation','Other']

def category(text):
    t=text.lower()
    rules=[('Billing & Refunds', ['refund','charged','billing','payment','money']),('Delivery',['delivery','package','shipping','order','late']),('Account & Login',['login','password','account','sign in']),('App & Website',['app','website','crash','browser']),('Product Quality',['broken','damaged','quality','defect','product']),('Cancellation',['cancel','cancellation']),('Customer Service',['support','agent','service','help'])]
    for c,words in rules:
        if any(w in t for w in words): return c
    return 'Other'

def urgency_score(text):
    t=text.lower(); score=10
    if any(w in t for w in ['safety','injury','health','danger']): score+=50
    if any(w in t for w in ['legal','lawyer','court','fraud','security']): score+=40
    if any(w in t for w in ['repeated','again','third time','still not']): score+=25
    if any(w in t for w in ['refund','late','cancel','broken','missing']): score+=20
    if any(w in t for w in ['angry','furious','unacceptable','worst']): score+=10
    return min(100,score)

def analyze_df(df):
    df=df.copy(); df.columns=[str(c).strip().lower() for c in df.columns]
    if 'complaint' not in df.columns: raise ValueError('CSV must contain a complaint column.')
    df['complaint']=df['complaint'].fillna('').astype(str); df=df[df.complaint.str.strip()!=''].copy()
    df['clean_text']=df.complaint.map(clean_text).map(remove_stopwords)
    probs=predict_proba(df.clean_text.tolist())
    df['sentiment_score']=(probs*10).round(1)
    df['sentiment']=np.where(probs>=.5,'Happy','Unhappy')
    df['category']=df.complaint.map(category)
    df['urgency_score']=df.complaint.map(urgency_score)
    df['urgency']=pd.cut(df.urgency_score,[-1,39,69,100],labels=['Low','Medium','High']).astype(str)
    df['feedback_score']=(df.sentiment_score*0.75 + (10-df.urgency_score/100*10)*0.25).clip(1,10).round(1)
    df['feedback_band']=pd.cut(df.feedback_score,[0,4,7,10.01],labels=['BAD','AVERAGE','GOOD']).astype(str)
    df['priority_score']=(df.urgency_score*0.6 + (10-df.feedback_score)*8).clip(0,100).round(0).astype(int)
    if 'date' not in df: df['date']='2026-10-17'
    if 'product' not in df: df['product']='Uploaded'
    if 'id' not in df: df['id']=range(1,len(df)+1)
    return df

@app.get('/')
def index(): return send_from_directory(BASE,'index.html')

@app.get('/api/health')
def health(): return jsonify({'status':'ok' if not LOAD_ERROR else 'model_load_failed','backend':'Flask'})

@app.get('/api/debug')
def debug():
    return jsonify({
        'load_error': LOAD_ERROR,
        'base_dir': BASE,
        'model_dir': MODEL_DIR,
        'model_dir_exists': os.path.isdir(MODEL_DIR),
        'model_dir_listing': MODEL_DIR_LISTING,
    })

@app.get('/api/metrics')
def get_metrics():
    if LOAD_ERROR: return jsonify({'error': 'model not loaded', 'detail': LOAD_ERROR}), 503
    return jsonify(metrics)

PLOT_FILES = {'training_loss.png', 'training_accuracy.png', 'confusion_matrix.png', 'weights.png'}

@app.get('/api/plots/<name>')
def plot(name):
    if name not in PLOT_FILES: return ('Not found',404)
    return send_from_directory(MODEL_DIR,name)

@app.post('/api/analyze')
def analyze():
    if LOAD_ERROR: return jsonify({'error': 'Model failed to load on the server: ' + LOAD_ERROR.splitlines()[-1]}),503
    if 'file' not in request.files: return jsonify({'error':'Upload a CSV file.'}),400
    f=request.files['file']
    try:
        df=pd.read_csv(f)
        out=analyze_df(df)
        records=out.drop(columns=['clean_text']).replace({np.nan:None}).to_dict(orient='records')
        summary={
            'total':len(out),
            'unhappy_pct':round((out.sentiment.eq('Unhappy').mean()*100) if len(out) else 0,1),
            'high_urgency':int((out.urgency=='High').sum()),
            'avg_urgency':round(out.urgency_score.mean() if len(out) else 0,1),
            'avg_feedback':round(out.feedback_score.mean() if len(out) else 0,1),
            'good':int((out.feedback_band=='GOOD').sum()),'average':int((out.feedback_band=='AVERAGE').sum()),'bad':int((out.feedback_band=='BAD').sum()),
            'model_metrics':metrics
        }
        return jsonify({'summary':summary,'records':records})
    except Exception as e: return jsonify({'error':str(e)}),400

if __name__=='__main__':
    app.run(host='127.0.0.1',port=5000,debug=False)