from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import re, os, json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(BASE, 'sentiment_training.csv')

# On a normal machine this writes model_artifacts/ next to backend.py. On a
# read-only serverless deployment (e.g. Vercel Functions) the project folder
# itself can't be written to, so fall back to the system temp dir in that case.
MODEL_DIR = os.path.join(BASE, 'model_artifacts')
try:
    os.makedirs(MODEL_DIR, exist_ok=True)
except OSError:
    import tempfile
    MODEL_DIR = os.path.join(tempfile.gettempdir(), 'complaintiq_model_artifacts')
    os.makedirs(MODEL_DIR, exist_ok=True)

# Serve ONLY index.html from this process (no static_folder=BASE). Serving the
# whole project directory as static files used to also publish backend.py,
# requirements.txt, etc. over HTTP, and left the plot route competing with a
# catch-all static rule -- any plot name not in an allowlist (e.g. a plot
# added later, like the ones Phase 2 adds below) would silently fall through
# to that catch-all and 404 instead of hitting /api/plots/<name>. Keeping
# Flask static-free and whitelisting plot files explicitly avoids both.
app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Phase 2: shared training data.
# sentiment_training.csv is the SAME 90-sentence, class-balanced dataset used
# by the in-browser JavaScript model (index.html's POS/NEG arrays). Both
# models train and evaluate on identical text, so the browser demo and the
# Python/NumPy pipeline are directly comparable instead of being two
# unrelated toy datasets.
# ---------------------------------------------------------------------------

# Same stopword list used by the JS tokenizer, so "removing stopwords" means
# the same thing on both sides of the project.
STOPWORDS = set(
    "a an the and or of to in on at for with is are was were be been am i me my "
    "we our you your it its this that these those as by from so then than there "
    "their they them he she his her will would can could should do does did have "
    "has had just very really also".split()
)


def clean_text(s):
    """Lowercase + normalize text (Pandas/regex step)."""
    s = '' if pd.isna(s) else str(s)
    s = s.lower()
    s = re.sub(r'http\S+|www\.\S+', ' ', s)
    s = s.replace("won't", "will not").replace("can't", "cannot")
    s = re.sub(r"n't\b", " not", s)
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def remove_stopwords(s):
    """Explicit stopword-removal step (kept separate from clean_text so it is
    easy to point to for the Phase 2 'removing stopwords' requirement)."""
    return ' '.join(w for w in s.split() if w not in STOPWORDS)


class NumpySentimentNN:
    """Two-layer neural net implemented only with NumPy for the train/predict math."""
    def __init__(self, input_dim, hidden=32, lr=0.08, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2/input_dim), (input_dim, hidden))
        self.b1 = np.zeros((1, hidden))
        self.W2 = rng.normal(0, np.sqrt(2/hidden), (hidden, 1))
        self.b2 = np.zeros((1, 1))
        self.lr = lr

    @staticmethod
    def sigmoid(z):
        z = np.clip(z, -30, 30)
        return 1/(1+np.exp(-z))

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self.sigmoid(z2)
        return z1, a1, a2

    def fit(self, X, y, epochs=300):
        y = y.reshape(-1,1)
        hist = {'loss': [], 'accuracy': []}
        for _ in range(epochs):
            z1,a1,p = self.forward(X)
            eps=1e-8
            loss = -np.mean(y*np.log(p+eps)+(1-y)*np.log(1-p+eps))
            pred=(p>=0.5).astype(int)
            acc=np.mean(pred==y)
            dz2=p-y
            dW2=a1.T @ dz2 / len(X); db2=np.mean(dz2,axis=0,keepdims=True)
            da1=dz2 @ self.W2.T; dz1=da1*(z1>0)
            dW1=X.T @ dz1 / len(X); db1=np.mean(dz1,axis=0,keepdims=True)
            self.W2-=self.lr*dW2; self.b2-=self.lr*db2
            self.W1-=self.lr*dW1; self.b1-=self.lr*db1
            hist['loss'].append(float(loss)); hist['accuracy'].append(float(acc))
        return hist

    def predict_proba(self,X):
        return self.forward(X)[2].ravel()

vectorizer = None
model = None
metrics = {}


def train_model():
    global vectorizer, model, metrics

    # --- Data Preprocessing (Pandas): load, clean, normalize, strip stopwords ---
    df = pd.read_csv(TRAIN_CSV)
    df['clean'] = df['text'].map(clean_text).map(remove_stopwords)

    X_train, X_test, y_train, y_test = train_test_split(
        df['clean'], df['label'], test_size=.25, random_state=42, stratify=df['label']
    )

    # --- Model Implementation: TF-IDF features + from-scratch NumPy NN ---
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=700, min_df=1, sublinear_tf=True)
    A = vectorizer.fit_transform(X_train).toarray(); B=vectorizer.transform(X_test).toarray()
    model = NumpySentimentNN(A.shape[1], hidden=32, lr=.08)
    hist=model.fit(A,y_train.to_numpy(),epochs=300)
    probs=model.predict_proba(B); pred=(probs>=.5).astype(int)
    y_test_arr = y_test.to_numpy()

    # --- Evaluation (Scikit-learn): accuracy / precision / recall / F1 + confusion matrix ---
    cm = confusion_matrix(y_test_arr, pred, labels=[0,1])
    metrics={
        'accuracy': round(accuracy_score(y_test_arr,pred)*100,1),
        'precision': round(precision_score(y_test_arr,pred,zero_division=0)*100,1),
        'recall': round(recall_score(y_test_arr,pred,zero_division=0)*100,1),
        'f1': round(f1_score(y_test_arr,pred,zero_division=0)*100,1),
        'test_size': int(len(y_test_arr)),
        'train_size': int(len(y_train)),
        'epochs': 300,
        'features': int(A.shape[1]),
        'confusion_matrix': cm.tolist(),  # [[TN, FP], [FN, TP]]
    }

    # --- Visualization (Matplotlib) ---
    plt.figure(figsize=(8,4.2)); plt.plot(hist['loss']); plt.title('ComplaintIQ -- NumPy NN Training Loss'); plt.xlabel('Epoch'); plt.ylabel('Binary Cross-Entropy'); plt.grid(alpha=.2); plt.tight_layout(); plt.savefig(os.path.join(MODEL_DIR,'training_loss.png'),dpi=150); plt.close()
    plt.figure(figsize=(8,4.2)); plt.plot(np.array(hist['accuracy'])*100); plt.title('ComplaintIQ -- NumPy NN Training Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)'); plt.grid(alpha=.2); plt.tight_layout(); plt.savefig(os.path.join(MODEL_DIR,'training_accuracy.png'),dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(4.6,4.2))
    im = ax.imshow(cm, cmap='Purples')
    ax.set_title('ComplaintIQ -- Confusion Matrix (test set)')
    ax.set_xticks([0,1]); ax.set_xticklabels(['Pred: Unhappy','Pred: Happy'])
    ax.set_yticks([0,1]); ax.set_yticklabels(['True: Unhappy','True: Happy'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i,j]), ha='center', va='center',
                     color='white' if cm[i,j] > cm.max()/2 else 'black', fontsize=14, fontweight='bold')
    fig.colorbar(im, fraction=0.046, pad=0.04); fig.tight_layout(); fig.savefig(os.path.join(MODEL_DIR,'confusion_matrix.png'),dpi=150); plt.close(fig)

    # Weight visualization: W1 (hidden units x TF-IDF features) gives a simple
    # picture of what the from-scratch network learned.
    plt.figure(figsize=(8,4.2))
    vmax = float(np.max(np.abs(model.W1))) or 1.0
    plt.imshow(model.W1.T, aspect='auto', cmap='coolwarm', vmin=-vmax, vmax=vmax)
    plt.colorbar(label='weight value')
    plt.title('ComplaintIQ -- Hidden Layer Weights (W1, hidden units x TF-IDF features)')
    plt.xlabel('TF-IDF feature index'); plt.ylabel('Hidden unit')
    plt.tight_layout(); plt.savefig(os.path.join(MODEL_DIR,'weights.png'),dpi=150); plt.close()

    with open(os.path.join(MODEL_DIR,'metrics.json'),'w') as f: json.dump(metrics,f,indent=2)
    return metrics

train_model()

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
    probs=model.predict_proba(vectorizer.transform(df.clean_text).toarray())
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
def health(): return jsonify({'status':'ok','model':'NumPy neural net','backend':'Flask'})

@app.get('/api/metrics')
def get_metrics(): return jsonify(metrics)

PLOT_FILES = {'training_loss.png', 'training_accuracy.png', 'confusion_matrix.png', 'weights.png'}

@app.get('/api/plots/<name>')
def plot(name):
    if name not in PLOT_FILES: return ('Not found',404)
    return send_from_directory(MODEL_DIR,name)

@app.post('/api/analyze')
def analyze():
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
    print(f"ComplaintIQ backend ready -- accuracy {metrics['accuracy']}% / F1 {metrics['f1']}% on {metrics['test_size']} held-out test sentences.")
    print("Open http://127.0.0.1:5000/ in your browser (the dashboard is served by this same process).")
    app.run(host='127.0.0.1',port=5000,debug=False)
