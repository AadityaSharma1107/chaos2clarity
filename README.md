# ComplaintIQ — ML Backend Submission

This version connects the ComplaintIQ dashboard to a real Python backend.

## What is implemented
- **Pandas**: loads `sentiment_training.csv`, cleans/normalizes text, and removes stopwords with an explicit `remove_stopwords` step.
- **TF-IDF**: converts cleaned complaint text into numerical features.
- **NumPy neural network from scratch**: one hidden ReLU layer + sigmoid output, manual forward pass, binary cross-entropy loss, backpropagation and gradient updates.
- **Scikit-learn**: train/test split, Accuracy/Precision/Recall/F1, and a confusion matrix on the held-out test set.
- **Matplotlib**: training-loss curve, training-accuracy curve, a confusion-matrix heatmap, and a hidden-layer weight-matrix (`W1`) heatmap, all saved in `model_artifacts/`.
- **Flask API**: `/api/analyze` accepts a CSV and returns analyzed complaints and dashboard metrics.
- **Dashboard integration**: the Upload CSV button sends the file to the Python backend when it is running, and falls back to the in-browser model otherwise.
- **Shared dataset (Phase 2 alignment)**: `sentiment_training.csv` holds the exact same 90 sentences (45 positive / 45 negative) as the in-browser JavaScript model's `POS`/`NEG` arrays in `index.html`, so both models are trained and evaluated on identical, class-balanced data instead of two unrelated toy sets. The earlier 30-sentence, imbalanced demo set produced an F1 of 0 on the Python side — this is now ~82% F1 on a matching held-out split.
- **Security/robustness fix**: the server no longer serves the whole project folder as static files (that previously published `backend.py`, `requirements.txt`, etc. over HTTP). It serves only `index.html` plus an explicit allowlist of plot filenames at `/api/plots/<name>`, so adding a new plot can't silently 404 by falling through to a stale route.

## Run
1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Run:
   `python -m pip install -r requirements.txt`
4. Start the backend:
   `python backend.py`
5. Open:
   `http://127.0.0.1:5000/`
6. Upload a CSV containing at least a `complaint` column. Optional columns are `id`, `date`, and `product`.

## API
- `GET /api/health` — backend status
- `GET /api/metrics` — neural-network evaluation metrics (now includes `confusion_matrix`)
- `POST /api/analyze` — analyze an uploaded CSV
- `GET /api/plots/training_loss.png` — training loss curve
- `GET /api/plots/training_accuracy.png` — training accuracy curve
- `GET /api/plots/confusion_matrix.png` — confusion matrix heatmap (test set)
- `GET /api/plots/weights.png` — hidden-layer weight matrix (`W1`) heatmap

## Why the backend shows "offline" until you run it
The "Python backend: offline" badge in the header is expected the moment you open `index.html` on its own (e.g. double-clicking the file, or viewing it via `file://`) — there is no server listening yet. It turns "online" only once `python backend.py` is actually running and you load the page from `http://127.0.0.1:5000/` (not from the raw file). This isn't something that can run itself in the background permanently; you (or whoever is demoing it) need to start it each time, from a terminal in this folder.

## Important academic note
The included sentiment training set is a small synthetic demonstration dataset so the project is self-contained. The neural network implementation is genuine, but the reported evaluation metrics are not a claim of production-grade sentiment accuracy. For a stronger submission, replace the demo training rows with a larger labeled customer-feedback dataset and report results on a held-out test set.

## In-browser neural network (new)
`index.html` now contains a self-contained JavaScript neural network, so the demo trains a real model even without Python:
- **Typed-array matrix ops** (`Float64Array`, NumPy-style `mm`, `mtm`, `mmt`) and a scikit-learn-style **TF-IDF** vectoriser.
- **Manual backpropagation** (chain rule, ReLU + sigmoid, binary cross-entropy, optional L2) with plain gradient descent or Adam.
- **Gradient check** button: compares backprop gradients to numerical derivatives (passes at ~1e-7).
- **Neural Lab page**: train/test split, loss and accuracy curves, accuracy / precision / recall / F1, confusion matrix, live predictions with per-word attribution.
- **CSV upload** uses the Python backend when it is running, and falls back to the in-browser model otherwise (selector in the header).
- Open `index.html` directly (double-click) or via `python backend.py`; both work. Chart.js (CDN) is only needed for the dashboard charts.

The in-browser model trains on a small hand-written 90-sentence dataset (see `POS` / `NEG` in `index.html`), so its test metrics (~70-75% on 22 texts) are a demonstration, not a production claim.
