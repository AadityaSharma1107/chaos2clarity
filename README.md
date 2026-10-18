# Chaos2Clarity — Vercel deploy build

This folder is a slimmed-down copy of the project made only for Vercel's
225 MB serverless function size limit. The real submission (full
pandas + scikit-learn + matplotlib pipeline, trained live on every run) is
`backend.py` and `requirements.txt` one level up — grade/run that one.

`backend.py` here loads `model_artifacts/trained_model.json` (the TF-IDF
vocabulary/IDF and the trained NumPy network's weights), which was exported
from one run of the real `backend.py`, instead of re-fitting scikit-learn and
re-rendering matplotlib plots on every cold start. The four PNGs in
`model_artifacts/` are the exact plots matplotlib produced during that run —
nothing here regenerates them.

## To deploy
In the Vercel project's settings, set **Root Directory** to this `deploy`
folder (not the repo root), then import/deploy as normal. If you retrain the
model (edit `sentiment_training.csv` or the network upstream), regenerate
`trained_model.json` by running the root `backend.py` once and re-exporting
`vectorizer.idf_` / `get_feature_names_out()` / `model.W1,b1,W2,b2`, then copy
the new `model_artifacts/` files into this folder.