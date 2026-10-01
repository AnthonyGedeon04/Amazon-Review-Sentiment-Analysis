"""Web app for the review sentiment model: type a review, get not satisfied / neutral / satisfied.

Run it from the project folder, with the same .venv the notebooks use:
    pip install -r Requirements/requirements-app.txt
    python app/app.py
then open http://127.0.0.1:5000

The model is loaded once when the app starts (it takes a little while and a few hundred MB of RAM), then every
request reuses it.
"""
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(APP_DIR))
from predict import LABELS, SentimentModel  # noqa: E402
from model_files import MODEL_NAME, ensure_model_files  # noqa: E402

MAX_CHARS = 5000        # the model only reads the first ~64 tokens anyway
GITHUB_URL = os.environ.get('GITHUB_URL', '')   # shown in the footer when set
MODEL_ROOT = Path(os.environ.get('MODEL_ROOT', ROOT))   # folder holding Output/models and Output/results (the tests point it elsewhere)

ensure_model_files(root=MODEL_ROOT)
print(f'Loading {MODEL_NAME} ...', flush=True)
_t0 = time.time()
model = SentimentModel(MODEL_NAME, MODEL_ROOT)
model.predict_proba(['warm up'])   # the first call is slow, so do it before any visitor arrives
print(f'Model ready in {time.time() - _t0:.0f}s', flush=True)
_lock = threading.Lock()   # one prediction at a time keeps memory flat on a small machine

_result = json.load(open(MODEL_ROOT / 'Output' / 'results' / f'{MODEL_NAME}.json'))
MODEL_INFO = {
    'name': MODEL_NAME,
    'macro_f1': _result.get('macro_f1'),
    'accuracy': _result.get('accuracy'),
}

app = Flask(__name__)


def score(text, title=''):
    """Label plus a score per class. The scores include the same cutoff adjustment as the label (and the reported
    test score), so the highest bar is always the predicted label."""
    with _lock:
        probs = model.predict_proba([text], [title])[0]
    logits = np.log(probs + 1e-12) + model.offsets
    adjusted = np.exp(logits - logits.max())
    adjusted /= adjusted.sum()
    return LABELS[int(adjusted.argmax())], {label: round(float(p), 4) for label, p in zip(LABELS, adjusted)}


@app.get('/')
def index():
    return render_template('index.html', info=MODEL_INFO, github_url=GITHUB_URL, max_chars=MAX_CHARS)


@app.post('/api/predict')
def api_predict():
    data = request.get_json(silent=True) or {}
    text = str(data.get('text', '')).strip()
    title = str(data.get('title', '')).strip()
    if not text:
        return jsonify(error='Write a review first.'), 400
    if len(text) + len(title) > MAX_CHARS:
        return jsonify(error=f'Please keep it under {MAX_CHARS:,} characters.'), 400
    t0 = time.perf_counter()
    label, scores = score(text, title)
    return jsonify(label=label, scores=scores, ms=round((time.perf_counter() - t0) * 1000))


@app.get('/api/health')
def health():
    return jsonify(status='ok', model=MODEL_NAME)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f'Open http://127.0.0.1:{port} in your browser (Ctrl+C to stop)', flush=True)
    app.run(host=os.environ.get('HOST', '127.0.0.1'), port=port, debug=False)
