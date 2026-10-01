# Web app

A small Flask app: type a review (and an optional title), click **Analyze**, and the final DeBERTa-v3 model shows
*not satisfied*, *neutral* or *satisfied* with a bar per class. Plain HTML, CSS and JavaScript, no frontend framework.

```
app/
  app.py            Flask server: loads the model once, serves the page and POST /api/predict
  model_files.py    finds the model files on disk
  templates/index.html
  static/style.css, static/app.js
```

The app imports `SentimentModel` from `predict.py`, so the page gives exactly the same answers as the command line
and notebook 07 (same title + text input, same cutoff adjustment).

## Run it on your laptop

From the project folder in VS Code's terminal, with the same `.venv` the notebooks use:

```powershell
.venv\Scripts\activate
pip install -r Requirements/requirements-app.txt
python app/app.py
```

Wait for `Model ready`, then open <http://127.0.0.1:5000>. Stop it with `Ctrl+C`.

- It needs `Output/models/deberta_v3_base_medium/` and `Output/results/deberta_v3_base_medium.json` (made by notebook 05).
- Loading takes a little while and uses around 1 to 1.5 GB of RAM (an estimate for DeBERTa-base on CPU), so close notebook kernels first.
- Flask prints a "development server" warning. That is fine for running it on your own laptop.
- Different port: `$env:PORT=8000; python app/app.py`.

## Running it from a fresh clone

The trained model is not on GitHub (it is too big). Someone who clones the repo creates it by running notebooks 01
and 05, which save it to `Output/models/deberta_v3_base_medium/`, and then starts the app as above.

## API

`POST /api/predict` with JSON `{"text": "...", "title": "..."}` (title optional) returns

```json
{"label": "satisfied", "scores": {"not satisfied": 0.01, "neutral": 0.02, "satisfied": 0.97}, "ms": 85}
```

The scores include the per-class cutoff adjustment, so the highest score is always the label. `GET /api/health`
returns `{"status": "ok"}` once the model is loaded.
