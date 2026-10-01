"""Predict review sentiment (not satisfied / neutral / satisfied) with the final fine-tuned DeBERTa-v3 model.

Usage (from the project folder, with the .venv active):
    python predict.py "Alexa keeps disconnecting from wifi"
    python predict.py "Works fine, speaker is weak" --title "It's ok"
    python predict.py --file my_reviews.csv          # CSV with a 'text' column (and optionally 'title')

In Python / a notebook:
    from predict import SentimentModel
    model = SentimentModel()
    model.predict(["Love it!", "Stopped working after a week"])
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent
LABELS = ['not satisfied', 'neutral', 'satisfied']
MODEL_NAME = 'deberta_v3_base_medium'   # the final model chosen in notebooks 04-06


def model_input(title, text):
    """Same input format as training: user-written title + ' . ' + review text."""
    title = (title or '').strip()
    return f'{title} . {text}' if title else str(text)


class SentimentModel:
    def __init__(self, name=MODEL_NAME, root=ROOT):
        model_dir = Path(root) / 'Output' / 'models' / name
        if not model_dir.exists():
            raise FileNotFoundError(f'{model_dir} not found: train it with notebook 05 first.')
        result = json.load(open(Path(root) / 'Output' / 'results' / f'{name}.json'))
        self.offsets = np.array(result.get('offsets', [0.0, 0.0, 0.0]))   # cutoff adjustment tuned in training
        self.max_len = 64 if result.get('preset') in ('lite', 'medium', 'medium_2ep', 'smoke') else 256 if result.get('preset') == 'full' else 128
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir, dtype=torch.float32).eval()

    @torch.inference_mode()
    def predict_proba(self, texts, titles=None, batch_size=16, progress=False, save_path=None):
        """Returns an n x 3 array of probabilities in LABELS order (before the cutoff adjustment).
        Small batches keep memory low on a laptop. With save_path (a .npy file), finished batches are saved as it
        goes, so if it crashes or is stopped, calling it again with the same inputs continues where it left off."""
        titles = titles if titles is not None else [''] * len(texts)
        inputs = [model_input(t, x) for t, x in zip(titles, texts)]
        done = np.zeros((0, 3), dtype=np.float32)
        if save_path is not None and Path(save_path).exists():
            done = np.load(save_path)
            if len(done) > len(inputs):
                done = np.zeros((0, 3), dtype=np.float32)   # saved for different inputs: start over
            elif progress and len(done):
                print(f'continuing from {len(done):,} saved predictions', flush=True)
        out = [done]
        for n_batch, i in enumerate(range(len(done), len(inputs), batch_size)):
            enc = self.tok(inputs[i:i + batch_size], truncation=True, max_length=self.max_len, padding=True, return_tensors='pt')
            out.append(torch.softmax(self.model(**enc).logits.float(), -1).numpy().astype(np.float32))
            if (n_batch + 1) % 25 == 0:
                if save_path is not None:
                    np.save(save_path, np.concatenate(out))
                if progress:
                    print(f'{min(i + batch_size, len(inputs)):,} / {len(inputs):,} reviews', flush=True)
        probs = np.concatenate(out)
        if save_path is not None:
            np.save(save_path, probs)
        return probs

    def predict(self, texts, titles=None, batch_size=16):
        """Returns a label per review, using the same cutoff adjustment as the reported test score."""
        if isinstance(texts, str):
            texts = [texts]
        probs = self.predict_proba(texts, titles, batch_size)
        return [LABELS[i] for i in (np.log(probs + 1e-12) + self.offsets).argmax(1)]


def main():
    ap = argparse.ArgumentParser(description='Predict Amazon review sentiment')
    ap.add_argument('text', nargs='?', help='review text')
    ap.add_argument('--title', default='', help='optional review title')
    ap.add_argument('--file', help='CSV with a text column (and optionally title); writes <file>_predictions.csv')
    args = ap.parse_args()
    if not args.text and not args.file:
        ap.error('give a review text or --file')

    model = SentimentModel()
    if args.file:
        import pandas as pd
        df = pd.read_csv(args.file)
        titles = df['title'].fillna('').astype(str).tolist() if 'title' in df else None
        probs = model.predict_proba(df['text'].astype(str).tolist(), titles)
        df['prediction'] = [LABELS[i] for i in (np.log(probs + 1e-12) + model.offsets).argmax(1)]
        for j, label in enumerate(LABELS):
            df[f'prob_{label}'] = probs[:, j].round(3)
        out = Path(args.file).with_name(Path(args.file).stem + '_predictions.csv')
        df.to_csv(out, index=False)
        print(df['prediction'].value_counts().to_string())
        print(f'saved {out}')
    else:
        probs = model.predict_proba([args.text], [args.title])[0]
        label = LABELS[int((np.log(probs + 1e-12) + model.offsets).argmax())]
        print(f'{label}  (' + ', '.join(f'{l}: {p:.2f}' for l, p in zip(LABELS, probs)) + ')')


if __name__ == '__main__':
    main()
