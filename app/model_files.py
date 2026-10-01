"""Makes sure the final model's files are on disk before the app loads it.

predict.py expects:
    Output/models/<name>/                 the fine-tuned model and tokenizer
    Output/results/<name>.json            its scores and cutoff offsets

On your laptop both already exist after notebook 05. Anywhere else (the Hugging Face Space, or someone who cloned
the GitHub repo) set MODEL_REPO to the Hugging Face model repo you uploaded them to, and they are downloaded once.

    python app/model_files.py             # download now (the Space's Dockerfile does this while building)
"""
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from predict import MODEL_NAME as DEFAULT_MODEL_NAME  # noqa: E402

MODEL_NAME = os.environ.get('MODEL_NAME', DEFAULT_MODEL_NAME)
MODEL_REPO = os.environ.get('MODEL_REPO', '').strip()   # e.g. your-username/echo-dot-sentiment-deberta


def ensure_model_files(name=MODEL_NAME, repo=MODEL_REPO, root=ROOT):
    model_dir = Path(root) / 'Output' / 'models' / name
    result_file = Path(root) / 'Output' / 'results' / f'{name}.json'
    if model_dir.exists() and result_file.exists():
        return
    if not repo:
        sys.exit(f'Model files not found:\n  {model_dir}\n  {result_file}\n'
                 'Train the model with notebook 05, or set MODEL_REPO to the Hugging Face repo that holds it.')
    from huggingface_hub import snapshot_download
    print(f'Downloading {repo} into {model_dir} ...', flush=True)
    snapshot_download(repo_id=repo, local_dir=model_dir)
    if not result_file.exists():   # the scores file is uploaded next to the model, under its original name
        uploaded = model_dir / f'{name}.json'
        if not uploaded.exists():
            sys.exit(f'{repo} has no {name}.json (the file from Output/results). Upload it to the model repo too.')
        result_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(uploaded, result_file)
    print('Model files ready.', flush=True)


if __name__ == '__main__':
    ensure_model_files()
