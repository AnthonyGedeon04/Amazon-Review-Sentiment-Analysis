"""Shared test setup: a tiny stand-in model, so the tests run in seconds without the real DeBERTa
(which is too big for GitHub). It has the same files the real one has: Output/models/<name>/ and
Output/results/<name>.json with the cutoff offsets."""
import json
import sys
from pathlib import Path

import pytest
import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TINY_NAME = 'tiny_test_model'
WORDS = ['love', 'it', 'great', 'works', 'fine', 'speaker', 'is', 'weak', 'stopped', 'working', 'after', 'a', 'week', '.']


def make_tiny_model(root, name=TINY_NAME, offsets=(0.0, 0.0, 0.0), preset='medium'):
    """Writes a randomly initialised 1-layer BERT with 3 labels to <root>/Output/models/<name>."""
    model_dir = Path(root) / 'Output' / 'models' / name
    model_dir.mkdir(parents=True, exist_ok=True)
    vocab = model_dir / 'vocab.txt'
    vocab.write_text('\n'.join(['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'] + WORDS) + '\n')
    BertTokenizerFast(vocab_file=str(vocab)).save_pretrained(model_dir)
    torch.manual_seed(0)
    config = BertConfig(vocab_size=5 + len(WORDS), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                        intermediate_size=32, max_position_embeddings=128, num_labels=3)
    BertForSequenceClassification(config).save_pretrained(model_dir)
    results = Path(root) / 'Output' / 'results'
    results.mkdir(parents=True, exist_ok=True)
    (results / f'{name}.json').write_text(json.dumps({'model': name, 'preset': preset, 'offsets': list(offsets),
                                                      'macro_f1': 0.5, 'accuracy': 0.5}))
    return Path(root)


@pytest.fixture(scope='session')
def tiny_root(tmp_path_factory):
    return make_tiny_model(tmp_path_factory.mktemp('tiny'))
