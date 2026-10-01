import json

import numpy as np
import pytest

from conftest import TINY_NAME, make_tiny_model
from predict import LABELS, SentimentModel, model_input


def test_model_input_joins_title_and_text():
    assert model_input('Great', 'Love it') == 'Great . Love it'


def test_model_input_without_title_is_just_the_text():
    assert model_input('', 'Love it') == 'Love it'
    assert model_input(None, 'Love it') == 'Love it'
    assert model_input('   ', 'Love it') == 'Love it'


def test_missing_model_gives_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match='notebook 05'):
        SentimentModel('no_such_model', tmp_path)


def test_predict_proba_returns_one_probability_row_per_review(tiny_root):
    model = SentimentModel(TINY_NAME, tiny_root)
    probs = model.predict_proba(['Love it', 'Stopped working after a week', 'Speaker is weak'], ['Great', '', ''])
    assert probs.shape == (3, 3)
    assert np.allclose(probs.sum(axis=1), 1, atol=1e-5)
    assert (probs >= 0).all()


def test_predict_returns_known_labels_and_accepts_a_single_string(tiny_root):
    model = SentimentModel(TINY_NAME, tiny_root)
    assert all(label in LABELS for label in model.predict(['Love it', 'Works fine']))
    assert len(model.predict('Love it')) == 1


def test_offsets_are_read_from_the_results_file(tiny_root):
    model = SentimentModel(TINY_NAME, tiny_root)
    saved = json.load(open(tiny_root / 'Output' / 'results' / f'{TINY_NAME}.json'))['offsets']
    assert model.offsets.tolist() == saved


@pytest.mark.parametrize('offsets, expected', [
    ((50.0, 0.0, 0.0), 'not satisfied'),
    ((0.0, 50.0, 0.0), 'neutral'),
    ((0.0, 0.0, 50.0), 'satisfied'),
])
def test_cutoff_offsets_change_the_predicted_label(tmp_path, offsets, expected):
    # a large offset for one class must win, which shows the offsets are applied to the prediction
    model = SentimentModel(TINY_NAME, make_tiny_model(tmp_path, offsets=offsets))
    assert model.predict(['Love it', 'Speaker is weak']) == [expected, expected]


@pytest.mark.parametrize('preset, max_len', [('lite', 64), ('medium', 64), ('full', 256), ('cpu', 128)])
def test_max_length_matches_the_training_preset(tmp_path, preset, max_len):
    model = SentimentModel(TINY_NAME, make_tiny_model(tmp_path, preset=preset))
    assert model.max_len == max_len


def test_saved_progress_is_reused(tiny_root, tmp_path):
    model = SentimentModel(TINY_NAME, tiny_root)
    texts = ['Love it'] * 40
    save = tmp_path / 'progress.npy'
    first = model.predict_proba(texts, batch_size=4, save_path=save)
    assert save.exists()
    np.testing.assert_allclose(model.predict_proba(texts, batch_size=4, save_path=save), first)
