import importlib
import sys

import pytest

from conftest import TINY_NAME


@pytest.fixture(scope='module')
def client(tiny_root):
    mp = pytest.MonkeyPatch()
    mp.setenv('MODEL_NAME', TINY_NAME)
    mp.setenv('MODEL_ROOT', str(tiny_root))
    for name in ('app', 'model_files'):
        sys.modules.pop(name, None)
    app_module = importlib.import_module('app')
    yield app_module.app.test_client()
    mp.undo()


def test_health(client):
    response = client.get('/api/health')
    assert response.status_code == 200
    assert response.get_json() == {'status': 'ok', 'model': TINY_NAME}


def test_home_page_loads(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'<form' in response.data


def test_predict_returns_a_label_and_a_score_per_class(client):
    response = client.post('/api/predict', json={'text': 'Stopped working after a week', 'title': 'Broken'})
    assert response.status_code == 200
    body = response.get_json()
    assert set(body['scores']) == {'not satisfied', 'neutral', 'satisfied'}
    assert abs(sum(body['scores'].values()) - 1) < 1e-3
    assert body['label'] == max(body['scores'], key=body['scores'].get)   # the top bar is the label


def test_title_is_optional(client):
    assert client.post('/api/predict', json={'text': 'Love it'}).status_code == 200


@pytest.mark.parametrize('payload', [{}, {'text': ''}, {'text': '   '}])
def test_empty_review_is_rejected(client, payload):
    response = client.post('/api/predict', json=payload)
    assert response.status_code == 400
    assert 'error' in response.get_json()


def test_too_long_review_is_rejected(client):
    response = client.post('/api/predict', json={'text': 'a' * 6000})
    assert response.status_code == 400
