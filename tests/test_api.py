"""
tests/test_api.py

Ağır bağımlılıkları (zeyrek, transformers/torch) CI ortamında kurmadan
API katmanını test edebilmek için zeyrek ve transformers sahte
(stub) modüllerle değiştiriliyor. Böylece bu testler saniyeler içinde,
GPU/model dosyası olmadan koşabilir.

Gerçek modelle uçtan uca test etmek istersen (ml_models/berturk_sentiment_pro
doluyken) bu stub'ları kaldırıp doğrudan `from app import create_app`
diyebilirsin.
"""
import sys
import types

import pytest


@pytest.fixture(scope="module")
def client():
    # --- zeyrek stub -------------------------------------------------
    zeyrek_stub = types.ModuleType("zeyrek")

    class _Analysis:
        def __init__(self, lemma):
            self.lemma = lemma

    class _MorphAnalyzer:
        def analyze(self, word):
            return [[_Analysis(word.lower())]]

    zeyrek_stub.MorphAnalyzer = _MorphAnalyzer
    sys.modules["zeyrek"] = zeyrek_stub

    # --- transformers stub (model yok senaryosu) ----------------------
    transformers_stub = types.ModuleType("transformers")

    def _pipeline(*args, **kwargs):
        raise OSError("test ortamında model yok (beklenen davranış)")

    transformers_stub.pipeline = _pipeline
    sys.modules["transformers"] = transformers_stub

    from fastapi.testclient import TestClient
    from app import create_app

    return TestClient(create_app())


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["zemberek_available"] is True
    assert body["fuzzy_available"] is True
    # Bu test ortamında model dosyaları yok, dolayısıyla False bekleniyor
    assert body["bert_available"] is False


def test_predict_positive_text(client):
    r = client.post("/predict", json={"text": "Kargo çok hızlı geldi harika bir ürün"})
    assert r.status_code == 200
    body = r.json()
    assert body["sentiment_class"] == "Pozitif"
    assert body["fuzzy_label"] == "Pozitif"
    assert 0.0 <= body["fuzzy_score"] <= 100.0
    assert isinstance(body["roots"], list) and len(body["roots"]) > 0


def test_predict_negative_text(client):
    r = client.post("/predict", json={"text": "Ürün berbat geldi ve kargo çok yavaş"})
    assert r.status_code == 200
    body = r.json()
    assert body["sentiment_class"] == "Negatif"


def test_predict_rejects_short_text(client):
    r = client.post("/predict", json={"text": "a"})
    assert r.status_code == 422  # min_length=2 ihlali


def test_root_serves_ui(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "SentiFuzzy" in r.text


def test_api_info(client):
    r = client.get("/api")
    assert r.status_code == 200
    assert "message" in r.json()


def test_predict_handles_negation(client):
    r = client.post("/predict", json={"text": "Bugün kötü bir gün değil bence"})
    assert r.status_code == 200
    body = r.json()
    assert body["negation_detected"] is True
