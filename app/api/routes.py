import logging

from fastapi import APIRouter, HTTPException

from app.schemas.payload import SentimentRequest, SentimentResponse, HealthResponse
from app.services.prediction import analyze_sentiment
from app.services.model_loader import get_engines

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """
    Servisin ve alt motorların (BERTürk, Zeyrek, Fuzzy Logic) ayakta olup
    olmadığını kontrol eder. BERTürk modeli henüz eğitilip klasöre
    konulmadıysa bert_available=false döner ama servis yine de çalışır
    (fuzzy-logic-only modda).
    """
    engines = get_engines()
    return HealthResponse(
        status="ok",
        bert_available=engines.bert_available,
        zemberek_available=engines.zemberek is not None,
        fuzzy_available=engines.fuzzy is not None,
    )


@router.post("/predict", response_model=SentimentResponse, tags=["Sentiment"])
def predict_sentiment(payload: SentimentRequest):
    """
    Türkçe bir metin alır; Zeyrek + Fuzzy Logic + BERTürk hibrit motorunu
    çalıştırıp nihai duygu sınıfını (Pozitif / Nötr / Negatif) döndürür.
    """
    try:
        return analyze_sentiment(payload.text)
    except Exception as e:
        logger.exception("Tahmin sırasında beklenmeyen hata")
        raise HTTPException(status_code=500, detail=f"Tahmin başarısız: {e}")
