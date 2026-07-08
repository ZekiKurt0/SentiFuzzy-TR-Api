from typing import Optional
from pydantic import BaseModel, Field


class SentimentRequest(BaseModel):
    # Kullanıcıdan gelecek metin. Boş olmaması için min_length ekliyoruz.
    text: str = Field(..., min_length=2, description="Analiz edilecek Türkçe metin")


class SentimentResponse(BaseModel):
    original_text: str
    cleaned_text: str
    sentiment_class: str      # Pozitif, Nötr, Negatif  (nihai/hibrit karar)
    sentiment_score: float    # 0.0 ile 1.0 arası BERT güven skoru (varsa)

    fuzzy_label: str          # Bulanık mantık çıktısı (Pozitif / Nötr / Negatif)
    fuzzy_score: float        # Bulanık mantık ham skoru (0-100 arası)

    bert_available: bool      # BERTürk modeli yüklenip yüklenemediği
    bert_label: Optional[str] = None
    bert_score: Optional[float] = None

    roots: list[str] = Field(default_factory=list)  # Zeyrek ile çıkarılan kökler
    negation_detected: bool = False  # "değil/yok" gibi bir olumsuzluk tespit edildi mi


class HealthResponse(BaseModel):
    status: str
    bert_available: bool
    zemberek_available: bool
    fuzzy_available: bool