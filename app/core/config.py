import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    SentiFuzzy-TR uygulama ayarları.
    Tüm değerler ortam değişkeni (environment variable) ile override edilebilir,
    örn: SENTIFUZZY_MODEL_DIR=/models/berturk_sentiment_pro
    """

    app_name: str = "SentiFuzzy-TR API"
    app_version: str = "0.1.0"

    # BERTürk fine-tune edilmiş modelin bulunduğu klasör (proje köküne göre göreceli
    # veya mutlak yol olabilir). train.py bu klasöre kaydeder, ml_model.py buradan okur.
    model_dir: str = "ml_models/berturk_sentiment_pro"

    # Sınıf sırası train.py / data_preprocessing.py ile birebir aynı olmalı.
    label_map: dict = {
        "LABEL_0": "Negatif",
        "LABEL_1": "Nötr",
        "LABEL_2": "Pozitif",
    }

    # Duygu sözlüğü (lexicon) dosyasının yolu
    lexicon_path: str = "data/lexicon_tr.json"

    # Bulanık mantık eşikleri (main.py'deki demo ile aynı varsayılanlar)
    fuzzy_positive_threshold: float = 55.0
    fuzzy_negative_threshold: float = 45.0

    # CORS
    cors_allow_origins: list = ["*"]

    model_config = SettingsConfigDict(env_prefix="SENTIFUZZY_", env_file=".env")


@lru_cache
def get_settings() -> Settings:
    """Ayarları tek sefer oluşturup önbelleğe alır (singleton)."""
    return Settings()


def project_root() -> str:
    """Proje kök dizinini, bu dosyanın konumundan bağımsız şekilde döndürür."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
