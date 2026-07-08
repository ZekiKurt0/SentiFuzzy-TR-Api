"""
model_loader.py

NOT: Bu dosya, projedeki 'model_loder.py' (yazım hatası) dosyasının
düzeltilmiş ve doldurulmuş halidir; eski boş dosya kaldırıldı.

Görevi: app/core altındaki ağır motorları (Zeyrek, BERTürk, Fuzzy Logic)
ve duygu sözlüğünü tek bir yerden, tek sefer (singleton) olarak yüklemek.
FastAPI uygulaması ayağa kalkarken bu motorlar burada ısıtılır (warm-up),
her istek için yeniden yüklenmez.
"""
import json
import logging
import os

from app.core.config import get_settings, project_root
from app.core.nlp_zemberek import zemberek_engine
from app.core.fuzzy_logic import fuzzy_engine
from app.core.ml_model import bert_engine

logger = logging.getLogger(__name__)


class EngineRegistry:
    """Tüm çıkarım motorlarını ve sözlüğü tutan basit bir kap (container)."""

    def __init__(self):
        self.zemberek = zemberek_engine
        self.fuzzy = fuzzy_engine
        self.bert = bert_engine
        self.lexicon: dict = {}
        self._load_lexicon()

    def _load_lexicon(self):
        settings = get_settings()
        lexicon_path = os.path.join(project_root(), settings.lexicon_path)
        try:
            with open(lexicon_path, "r", encoding="utf-8") as f:
                self.lexicon = json.load(f)
            logger.info(f"✅ Duygu sözlüğü yüklendi: {len(self.lexicon)} kök ({lexicon_path})")
        except FileNotFoundError:
            logger.warning(f"⚠️ Duygu sözlüğü bulunamadı ({lexicon_path}), boş sözlükle devam ediliyor.")
            self.lexicon = {}

    @property
    def bert_available(self) -> bool:
        return self.bert.classifier is not None


# Uygulama genelinde tek bir kayıt (registry) - import edildiği an motorlar yüklenir.
engines = EngineRegistry()


def get_engines() -> EngineRegistry:
    """FastAPI dependency injection için kullanılabilecek erişim fonksiyonu."""
    return engines
