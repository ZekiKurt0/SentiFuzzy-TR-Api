import os
from transformers import pipeline
import logging

from app.core.config import get_settings, project_root

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentBertEngine:
    """
    SentiFuzzy-TR projesi için ince ayar (fine-tune) yapılmış 
    BERTürk modelini çalıştıran çekirdek sınıf.
    """
    def __init__(self, model_folder: str | None = None):
        # model_folder verilmezse config.py'deki merkezi ayardan okunur.
        # Bu sayede train.py'nin kaydettiği yer ile burada okunan yer
        # HER ZAMAN aynı yerden (config.py) yönetilir.
        settings = get_settings()
        model_folder = model_folder or settings.model_dir

        # Kodun nerede çalıştığından bağımsız olarak kök dizini dinamik buluyoruz
        base_dir = project_root()
        full_path = os.path.join(base_dir, model_folder)
        
        logger.info(f"⏳ BERT Modeli yükleniyor... ({full_path})")
        try:
            # top_k=None parametresi ile her 3 sınıfın (Pozitif, Nötr, Negatif) olasılıklarını da istiyoruz
            self.classifier = pipeline(
                "text-classification", 
                model=full_path, 
                tokenizer=full_path,
                top_k=None 
            )
            logger.info("✅ BERT Modeli başarıyla hafızaya alındı.")
        except Exception as e:
            logger.error(f"❌ Model yüklenirken hata oluştu: {e}")
            self.classifier = None

    def tahmin_et(self, metin):
        """
        Gelen metni modele sokar ve duygu olasılıklarını döndürür.
        """
        if not self.classifier:
            return None
            
        sonuc = self.classifier(metin)
        return sonuc[0] # [[{label: ..., score: ...}, ...]] formatından kurtarmak için

# Başka dosyalardan doğrudan çağırabilmek için objeyi oluşturuyoruz
bert_engine = SentimentBertEngine()