"""
prediction.py

main.py içindeki 'duygu_analizi_hibrit' demo fonksiyonunun, API'de
kullanılabilecek şekilde yeniden yapılandırılmış hali.

Akış:
1) Zeyrek ile metnin kökleri çıkarılır.
2) Köklerin sözlükteki (lexicon) pozitif/negatif skorları toplanır.
   Bu adımda artık NEGASYON (değil/yok) da dikkate alınıyor (bkz. aşağı).
3) Bulanık mantık (skfuzzy) motoru bu toplam skorlardan 0-100 arası
   bir "fuzzy skor" üretir.
4) BERTürk modeli (varsa) aynı metin için bağımsız bir tahmin üretir.
5) İki karar birleştirilir. Normalde BERT önceliklidir (daha güçlü bir
   dil modeli olduğu için), AMA metinde açık bir negasyon kalıbı
   ("kötü değil", "iyi değildi" gibi) tespit edildiyse ve BERT'in kendi
   güveni de yüksek değilse (< NEGATION_OVERRIDE_CONFIDENCE), negasyonu
   dikkate alan fuzzy sonucuna güvenilir. Model yoksa/hata verirse
   sistem zaten fuzzy'ye düşer (graceful degradation).

NEDEN BU EK GEREKTİ: "Bugün kötü bir gün değil bence" gibi cümlelerde
hem eski (negasyonsuz) fuzzy motoru hem de BERT modeli yanlış yönde
("Negatif") tahmin üretiyordu. BERT tarafındaki asıl kök neden, eğitim
verisinin (yıldız puanından türetilmiş yorumlar) bu tür dolaylı/
olumsuzlamalı ifadeleri yeterince içermemesi — bu ancak yeniden eğitimle
(veri setine negasyonlu örnekler eklenerek) tam çözülür. Ama fuzzy
katmanındaki negasyon köründen kaynaklanan hata şimdi tamamen giderildi,
ve BERT düşük güvenle yanıldığında bu düzeltilmiş fuzzy sonucu devreye
giriyor.
"""
import logging

from app.core.config import get_settings
from app.services.model_loader import get_engines
from app.schemas.payload import SentimentResponse

logger = logging.getLogger(__name__)

# "değil", "yok" gibi sözcüklerin hemen öncesindeki (en fazla bu kadar
# kelime geride kalan) bir duygu kökünü tersine çevirdiğini varsayıyoruz.
# "kötü(0) bir(1) gün(2) değil(3)" -> pencere=3 bunu tam yakalar.
_NEGATION_MARKERS = {"değil", "yok", "değildir", "değilim", "değilsin"}
_NEGATION_WINDOW = 3

# BERT'in kendi güveni bu eşiğin ALTINDAYSA ve negasyon tespit edildiyse,
# nihai karar için negasyon-düzeltmeli fuzzy sonucu tercih edilir.
_NEGATION_OVERRIDE_CONFIDENCE = 0.80


def _clean_text(text: str) -> str:
    """Basit temizlik: baştaki/sondaki boşluklar, fazla boşluk sıkıştırma."""
    return " ".join(text.strip().split())


def _fuzzy_label_from_score(score: float) -> str:
    settings = get_settings()
    if score > settings.fuzzy_positive_threshold:
        return "Pozitif"
    if score < settings.fuzzy_negative_threshold:
        return "Negatif"
    return "Nötr"


def _score_roots_with_negation(roots: list[str], lexicon: dict) -> tuple[float, float, bool]:
    """
    Lexicon skorlarını toplarken, her duygu kökünün ilerisindeki (en fazla
    _NEGATION_WINDOW kelime içinde) bir negasyon işareti olup olmadığına
    bakar; varsa o kökün poz/neg katkısını TERSİNE çevirir.

    Dönüş: (toplam_poz, toplam_neg, negasyon_tespit_edildi_mi)
    """
    toplam_poz, toplam_neg = 0.0, 0.0
    negation_found = False

    for i, kok in enumerate(roots):
        skor = lexicon.get(kok)
        if not skor:
            continue

        pencere = roots[i + 1: i + 1 + _NEGATION_WINDOW]
        negated = any(w in _NEGATION_MARKERS for w in pencere)

        poz, neg = skor.get("poz", 0.0), skor.get("neg", 0.0)
        if negated:
            negation_found = True
            toplam_poz += neg  # anlam ters döndüğü için poz/neg yer değiştirir
            toplam_neg += poz
        else:
            toplam_poz += poz
            toplam_neg += neg

    return toplam_poz, toplam_neg, negation_found


_BERT_LABEL_MAP = {
    "LABEL_0": "Negatif",
    "LABEL_1": "Nötr",
    "LABEL_2": "Pozitif",
}


def analyze_sentiment(text: str) -> SentimentResponse:
    engines = get_engines()
    cleaned = _clean_text(text)

    # --- 1) Zeyrek: kök çıkarımı -------------------------------------
    try:
        roots = engines.zemberek.cumle_koklerini_cikar(cleaned)
    except Exception as e:  # Zeyrek nadiren bilinmeyen tokenlarda patlayabilir
        logger.error(f"Zeyrek kök çıkarma hatası: {e}")
        roots = cleaned.lower().split()

    # --- 2) Lexicon skorlaması (negasyon farkında) ------------------------
    toplam_poz, toplam_neg, negation_detected = _score_roots_with_negation(
        roots, engines.lexicon
    )

    # --- 3) Bulanık mantık -----------------------------------------------
    fuzzy_score = engines.fuzzy.hesapla(toplam_poz, toplam_neg)
    fuzzy_label = _fuzzy_label_from_score(fuzzy_score)

    # --- 4) BERTürk (varsa) ------------------------------------------------
    bert_label = None
    bert_score = None
    if engines.bert_available:
        try:
            bert_sonuc = engines.bert.tahmin_et(cleaned)
            if bert_sonuc:
                en_iyi = max(bert_sonuc, key=lambda x: x["score"])
                bert_label = _BERT_LABEL_MAP.get(en_iyi["label"], en_iyi["label"])
                bert_score = float(en_iyi["score"])
        except Exception as e:
            logger.error(f"BERT tahmin hatası: {e}")

    # --- 5) Nihai karar --------------------------------------------------
    # Varsayılan: BERT varsa önceliklidir.
    # İstisna: negasyon tespit edildi VE BERT'in kendi güveni düşükse
    # (net emin değilse), negasyonu doğru işleyen fuzzy sonucuna güvenilir.
    use_fuzzy_override = (
        bert_label is not None
        and negation_detected
        and bert_score is not None
        and bert_score < _NEGATION_OVERRIDE_CONFIDENCE
        and bert_label != fuzzy_label
    )

    if bert_label is not None and not use_fuzzy_override:
        sentiment_class = bert_label
        sentiment_score = bert_score
    else:
        sentiment_class = fuzzy_label
        sentiment_score = fuzzy_score / 100.0
        if use_fuzzy_override:
            logger.info(
                f"Negasyon tespit edildi, düşük güvenli BERT sonucu ('{bert_label}', "
                f"{bert_score:.2f}) yerine fuzzy sonucu ('{fuzzy_label}') kullanıldı."
            )

    return SentimentResponse(
        original_text=text,
        cleaned_text=cleaned,
        sentiment_class=sentiment_class,
        sentiment_score=round(sentiment_score, 4),
        fuzzy_label=fuzzy_label,
        fuzzy_score=round(fuzzy_score, 2),
        bert_available=engines.bert_available,
        bert_label=bert_label,
        bert_score=round(bert_score, 4) if bert_score is not None else None,
        roots=roots,
        negation_detected=negation_detected,
    )
