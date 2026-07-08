from app.core.nlp_zemberek import zemberek_engine
from app.core.fuzzy_logic import fuzzy_engine
from app.core.ml_model import bert_engine

# ---------------------------------------------------------
# MOCK LEXICON (Duygu Sözlüğü)
# ---------------------------------------------------------
lexicon = {
    "harika": {"poz": 0.9, "neg": 0.0},
    "berbat": {"poz": 0.0, "neg": 0.9},
    "güzel": {"poz": 0.7, "neg": 0.1},
    "kötü": {"poz": 0.1, "neg": 0.8},
    "kargo": {"poz": 0.0, "neg": 0.0},
    "hızlı": {"poz": 0.8, "neg": 0.0},
    "yavaş": {"poz": 0.0, "neg": 0.7},
    "ulaşmak": {"poz": 0.3, "neg": 0.0},
    "ürün": {"poz": 0.0, "neg": 0.0},
    "kırık": {"poz": 0.0, "neg": 0.8}
}

def duygu_analizi_hibrit(cumle):
    print(f"\n{'='*65}")
    print(f"🔍 İNCELENEN CÜMLE: '{cumle}'")
    print(f"{'='*65}")
    
    # ---------------------------------------------------------
    # 1. BULANIK MANTIK (FUZZY LOGIC) ANALİZİ
    # ---------------------------------------------------------
    kokler = zemberek_engine.cumle_koklerini_cikar(cumle)
    toplam_poz, toplam_neg = 0.0, 0.0
    
    for kok in kokler:
        if kok in lexicon:
            toplam_poz += lexicon[kok]["poz"]
            toplam_neg += lexicon[kok]["neg"]
            
    fuzzy_skor = fuzzy_engine.hesapla(toplam_poz, toplam_neg)
    fuzzy_durum = "Nötr 😐"
    if fuzzy_skor > 55: fuzzy_durum = "Pozitif 🟢"
    elif fuzzy_skor < 45: fuzzy_durum = "Negatif 🔴"
    
    print("🧠 [1] BULANIK MANTIK (Kural Tabanlı) KARARI")
    print(f"   Çıkarılan Kökler: {kokler}")
    print(f"   Skor: {fuzzy_skor:.2f}/100 -> {fuzzy_durum}")
    print("-" * 65)
    
    # ---------------------------------------------------------
    # 2. BERTÜRK DERİN ÖĞRENME ANALİZİ
    # ---------------------------------------------------------
    print("🤖 [2] BERTÜRK (Veri Tabanlı) KARARI")
    bert_sonuc = bert_engine.tahmin_et(cumle)
    
    if bert_sonuc:
        # Colab'de eğitilen label_id yapısı: 0=Negatif, 1=Nötr, 2=Pozitif
        for tahmin in bert_sonuc:
            label = tahmin['label']
            score = tahmin['score']
            
            if label == 'LABEL_0': label_isim = "Negatif 🔴"
            elif label == 'LABEL_1': label_isim = "Nötr 😐"
            elif label == 'LABEL_2': label_isim = "Pozitif 🟢"
            else: label_isim = label
            
            print(f"   {label_isim}: %{score*100:.2f} Olasılık")
    else:
        print("   [Model yüklenemedi veya tahmin başarısız]")
    
    print(f"{'='*65}\n")

# ---------------------------------------------------------
# SİSTEM TESTLERİ
# ---------------------------------------------------------
if __name__ == "__main__":
    test_cumleleri = [
        "Kargom çok hızlı ulaştı harika bir ürün",
        "Ürün berbat geldi ve kargo çok yavaş",
        "Ürün güzel ama kargo çok yavaş", # Zorlu bağlaç testi
        "Beş Yıldız. Çok kullanışlı" # Veri setinden tanıdık bir cümle
    ]

    for cumle in test_cumleleri:
        duygu_analizi_hibrit(cumle)