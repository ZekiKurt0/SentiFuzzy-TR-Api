import zeyrek
import logging

# Zeyrek'in kalabalık log çıktılarını (APPENDING RESULT) susturuyoruz
logging.getLogger("zeyrek").setLevel(logging.WARNING)

# Motoru başlatıyoruz
analyzer = zeyrek.MorphAnalyzer()

# SentiFuzzy-TR için örnek bir Lexicon (Duygu Sözlüğü)
# Gerçek projede bu veriler büyük bir JSON veya CSV dosyasından çekilecektir
duygu_sozlugu = {
    "kargo": {"pozitif": 0.0, "negatif": 0.0},
    "ulaşmak": {"pozitif": 0.4, "negatif": 0.0},
    "film": {"pozitif": 0.0, "negatif": 0.0},
    "son": {"pozitif": 0.0, "negatif": 0.0},
    "gerçekten": {"pozitif": 0.5, "negatif": 0.0},
    "inanılmaz": {"pozitif": 0.9, "negatif": 0.1}
}

test_sentences = [
    "Kargom iki gün içinde elime ulaştı", 
    "Filmin sonu gerçekten inanılmazdı"
]

print("--- SentiFuzzy-TR Kök Eşleştirme Testi ---\n")

for sentence in test_sentences:
    words = sentence.split()
    print(f"Orijinal Cümle: '{sentence}'")
    
    cumle_pozitif_skor = 0.0
    cumle_negatif_skor = 0.0
    
    for word in words:
        analysis = analyzer.analyze(word)
        if analysis:
            # Kelimenin kökünü alıyoruz ve küçük harfe çeviriyoruz
            kok = analysis[0][0].lemma.lower()
            
            # Kök eğer duygu sözlüğümüzde varsa skorlarını çekiyoruz
            if kok in duygu_sozlugu:
                poz = duygu_sozlugu[kok]["pozitif"]
                neg = duygu_sozlugu[kok]["negatif"]
                cumle_pozitif_skor += poz
                cumle_negatif_skor += neg
                
                print(f"  -> {word:15} | Kök: {kok:10} | Pozitif: {poz}, Negatif: {neg}")
            else:
                print(f"  -> {word:15} | Kök: {kok:10} | Sözlükte Yok (Nötr)")
                
    print(f">> Cümlenin Toplam Ham Skoru -> Pozitif: {cumle_pozitif_skor:.1f}, Negatif: {cumle_negatif_skor:.1f}\n")