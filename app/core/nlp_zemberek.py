import zeyrek
import logging

# API ve sunucu dağıtımlarında terminali temiz tutmak için logları susturuyoruz
logging.getLogger("zeyrek").setLevel(logging.WARNING)

class ZemberekPipeline:
    """
    SentiFuzzy-TR projesi için saf Python tabanlı morfolojik analiz motoru.
    Java bağımlılığını ortadan kaldırmak için Zeyrek altyapısını kullanır.
    """
    def __init__(self):
        print("⏳ Morfoloji Motoru (Zeyrek) başlatılıyor...")
        self.analyzer = zeyrek.MorphAnalyzer()
        print("✅ Morfoloji Motoru başarıyla başlatıldı.")

    def kelime_koku_bul(self, word):
        """
        Verilen kelimenin kökünü (lemma) döndürür. 
        Bulamazsa kelimenin orijinal halini küçük harfle döndürür.
        """
        analysis = self.analyzer.analyze(word)
        if analysis and len(analysis) > 0 and len(analysis[0]) > 0:
            return analysis[0][0].lemma.lower()
        return word.lower()

    def cumle_koklerini_cikar(self, sentence):
        """
        Bir cümledeki tüm kelimelerin köklerini bir liste olarak döndürür.
        """
        words = sentence.split()
        kokler = [self.kelime_koku_bul(word) for word in words]
        return kokler

# Motoru dışarıdan içe aktarmak (import) için hazır bir obje oluşturuyoruz
zemberek_engine = ZemberekPipeline()