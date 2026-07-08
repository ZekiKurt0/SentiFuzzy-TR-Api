import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentFuzzyEngine:
    """
    SentiFuzzy-TR için Bulanık Mantık (Fuzzy Logic) Çıkarım Motoru.
    Girdi olarak kelimelerin pozitif ve negatif skorlarını alır,
    kurallara göre cümlenin nihai duygu durumunu hesaplar.
    """
    def __init__(self):
        logger.info("⏳ Bulanık Mantık Motoru başlatılıyor...")
        self._setup_fuzzy_system()
        logger.info("✅ Bulanık Mantık Motoru hazır.")

    def _setup_fuzzy_system(self):
        
        self.poz_skor = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'pozitif_skor')
        self.neg_skor = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'negatif_skor')

        
        self.nihai_duygu = ctrl.Consequent(np.arange(0, 101, 1), 'nihai_duygu')

        
        self.poz_skor.automf(3, names=['dusuk', 'orta', 'yuksek'])
        self.neg_skor.automf(3, names=['dusuk', 'orta', 'yuksek'])

        
        self.nihai_duygu['negatif'] = fuzz.trimf(self.nihai_duygu.universe, [0, 0, 50])
        self.nihai_duygu['notr'] = fuzz.trimf(self.nihai_duygu.universe, [25, 50, 75])
        self.nihai_duygu['pozitif'] = fuzz.trimf(self.nihai_duygu.universe, [50, 100, 100])

       
        kural1 = ctrl.Rule(self.poz_skor['yuksek'] & self.neg_skor['dusuk'], self.nihai_duygu['pozitif'])
        kural2 = ctrl.Rule(self.poz_skor['dusuk'] & self.neg_skor['yuksek'], self.nihai_duygu['negatif'])
        kural3 = ctrl.Rule(self.poz_skor['orta'] & self.neg_skor['orta'], self.nihai_duygu['notr'])
        kural4 = ctrl.Rule(self.poz_skor['dusuk'] & self.neg_skor['dusuk'], self.nihai_duygu['notr'])
        kural5 = ctrl.Rule(self.poz_skor['yuksek'] & self.neg_skor['yuksek'], self.nihai_duygu['notr']) # Çelişkili durum

        
        self.kural_kontrolu = ctrl.ControlSystem([kural1, kural2, kural3, kural4, kural5])
        self.simulasyon = ctrl.ControlSystemSimulation(self.kural_kontrolu)

    def hesapla(self, toplam_pozitif, toplam_negatif):
        """
        Dışarıdan gelen pozitif ve negatif skor toplamlarını sisteme verir 
        ve nihai duygu skorunu döndürür.
        """
        try:
            
            self.simulasyon.input['pozitif_skor'] = min(max(toplam_pozitif, 0.0), 1.0)
            self.simulasyon.input['negatif_skor'] = min(max(toplam_negatif, 0.0), 1.0)
            
            
            self.simulasyon.compute()
            
            sonuc_skor = self.simulasyon.output['nihai_duygu']
            return sonuc_skor
            
        except Exception as e:
            logger.error(f"❌ Bulanık mantık hesaplama hatası: {e}")
            return 50.0 


fuzzy_engine = SentimentFuzzyEngine()