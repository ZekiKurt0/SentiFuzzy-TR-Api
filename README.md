

# SentiFuzzy-TR

**Türkçe metinler için Bulanık Mantık (Fuzzy Logic) + BERTürk hibrit duygu analizi sistemi.**

Bir cümlenin **Pozitif / Nötr / Negatif** olduğunu iki farklı yöntemi birleştirerek
tahmin eder:

1. **Zeyrek** (Türkçe morfolojik analiz) ile cümledeki kelimelerin kökleri çıkarılır,
   bir duygu sözlüğüne (lexicon) göre puanlanır, **Bulanık Mantık** (skfuzzy) motoru
   bu puanları 0-100 arası bir skora dönüştürür.
2. **BERTürk** (ince ayar yapılmış bir BERT modeli) aynı cümle için bağımsız bir
   tahmin üretir.
3. İkisi birleştirilir: BERT varsa ve kendinden eminse ona güvenilir; BERT yoksa
   veya emin değilse (özellikle "değil/yok" gibi olumsuzlama içeren cümlelerde)
   sistem otomatik olarak bulanık mantık sonucuna döner.

Sonuç, bir **FastAPI** servisi (`/predict`) ve basit bir **web arayüzü** (`/`)
üzerinden sunulur.

---

## İçindekiler

1. [Mimariye genel bakış](#1-mimariye-genel-bakış)
2. [Kurulum](#2-kurulum)
3. [Veri setini hazırlama](#3-veri-setini-hazırlama)
4. [Modeli eğitme (Colab / GPU)](#4-modeli-eğitme-colab--gpu)
5. [Eğitilen modeli projeye yerleştirme](#5-eğitilen-modeli-projeye-yerleştirme)
6. [API'yi çalıştırma](#6-apiyi-çalıştırma)
7. [Web arayüzü](#7-web-arayüzü)
8. [Testler](#8-testler)
9. [Docker](#9-docker)
10. [Hugging Face Spaces'e canlıya alma](#10-hugging-face-spacese-canlıya-alma)
11. [Proje yapısı](#11-proje-yapısı)
12. [Bilinen sınırlamalar](#12-bilinen-sınırlamalar)

---

## 1. Mimariye genel bakış

```
main.py                      # uvicorn giriş noktası (FastAPI app)
app/
  __init__.py                 # create_app() factory (CORS, router, statik arayüz)
  core/
    config.py                 # merkezi ayarlar (model yolu, eşikler, vs.)
    nlp_zemberek.py            # Zeyrek morfolojik kök çıkarımı
    fuzzy_logic.py              # skfuzzy kural tabanlı motor
    ml_model.py                 # BERTürk pipeline (transformers)
  services/
    model_loader.py             # tüm motorların tekil (singleton) yüklenmesi
    prediction.py                # hibrit karar mantığı (fuzzy + BERT + negasyon)
  schemas/
    payload.py                   # Pydantic request/response modelleri
  api/
    routes.py                     # /health, /predict endpoint'leri
  static/
    index.html                    # kullanıcıya yönelik web arayüzü
data/
  data_preprocessing.py         # ham veri -> final_set.csv (temizlik dahil)
  lexicon_tr.json                # duygu sözlüğü (kelime -> poz/neg puan)
  processed/final_set.csv        # eğitime hazır, dengeli, temizlenmiş veri seti
  raw/                            # ham veri (bu repoda YOK, ayrıca indirilmeli)
train.py                       # Colab/GPU üzerinde çalıştırılacak eğitim scripti
scripts/demo_hybrid.py          # API'siz, konsolda hızlı test scripti
tests/test_api.py               # zeyrek/transformers stub'larıyla hızlı API testleri
```

**Önemli tasarım kararı — model klasörü boşken de sistem çalışır:**
`ml_models/berturk_sentiment_pro/` klasörü boşsa (henüz eğitilmiş model
konulmadıysa) API çökmez; `/health` çağrısı `bert_available: false` döner
ve sistem otomatik olarak sadece bulanık mantık motoruyla çalışır. Yani
aşağıdaki adımları takip etmeden de projeyi ayağa kaldırıp deneyebilirsin —
gerçek BERT tahminlerini görmek için 3-5. adımları tamamlaman gerekir.

---

## 2. Kurulum

```bash
git clone <bu-repo>
cd SentiFuzzy-TR
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 3. Veri setini hazırlama

Bu repoda **`data/processed/final_set.csv`** zaten hazır halde bulunuyor
(15.000 satır, 3 sınıf dengeli: 5000 Negatif / 5000 Nötr / 5000 Pozitif).
Doğrudan 4. adıma (eğitim) geçebilirsin.

Kendi ham verinle sıfırdan üretmek istersen:

1. `data/raw/dataset_1.csv` ve `data/raw/dataset_2.csv`'yi (veya kendi ham
   veri kaynaklarını) `data/raw/` klasörüne koy.
2. `data/data_preprocessing.py` dosyasının en üstündeki yorum bloğunda,
   bu iki ham veri kaynağının nasıl birleştirilip `all_datasets_shuffled.csv`
   üretildiği gösteriliyor — referans olarak kullan.
3. Sonra temizlik + dengeli örnekleme adımını çalıştır:
   ```bash
   python data/data_preprocessing.py
   ```
   Bu komut şunları yapar:
   - Metinlerin başındaki "Beş Yıldız.", "Bir Yıldız." gibi puan ifadelerini
     temizler (aksi halde model gerçek duyguyu değil bu yüzeysel kalıbı
     öğrenmeye çalışır).
   - Tam yinelenen (duplicate) satırları atar.
   - Her sınıftan eşit sayıda örnek seçerek dengeli bir `final_set.csv`
     üretir (`data/processed/final_set.csv`).

---

## 4. Modeli eğitme (Colab / GPU)

`train.py`, BERTürk'ü (`dbmdz/bert-base-turkish-cased`) `final_set.csv`
üzerinde ince ayar (fine-tune) yapar. **GPU gerektirir** — CPU'da eğitim
saatlerce sürebilir, bu yüzden Google Colab önerilir.

### Adımlar

1. [Google Colab](https://colab.research.google.com/)'da yeni bir not
   defteri aç. **Runtime → Change runtime type → GPU** seç (ücretsiz T4
   genelde yeterlidir).
2. Bu projenin klasörünü Colab'a yükle (Google Drive'a bağlayabilir ya da
   doğrudan sürükle-bırak ile yükleyebilirsin) — en az şu dosyalar gerekli:
   - `train.py`
   - `data/processed/final_set.csv`
3. Gerekli kütüphaneleri kur:
   ```python
   !pip install transformers datasets scikit-learn torch -q
   ```
4. Eğitimi başlat:
   ```python
   !python train.py
   ```
5. Script şunları otomatik yapar:
   - Veriyi %80 eğitim / %20 test olacak şekilde, **sınıf oranlarını
     koruyarak** (stratified) böler.
   - `dbmdz/bert-base-turkish-cased` modelini indirir, 3 sınıflı bir
     sınıflandırma başlığı ekler.
   - 3 epoch boyunca eğitir; her epoch sonunda **accuracy, macro-F1 ve
     sınıf başı F1** metriklerini konsola basar (bunlara bakarak modelin
     gerçekten iyi öğrenip öğrenmediğini takip edebilirsin — sadece loss'a
     bakmak yeterli değildir).
   - En iyi checkpoint'i otomatik seçer (`load_best_model_at_end=True`).
   - Eğitilmiş modeli **`ml_models/berturk_sentiment_pro/`** klasörüne
     kaydeder.
6. Eğitim bitince o klasörü (içindeki `config.json`, `model.safetensors`,
   `vocab.txt`, `tokenizer_config.json` gibi dosyalarla birlikte) Colab'dan
   bilgisayarına indir (klasörü zip'leyip indirmek en pratiği):
   ```python
   !zip -r berturk_sentiment_pro.zip ml_models/berturk_sentiment_pro
   from google.colab import files
   files.download('berturk_sentiment_pro.zip')
   ```

---

## 5. Eğitilen modeli projeye yerleştirme

İndirdiğin `berturk_sentiment_pro.zip`'i aç ve içeriğini, projenin
kök dizininde **tam olarak şu yola** kopyala:

```
ml_models/berturk_sentiment_pro/
├── config.json
├── model.safetensors        (veya pytorch_model.bin)
├── tokenizer_config.json
├── vocab.txt
└── ... (tokenizer'a ait diğer dosyalar)
```

> **Neden bu yol önemli?** `app/core/config.py` içindeki `model_dir`
> ayarı, API'nin modeli tam olarak bu klasörden okumasını söylüyor
> (`app/core/ml_model.py`). Yanlış bir klasöre koyarsan API model
> dosyalarını bulamaz ve sessizce fuzzy-logic-only moda düşer — hata
> vermez, ama BERT tahminlerini de göremezsin. Kontrol etmek için
> `/health` endpoint'ine bak; `bert_available: true` görmelisin.

---

## 6. API'yi çalıştırma

```bash
uvicorn main:app --reload
# veya
python main.py
```

- Web arayüzü: `http://localhost:8000/`
- Swagger (API dokümantasyonu / manuel test paneli): `http://localhost:8000/docs`
- Sağlık kontrolü: `http://localhost:8000/health`

### Örnek istek

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Kargo çok hızlı geldi, harika bir ürün"}'
```

### Örnek cevap

```json
{
  "original_text": "Kargo çok hızlı geldi, harika bir ürün",
  "cleaned_text": "Kargo çok hızlı geldi, harika bir ürün",
  "sentiment_class": "Pozitif",
  "sentiment_score": 0.9055,
  "fuzzy_label": "Pozitif",
  "fuzzy_score": 81.43,
  "bert_available": true,
  "bert_label": "Pozitif",
  "bert_score": 0.9055,
  "roots": ["kargo", "çok", "hızlı", "gelmek", "harika", "bir", "ürün"],
  "negation_detected": false
}
```

| Alan | Açıklama |
|---|---|
| `sentiment_class` | Sistemin nihai kararı (hibrit mantıkla belirlenir) |
| `fuzzy_label` / `fuzzy_score` | Sadece bulanık mantık motorunun sonucu (0-100) |
| `bert_label` / `bert_score` | Sadece BERTürk'ün sonucu (varsa) |
| `negation_detected` | "değil/yok" gibi bir olumsuzlama tespit edildi mi |
| `roots` | Zeyrek'in çıkardığı kelime kökleri |

---

## 7. Web arayüzü

Kök adres (`/`), retro bir "cihaz kontrol paneli" temalı basit bir arayüz
sunar: metin kutusu, döner gösterge (fuzzy skor), LCD ekran (nihai karar),
Pozitif/Nötr/Negatif LED'leri, BERT güven çubuğu ve kök kelime etiketleri.
Kaynağı `app/static/index.html`'de — vanilla HTML/CSS/JS, ek bir frontend
derleme adımı gerektirmez, doğrudan FastAPI tarafından sunulur.

---

## 8. Testler

```bash
pytest tests/test_api.py -v
```

Bu testler `zeyrek` ve `transformers` paketlerini sahte (stub) modüllerle
değiştirir, böylece **GPU veya eğitilmiş model dosyası olmadan** saniyeler
içinde koşar. Şunları doğrular: `/health`, `/predict` (pozitif/negatif/
negasyonlu cümleler), kısa metin validasyonu, ana sayfa arayüzünün açılması.

Gerçek modelle uçtan uca denemek istersen, `ml_models/berturk_sentiment_pro`
doluyken testteki stub bloklarını kaldırıp doğrudan test edebilirsin.

---

## 9. Docker

```bash
docker build -t sentifuzzy-tr .
docker run -p 7860:7860 sentifuzzy-tr
# tarayıcıda: http://localhost:7860/
```

`Dockerfile`, Hugging Face Spaces'in Docker SDK'sıyla uyumlu olacak şekilde
hazırlandı (port `7860`, root olmayan kullanıcı). Modeli konteynerin
içine dahil etmek için (adım 5'te belirtilen) `ml_models/berturk_sentiment_pro/`
klasörünün build sırasında mevcut olması gerekir.

---

## 10. Hugging Face Spaces'e canlıya alma

### Sıfırdan

1. https://huggingface.co/spaces/new → SDK: **Docker**, Hardware: ücretsiz
   **CPU basic** (BERT-base inference CPU'da rahat çalışır, GPU sadece
   eğitim için gerekliydi).
2. ```bash
   git clone https://huggingface.co/spaces/KULLANICI_ADIN/sentifuzzy-tr
   cd sentifuzzy-tr
   # proje dosyalarını (data/raw hariç) buraya kopyala
   git lfs install
   git lfs track "ml_models/berturk_sentiment_pro/*.safetensors"
   git add .
   git commit -m "SentiFuzzy-TR API + model"
   git push
   ```

### Kod zaten bir GitHub reposundaysa

Sıfırdan başlamana gerek yok — aynı yerel repoya HF Space'ini **ikinci bir
remote** olarak ekleyip oraya da push edebilirsin:

```bash
git remote add space https://huggingface.co/spaces/KULLANICI_ADIN/sentifuzzy-tr
git lfs install
git add .gitignore .gitattributes ml_models/berturk_sentiment_pro/
git commit -m "Model ve HF Space ayarlarını ekle"
git push origin main         # GitHub
git push --force space main:main   # Hugging Face Space (ilk push'ta gerekebilir)
```

Not: `.gitignore`'un **model klasörünü artık hariç tutmadığından** emin ol
(`git status` ile modelin "yeni dosya" olarak göründüğünü doğrula) — aksi
halde model sessizce (hatasız) commit'e girmez.

Space sayfasında **"Building"** durumunu izle (model 400MB+ olduğu için
ilk build birkaç dakika sürebilir). Bitince adres şu şekilde olur:
```
https://KULLANICI_ADIN-sentifuzzy-tr.hf.space
```

---

## 11. Proje yapısı

Bkz. [1. Mimariye genel bakış](#1-mimariye-genel-bakış).

---

## 12. Bilinen sınırlamalar

- **Duygu sözlüğü küçük** (`data/lexicon_tr.json`, ~20 kök). Gerçek
  kullanımda SentiTurkNet gibi hazır bir Türkçe duygu sözlüğüyle
  genişletmek fuzzy-logic tarafının kalitesini artırır.
- **Negasyon tespiti basit bir pencere kuralıdır** (kelime bazlı, 3
  kelimelik kapsam), gerçek bir bağımlılık ayrıştırıcısı (dependency
  parser) değil. "Ürün pahalı değil ama kalitesiz" gibi negasyonun başka
  bir kelimeyi hedeflediği karmaşık cümlelerde hâlâ hata payı var. Kalıcı
  çözüm, eğitim verisine bilinçli olarak negasyonlu örnekler eklemektir.
- **`dataset_2` (film yorumları)** puan eşiğiyle etiketlenmiş; sarkastik
  yorumlarda bu etiketleme gürültülü olabilir.
- Sağlık kontrolü (`/health`) ve `bert_available` alanı, model dosyaları
  eksikse veya bozuksa bunu şeffaf şekilde bildirir — API yine de çalışmaya
  devam eder (sadece fuzzy-logic-only modda).
