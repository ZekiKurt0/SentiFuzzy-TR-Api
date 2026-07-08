---
title: SentiFuzzy-TR
emoji: 🇹🇷
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# SentiFuzzy-TR

Türkçe metinler için **Bulanık Mantık (Fuzzy Logic) + BERTürk** hibrit duygu analizi projesi.

## 1. Bu turda ne değişti?

Proje daha önce şu haldeydi: Zeyrek (morfoloji), skfuzzy (bulanık mantık) ve BERTürk
motorları ayrı ayrı çalışıyordu (`main.py` içinde konsola yazan bir demo olarak),
ama API katmanı (`app/api`, `app/services`, `app/core/config.py`) tamamen boştu,
ve Colab'da eğitilen model API'nin aradığı yere hiç konulmamıştı.

### Bulunan ve düzeltilen sorunlar

| # | Sorun | Düzeltme |
|---|-------|----------|
| 1 | `train.py` modeli `app/models/berturk_sentiment`'e kaydediyordu, API ise `ml_models/berturk_sentiment_pro`'dan okuyordu → model **hiçbir zaman bulunamıyordu**. | Tek kaynak: `app/core/config.py -> Settings.model_dir`. Hem `train.py` hem `ml_model.py` artık aynı yeri kullanıyor. |
| 2 | `dataset_1.csv`'deki yorumların **~%26'sı** metnin başında "Beş Yıldız.", "Bir Yıldız.", "Satıcıya 2 Yıldız." gibi puan ifadeleri barındırıyor; etiketler de zaten bu puandan türetilmiş. Model gerçek duyguyu değil bu yüzeysel örüntüyü öğrenip genelleyemiyor olabilir (ör. "İki Yıldız. **iyi**" → negatif etiketli). | `data/data_preprocessing.py`'a `clean_star_prefix()` eklendi, `final_set.csv` bu temizlikle + duplicate temizliğiyle **yeniden üretildi**. |
| 3 | `Trainer`'da `compute_metrics` yoktu → sadece loss görünüyordu, "iyi çalışmıyor" derken elde accuracy/F1 gibi somut bir ölçüt yoktu. | `accuracy`, `macro_f1` ve sınıf başı F1 eklendi, `load_best_model_at_end=True` ile en iyi checkpoint otomatik seçiliyor. |
| 4 | `max_length=128` — yorumların çoğu bundan uzun, kesiliyordu. | `256`'ya çıkarıldı. |
| 5 | Split rastgele, sınıf oranı garanti değildi; seed yoktu. | `stratify_by_column="label"` + sabit `seed=42`. |
| 6 | API katmanı tamamen boştu (`routes.py`, `config.py`, `model_loder.py` (yazım hatası), `prediction.py`, `app/__init__.py`). | Hepsi dolduruldu (aşağıya bakın). `model_loder.py` → `model_loader.py` olarak yeniden adlandırıldı. |
| 7 | `Dockerfile` bir klasör olarak kalmıştı (boş). | Gerçek bir `Dockerfile` yazıldı. |
| 8 | `requirements.txt` boştu. | Dolduruldu. |
| 9 | `main.py` bir API değil, konsol demo scriptiydi. | Demo `scripts/demo_hybrid.py`'ye taşındı; `main.py` artık gerçek FastAPI giriş noktası. |
| 10 | Sözlük eşleştirmesi **negasyonu ("değil", "yok") anlamıyordu** — "kötü bir gün değil" gibi cümleler yanlış yönde puanlanıyordu; BERT de bu tür dolaylı ifadelerde düşük güvenle yanılabiliyordu. | `prediction.py`'a pencere tabanlı negasyon tespiti eklendi: bir duygu kökünün 3 kelime içinde "değil/yok" varsa katkısı ters çevrilir. Ayrıca negasyon tespit edilip **BERT'in kendi güveni de düşükse** (< 0.80), nihai karar için düzeltilmiş fuzzy sonucu kullanılıyor; BERT yüksek güvenliyse yine BERT'e güveniliyor. |
| 11 | Kullanıcıya yönelik bir arayüz yoktu, sadece `/docs` (Swagger, geliştirici paneli). | `app/static/index.html` — retro "cihaz kontrol paneli" temalı, döner göstergeli bir web arayüzü eklendi; kök adres (`/`) artık bunu sunuyor. |

## 1.1 Bilinen sınırlama: negasyon tam çözülmedi, azaltıldı

`değil/yok` tespiti basit bir pencere kuralı (kelime bazlı, 3 kelimelik
kapsam); gerçek bir bağımlılık ayrıştırıcısı (dependency parser) değil.
"Ürün pahalı değil ama kalitesiz" gibi, negasyonun cümledeki başka bir
kelimeyi değil de yanındaki kelimeyi hedeflediği karmaşık cümlelerde hâlâ
hata payı var. Kalıcı çözüm, `train.py` ile yeniden eğitim yapılırken
veri setine bilinçli olarak negasyonlu örnekler eklemektir (data
augmentation) — BERT bunu gördükçe kalıbı gerçekten öğrenir.

## 2. Mimari

```
main.py                      # uvicorn giriş noktası (FastAPI app)
app/
  __init__.py                 # create_app() factory (CORS, router)
  core/
    config.py                 # merkezi ayarlar (model yolu, eşikler, vs.)
    nlp_zemberek.py            # Zeyrek morfolojik kök çıkarımı
    fuzzy_logic.py              # skfuzzy kural tabanlı motor
    ml_model.py                 # BERTürk pipeline (transformers)
  services/
    model_loader.py             # tüm motorların tekil (singleton) yüklenmesi
    prediction.py                # hibrit karar mantığı (fuzzy + BERT)
  schemas/
    payload.py                   # Pydantic request/response modelleri
  api/
    routes.py                     # /health, /predict endpoint'leri
data/
  data_preprocessing.py         # ham veri -> final_set.csv (+ artık temizlik dahil)
  lexicon_tr.json                # duygu sözlüğü (dışarı çıkarıldı, artık kod içinde gömülü değil)
train.py                       # Colab/GPU üzerinde çalıştırılacak eğitim scripti
scripts/demo_hybrid.py          # eski main.py: konsolda hızlı test için
tests/test_api.py               # zeyrek/transformers stub'larıyla hızlı API testleri
```

**Önemli tasarım kararı:** `app/__init__.py` içindeki ağır import'lar (BERT, Zeyrek)
`create_app()` fonksiyonunun **içine** alındı, dosyanın en üstüne değil. Aksi halde
Python'ın paket `__init__.py` çalıştırma kuralı yüzünden `app.core.config` gibi
masum bir import bile tüm modelleri belleğe yükletirdi.

**Zarif bozulma (graceful degradation):** `ml_models/berturk_sentiment_pro` klasörü
boşsa (henüz model konulmadıysa) API çökmez; `bert_available: false` döner ve
sadece kural tabanlı bulanık mantık sonucunu kullanır. Bunu `tests/test_api.py`'de
hem model-var hem model-yok senaryosu olarak test ettim, ikisi de çalışıyor.

## 3. Modeli Colab'da eğitip buraya taşıma

1. `train.py`'yi Colab'da GPU ile çalıştır (Runtime > Change runtime type > GPU).
2. `data/processed/final_set.csv` içeriğini de Colab'a yükle (veya yeniden üret:
   `python data/data_preprocessing.py`, `data/raw/all_datasets_shuffled.csv` gerekir).
3. Eğitim bitince Colab şu klasörü oluşturur: `ml_models/berturk_sentiment_pro/`
4. O klasörü indirip **projenin kökünde aynı yola** (`ml_models/berturk_sentiment_pro/`)
   koy. `config.json`, `model.safetensors` (veya `pytorch_model.bin`), `vocab.txt`,
   `tokenizer_config.json` gibi dosyalar içinde olmalı.
5. API'yi başlat, `/health` endpoint'i `bert_available: true` dönmeli.

## 4. Çalıştırma

```bash
pip install -r requirements.txt
uvicorn main:app --reload
# veya
python main.py
```

Swagger arayüzü: `http://localhost:8000/docs`

### Örnek istek

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Kargo çok hızlı geldi, harika bir ürün"}'
```

### Docker

```bash
docker build -t sentifuzzy-tr .
docker run -p 7860:7860 sentifuzzy-tr
# tarayıcıda: http://localhost:7860/docs
```

## 5. Testler

```bash
pytest tests/test_api.py -v
```

Bu testler `zeyrek` ve `transformers` paketlerini sahte (stub) modüllerle
değiştirir, böylece GPU/model dosyası olmadan da saniyeler içinde koşar.
Gerçek modelle uçtan uca denemek istersen `ml_models/berturk_sentiment_pro`
doluyken stub'ları kaldırıp doğrudan test edebilirsin.

## 7. Hugging Face Spaces'e Canlıya Alma (Docker SDK)

Bu proje zaten Hugging Face Spaces'in Docker SDK'sının beklediği şekilde
hazırlandı (`Dockerfile` port `7860`'ı dinliyor, root olmayan kullanıcı
kullanıyor; `README.md`'nin en üstünde `sdk: docker` frontmatter'ı var).

### Adımlar

1. **Yeni bir Space oluştur:** https://huggingface.co/spaces/new
   - SDK olarak **Docker** seç (boş şablon yeterli).
   - Public/Private seç, ismini ver (ör. `sentifuzzy-tr`).
   - Hardware: ücretsiz **CPU basic** yeterli (BERT-base inference'ı CPU'da
     rahatça çalışır, GPU şart değil).

2. **Git ile klonla ve dosyaları koy:**
   ```bash
   git clone https://huggingface.co/spaces/KULLANICI_ADIN/sentifuzzy-tr
   cd sentifuzzy-tr
   # Bu projenin tüm içeriğini (data/raw hariç) buraya kopyala
   ```

3. **Model dosyaları büyük olduğu için Git LFS gerekiyor** (BERT checkpoint'i
   genelde 400-450MB civarındadır, GitHub/HF'nin normal dosya limitini aşar):
   ```bash
   git lfs install
   git lfs track "ml_models/berturk_sentiment_pro/*.safetensors"
   git lfs track "ml_models/berturk_sentiment_pro/*.bin"
   git add .gitattributes
   ```

4. **Colab'da eğittiğin modeli** `ml_models/berturk_sentiment_pro/` klasörüne
   koy (bkz. bu README'nin 3. bölümü), sonra:
   ```bash
   git add .
   git commit -m "SentiFuzzy-TR API + model"
   git push
   ```

5. Space sayfasında **"Building"** durumunu izle (Dockerfile build logları
   canlı akar). Bitince **"Running"** olur ve şu adresten erişilebilir olur:
   ```
   https://KULLANICI_ADIN-sentifuzzy-tr.hf.space
   ```
   Swagger arayüzü: `https://KULLANICI_ADIN-sentifuzzy-tr.hf.space/docs`

### Dikkat edilecekler

- **`data/raw/` klasörünü Space'e koyma** — 294MB ham veri API'nin çalışması
  için gerekli değil, sadece build süresini ve repo boyutunu şişirir.
  Sadece `data/lexicon_tr.json` gerekli (zaten Dockerfile bunu kopyalıyor).
- **Ücretsiz CPU basic** katmanında ilk istek biraz yavaş olabilir (model
  belleğe ilk kez yükleniyor); sonraki istekler hızlıdır.
- Space'i uzun süre kullanmazsan ücretsiz katmanda **uykuya geçebilir**
  (sonraki istekte birkaç saniye "uyanma" gecikmesi olur) — bu normal.
- `app/core/config.py` içindeki `cors_allow_origins: ["*"]` şu an her yerden
  erişime izin veriyor; bu bir demo için sorun değil ama gerçek bir üretim
  servisine dönüştürürsen bunu kendi frontend domain'inle sınırlamanı öneririm.

## 6. Sonraki adımlar için öneriler

- `data/lexicon_tr.json`'daki sözlük hâlâ küçük (demo amaçlı ~20 kök). Gerçek
  kullanımda SentiTurkNet gibi hazır bir Türkçe duygu sözlüğüyle genişletmek
  fuzzy-logic tarafının kalitesini ciddi şekilde artırır.
- `dataset_2` (film yorumları) puan eşiğiyle etiketlenmiş; sarkastik yorumlarda
  bu etiketleme gürültülü olabilir. Mümkünse elle etiketlenmiş küçük bir
  doğrulama (validation) alt kümesi ayırıp gerçek performansı orada ölçmek daha
  güvenilir olur.
- `class_weights` şu an eklenmedi çünkü veri seti zaten dengeli (5000/5000/5000);
  kendi veri setini değiştirirsen `train.py`'deki ilgili yorumu aç.
