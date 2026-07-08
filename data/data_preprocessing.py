"""
data_preprocessing.py

--------------------------------------------------------------------
TARİHÇE (orijinal dosyadan korunan kısım):
Aşağıdaki yorum satırı içindeki blok, dataset_1.csv + dataset_2.csv'nin
nasıl birleştirilip data/raw/all_datasets_shuffled.csv üretildiğini
gösteriyor (bir kereye mahsus çalıştırılıp diskteki dosya üretildi).
Elindeki all_datasets_shuffled.csv zaten var, bu kısmı tekrar
çalıştırmana gerek yok; sadece referans olarak duruyor.
--------------------------------------------------------------------

df1=pd.read_csv('data/raw/dataset_1.csv')
df2=pd.read_csv('data/raw/dataset_2.csv')

df2["point"] = df2["point"].apply(lambda x: float(x.split('/')[0].replace(',', '.')))
def assign_sentiment(rating):
    if rating >= 3.8: return 'positive'
    elif rating >= 2.5: return 'neutral'
    else: return 'negative'
df2['label'] = df2['point'].apply(assign_sentiment)

def assign_sentiment_numeric(rating):
    if rating == 2: return 'positive'
    elif rating == 1: return 'neutral'
    else: return 'negative'
df1['label'] = df1['label'].apply(assign_sentiment_numeric)

first_dataset = df1[['combined_text', 'label']]
second_dataset = pd.read_csv('dataset_2_processed.csv')[['comment', 'label']]
all_datasets = pd.concat([first_dataset, second_dataset.rename(columns={'comment': 'combined_text'})], ignore_index=True)
all_datasets = all_datasets.sample(frac=1).reset_index(drop=True)
all_datasets.to_csv('data/raw/all_datasets_shuffled.csv', index=False)

--------------------------------------------------------------------
YENİ EKLENEN KISIM: clean_star_prefix() + build_final_dataset()
--------------------------------------------------------------------
TESPİT EDİLEN SORUN: dataset_1.csv'deki yorumların ~%26'sı
("Beş Yıldız.", "Üç Yıldız.", "Bir Yıldız.", "Satıcıya 2 Yıldız." gibi)
doğrudan puan bilgisini metnin başında kelime olarak barındırıyor.
Etiketler de zaten bu puandan türetildiği için, model gerçek anlamsal
duyguyu öğrenmek yerine "Beş Yıldız" -> pozitif, "Bir Yıldız" -> negatif
gibi yüzeysel bir kısayol öğrenebiliyor; üstelik bazı örnekler tutarsız
(örn. "İki Yıldız. iyi" -> negatif etiketli ama metin "iyi" diyor).
Bu, modelin "iyi çalışmamasının" en olası nedenlerinden biri.

clean_star_prefix() bu puan ifadesini metnin başından temizler.
build_final_dataset() ise final_set.csv'yi bu temizlik + duplicate
temizliği + dengeli örnekleme ile YENİDEN üretir.
"""
import re

import pandas as pd

# Başta en fazla 3 kelime + "Yıldız." kalıbını yakalar:
# "Beş Yıldız."  / "Satıcıya 2 Yıldız." / "10 numara 5 Yıldız." / "Hakkı Sıfır Yıldız."
_STAR_PREFIX_RE = re.compile(
    r"^(?:[A-Za-zÇĞİÖŞÜçğıöşü0-9]+\s+){0,3}Yıldız\.\s*",
    flags=re.IGNORECASE,
)


def clean_star_prefix(text: str) -> str:
    if not isinstance(text, str):
        return text
    return _STAR_PREFIX_RE.sub("", text, count=1).strip()


def build_final_dataset(
    raw_path: str = "data/raw/all_datasets_shuffled.csv",
    out_path: str = "data/processed/final_set.csv",
    n_per_class: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    df = pd.read_csv(raw_path)

    # 1) Puan-öneki temizliği (etiket sızıntısını azaltmak için)
    before_example = df["combined_text"].iloc[0]
    df["combined_text"] = df["combined_text"].apply(clean_star_prefix)

    # 2) Temizlik sonrası boşalan / çok kısa kalan satırları at
    df = df[df["combined_text"].str.len() >= 3]

    # 3) Tam yinelenen (duplicate) metinleri at
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["combined_text"])
    print(f"🧹 Duplicate temizliği: {before_dedup} -> {len(df)} satır")

    # 4) Etiketleri sayısala çevir
    label_mapping = {"negative": 0, "neutral": 1, "positive": 2}
    df["label_id"] = df["label"].map(label_mapping)
    df = df.dropna(subset=["label_id"])
    df["label_id"] = df["label_id"].astype(int)

    # 5) Sınıf başına dengeli örnekleme
    dev_df = (
        df.groupby("label")
        .sample(n=n_per_class, random_state=seed, replace=False)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

    dev_df.to_csv(out_path, index=False)
    print(f"✅ Geliştirme seti oluşturuldu! Boyut: {dev_df.shape} -> {out_path}")
    print(f"   Örnek metin (temizlik öncesi): {before_example[:80]!r}")
    return dev_df


if __name__ == "__main__":
    build_final_dataset()
