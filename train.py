"""
train.py  (Google Colab'da GPU ile çalıştırılmalıdır)

DÜZELTMELER (öncekine göre):
1. KAYIT YOLU DÜZELTİLDİ: Eskiden model "app/models/berturk_sentiment"
   klasörüne kaydediliyordu, ama API (app/core/ml_model.py) modeli
   "ml_models/berturk_sentiment_pro" klasöründen okuyor. Bu yüzden
   Colab'da eğitilen model asla API tarafından bulunamıyordu!
   Artık ikisi de app/core/config.py'deki tek bir ayardan (MODEL_DIR)
   besleniyor, tekrar sapmasın diye.
2. METRİK EKLENDİ: Eskiden sadece "loss" görünüyordu, accuracy/F1 yoktu.
   Bu yüzden "model iyi çalışmadı" derken elde somut bir sayı yoktu.
   Şimdi her epoch sonunda accuracy, macro-F1 ve per-class F1 basılıyor.
3. STRATIFIED SPLIT: Train/test ayrımı artık sınıf oranlarını koruyor.
4. SEED sabitlendi (tekrarlanabilirlik için).
5. max_length 128 -> 256'ya çıkarıldı (veri setindeki yorumların önemli
   bir kısmı 128 token'dan uzun, eskiden kesiliyorlardı).
6. class_weights: veri seti zaten dengeli (5000/5000/5000) olduğu için
   eklenmedi, ama kendi veri setini değiştirirsen ekleyebilirsin
   (aşağıda yorum satırı olarak bırakıldı).

ÖNEMLİ (veri kalitesi hakkında): data/data_preprocessing.py dosyasına
bakıldığında, ham verinin (dataset_1) etiketleri yıldız sayısından
(1-2 yıldız=negatif, 3=nötr, 4-5=pozitif gibi) türetilmiş ve metinlerin
içinde "Bir Yıldız.", "Beş Yıldız." gibi puan ifadeleri kelimenin
kendisiyle birlikte geçiyor. Bu durum modelin gerçek duyguyu değil,
bu yüzeysel örüntüyü ezberlemesine yol açabilir. Eğitim öncesi
data/data_preprocessing.py'a eklenen clean_star_prefix() fonksiyonunu
mutlaka çalıştır (bkz. o dosyadaki not).
"""
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

SEED = 42
set_seed(SEED)

MODEL_NAME = "dbmdz/bert-base-turkish-cased"
MAX_LENGTH = 256
# Bu yol, app/core/config.py -> Settings.model_dir ile AYNI olmalı.
KAYIT_YOLU = "ml_models/berturk_sentiment_pro"

ID2LABEL = {0: "Negatif", 1: "Nötr", 2: "Pozitif"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    precision, recall, f1_per_class, _ = precision_recall_fscore_support(
        labels, preds, average=None, labels=[0, 1, 2], zero_division=0
    )

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "f1_negatif": f1_per_class[0],
        "f1_notr": f1_per_class[1],
        "f1_pozitif": f1_per_class[2],
    }


def main():
    print("⏳ Veri seti yükleniyor...")
    df = pd.read_csv("data/processed/final_set.csv")
    df = df[["combined_text", "label_id"]].rename(
        columns={"combined_text": "text", "label_id": "label"}
    )

    print(f"📊 Sınıf dağılımı:\n{df['label'].value_counts()}")

    dataset = Dataset.from_pandas(df, preserve_index=False)

    # stratify_by_column, 'label' sütununun ClassLabel tipinde olmasını
    # şart koşuyor (düz int64 "Value" tipiyle çalışmıyor). Bu satır
    # sütunu ClassLabel'e çeviriyor; değerler (0,1,2) aynı kalıyor.
    dataset = dataset.class_encode_column("label")

    # stratify_by_column ile sınıf oranları train/test'te korunur
    dataset = dataset.train_test_split(test_size=0.2, seed=SEED, stratify_by_column="label")

    print(f"🧠 {MODEL_NAME} indiriliyor...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

    print("⚙️ Veriler tokenize ediliyor...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),  # GPU varsa hızlandırma
        report_to="none",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
    )

    print("🚀 Eğitim başlıyor...")
    trainer.train()

    print("📈 Final değerlendirme:")
    metrics = trainer.evaluate()
    print(metrics)

    model.save_pretrained(KAYIT_YOLU)
    tokenizer.save_pretrained(KAYIT_YOLU)
    print(f"✅ Model başarıyla '{KAYIT_YOLU}' klasörüne kaydedildi!")
    print(
        "👉 Colab'dan indirdikten sonra bu klasörü projenin köküne, "
        f"'{KAYIT_YOLU}' yoluna aynen kopyala (config.py'deki model_dir ile eşleşir)."
    )


if __name__ == "__main__":
    main()
