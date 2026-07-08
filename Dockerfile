FROM python:3.11-slim

# Zeyrek/torch bazı sistem paketlerine ihtiyaç duyabiliyor
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces container'ları root olmayan bir kullanıcı (UID 1000)
# bekler. Bunu Space dışında (kendi sunucunda / plain Docker'da) çalıştırırken
# de bir sakıncası yok.
RUN useradd -m -u 1000 user

WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=user app ./app
COPY --chown=user data/lexicon_tr.json ./data/lexicon_tr.json
COPY --chown=user main.py .

# Eğitilmiş model burada bulunmalı (bkz. README):
# ml_models/berturk_sentiment_pro/{config.json, model.safetensors, vocab.txt, ...}
COPY --chown=user ml_models ./ml_models

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Hugging Face Spaces (Docker SDK) varsayılan olarak 7860 portunu bekler
# (README.md üstündeki app_port: 7860 ile eşleşmeli).
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
