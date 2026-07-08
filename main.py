"""
main.py

SentiFuzzy-TR API'sinin giriş noktası.

NOT: Bu dosyanın eski hali (konsol çıktısı basan hibrit-analiz demo
scripti) kaldırılmadı, `scripts/demo_hybrid.py` içine taşındı. Onu
API'yi ayağa kaldırmadan, tek seferlik hızlı bir test yapmak için
şu şekilde çalıştırabilirsin:

    python scripts/demo_hybrid.py

Bu dosya (main.py) ise gerçek HTTP API'sini başlatır:

    uvicorn main:app --reload --host 0.0.0.0 --port 8000

veya doğrudan:

    python main.py
"""
from app import create_app

app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
