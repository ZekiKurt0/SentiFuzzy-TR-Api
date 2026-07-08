"""
app/__init__.py

FastAPI uygulamasını oluşturan factory fonksiyonu (create_app).
Bu sayede hem 'main.py' hem de testler (tests/test_api.py) aynı
uygulama örneğini tutarlı şekilde kurabilir.
"""
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_app():
    """
    NOT: Ağır bağımlılıkların (routes -> prediction -> model_loader ->
    zeyrek/transformers) import'u bilinçli olarak burada, fonksiyon
    içinde yapılıyor (üst seviyede değil). Aksi halde sadece
    'from app.core.config import ...' gibi masum bir import bile,
    Python'ın paket __init__.py çalıştırma kuralı yüzünden tüm
    BERT/Zeyrek motorlarını belleğe yükletirdi. Bu da config.py'yi
    veya schemas'ı tek başına test etmeyi/kullanmayı imkansızlaştırırdı.
    """
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
    import os

    from app.core.config import get_settings
    from app.api.routes import router

    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Türkçe metinler için Bulanık Mantık (Fuzzy Logic) + BERTürk "
            "hibrit duygu analizi API'si."
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    _static_dir = os.path.join(os.path.dirname(__file__), "static")

    @app.get("/", tags=["System"], include_in_schema=False)
    def root():
        """Kök adres, kullanıcıya yönelik arayüzü (index.html) sunar."""
        return FileResponse(os.path.join(_static_dir, "index.html"))

    @app.get("/api", tags=["System"])
    def api_info():
        """Programatik istemciler için kısa bir bilgi/keşif endpoint'i."""
        return {
            "message": "SentiFuzzy-TR API çalışıyor.",
            "ui": "/",
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict (POST)",
        }

    return app
