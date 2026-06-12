"""Uygulama yapılandırması ve sabitler.

API anahtarı .env üzerinden okunur, asla kod içine yazılmaz.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# .env dosyasını proje kökünden yükle
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SAMPLE_DIR = PROJECT_ROOT / "sample_files"

# Desteklenen dosya tipleri (uzantı -> tip)
SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".txt": "txt",
    ".md": "md",
    ".html": "html",
    ".htm": "html",
}

# Streamlit file_uploader için kabul edilen uzantılar
UPLOADER_TYPES = ["pdf", "docx", "txt", "md", "html", "htm"]

# Kullanılabilir modeller (UI seçimi)
AVAILABLE_MODELS = ["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini"]

# Chunking varsayılanları
DEFAULT_CHUNK_SIZE = 11000
DEFAULT_OVERLAP = 1000
MIN_CHUNK_SIZE = 4000
MAX_CHUNK_SIZE = 16000

# Confidence varsayılanı
DEFAULT_MIN_CONFIDENCE = 0.0
LOW_CONFIDENCE_THRESHOLD = 0.5

# Mekanik analiz için kategori listesi (report_builder bunları normalize eder)
MECHANICAL_CATEGORIES = [
    "Genel Mekanik Kapsam",
    "Tank / Kazan / Kapak",
    "Dalga Duvar / Radyatör / Soğutma",
    "Sac Kalınlığı ve Malzeme",
    "Kaynak ve Sızdırmazlık",
    "Conta / Flanş / Bağlantı Elemanları",
    "Basınç / Vakum / Emniyet Donanımları",
    "Yağ Doldurma / Boşaltma / Vanalar",
    "Boya / Kaplama / Korozyon",
    "Taşıma / Kaldırma / Tekerlek / Şasi",
    "Buşing / Kablo Kutusu / Mahfaza",
    "Ölçüler / Ağırlıklar / Toleranslar",
    "Mekanik Testler",
    "Ambalaj / Sevkiyat / Montaj",
    "Eksik veya Belirsiz Noktalar",
]


@dataclass
class AppConfig:
    """Çalışma zamanı analiz ayarları."""

    model: str = "gpt-4.1-mini"
    chunk_size: int = DEFAULT_CHUNK_SIZE
    overlap: int = DEFAULT_OVERLAP
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    use_ai_summary: bool = True
    api_key: str | None = field(default=None)


def get_api_key() -> str | None:
    """Ortamdan OpenAI API anahtarını döndürür (yoksa None)."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    return key or None


def get_default_model() -> str:
    """Ortamdan varsayılan modeli döndürür."""
    model = os.getenv("OPENAI_MODEL", "").strip()
    return model if model in AVAILABLE_MODELS else "gpt-4.1-mini"
