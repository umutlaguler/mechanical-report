"""Yardımcı fonksiyonlar: logging, metin temizleme, geçici dosya yönetimi."""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path

_LOGGER_NAME = "mechanical_spec_analyzer"


def get_logger() -> logging.Logger:
    """Uygulama genelinde tek bir logger döndürür."""
    logger = logging.getLogger(_LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = get_logger()


def clean_text(text: str) -> str:
    """Metni temizler: gereksiz boşlukları ve kontrol karakterlerini sadeleştirir.

    Sayfa işaretleri (--- PAGE X ---) ve satır yapısı korunur.
    """
    if not text:
        return ""

    # Windows satır sonlarını normalize et
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Görünmez/kontrol karakterlerini temizle (newline ve tab hariç)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Satır içindeki ardışık boşlukları teke indir
    text = re.sub(r"[ \t]+", " ", text)

    # 3+ ardışık boş satırı 2'ye indir
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Her satırın baş/son boşluklarını kırp
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()


def save_temp_file(data: bytes, suffix: str) -> Path:
    """Veriyi güvenli bir geçici dosyaya yazar ve yolunu döndürür.

    suffix nokta ile başlamalıdır (ör. ".pdf").
    """
    if not suffix.startswith("."):
        suffix = "." + suffix
    fd, name = tempfile.mkstemp(suffix=suffix, prefix="mecspec_")
    path = Path(name)
    try:
        with open(fd, "wb") as f:
            f.write(data)
    except Exception:
        # Hata olursa fd'yi kapatmayı dene
        try:
            import os

            os.close(fd)
        except OSError:
            pass
        raise
    return path


def cleanup_temp_file(path: Path | None) -> None:
    """Geçici dosyayı sessizce siler."""
    if path is None:
        return
    try:
        if path.exists():
            path.unlink()
    except OSError as exc:  # pragma: no cover - temizlik hatası kritik değil
        logger.warning("Geçici dosya silinemedi: %s (%s)", path, exc)


def truncate(text: str, length: int = 120) -> str:
    """Metni belirtilen uzunlukta keser (UI gösterimi için)."""
    if text is None:
        return ""
    text = text.strip()
    if len(text) <= length:
        return text
    return text[: length - 1].rstrip() + "…"


def safe_filename(name: str, default: str = "rapor") -> str:
    """Dosya adı olarak güvenli bir string üretir."""
    name = (name or "").strip()
    if not name:
        name = default
    name = re.sub(r"[^\w\-. ]", "_", name, flags=re.UNICODE)
    name = re.sub(r"\s+", "_", name)
    return name[:80] or default
