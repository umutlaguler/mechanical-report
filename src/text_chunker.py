"""Metin chunk'lama modülü.

Uzun şartname metinlerini, sayfa sınırlarını ve overlap'i koruyarak
yönetilebilir parçalara böler. Her chunk başına numara ve (varsa) sayfa
bilgisi eklenir; böylece AI çıktısındaki page_or_section alanı zenginleşir.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .config import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP

_PAGE_MARKER_RE = re.compile(r"--- PAGE (\d+) ---")


@dataclass
class TextChunk:
    """Tek bir metin parçası."""

    index: int
    total: int
    text: str
    start_page: int | None = None
    end_page: int | None = None

    def with_header(self) -> str:
        """AI'ya gönderilecek, başlık bilgisi içeren metni döndürür."""
        if self.start_page and self.end_page:
            page_info = (
                f"(Sayfa {self.start_page})"
                if self.start_page == self.end_page
                else f"(Sayfa {self.start_page}-{self.end_page})"
            )
        else:
            page_info = ""
        header = f"[CHUNK {self.index}/{self.total}] {page_info}".strip()
        return f"{header}\n{self.text}"


def _detect_pages(text: str) -> int | None:
    """Metindeki son sayfa işaretinin numarasını döndürür (yoksa None)."""
    pages = _PAGE_MARKER_RE.findall(text)
    return int(pages[-1]) if pages else None


def _page_at(text: str, pos: int) -> int | None:
    """Verilen karakter pozisyonundan önceki en yakın sayfa numarasını bulur."""
    last_page: int | None = None
    for match in _PAGE_MARKER_RE.finditer(text):
        if match.start() > pos:
            break
        last_page = int(match.group(1))
    return last_page


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[TextChunk]:
    """Metni overlap'li chunk'lara böler.

    Mümkün olduğunda paragraf/satır sınırlarında böler. Her chunk için
    başlangıç ve bitiş sayfa numaraları (varsa) hesaplanır.
    """
    text = text or ""
    if chunk_size <= 0:
        chunk_size = DEFAULT_CHUNK_SIZE
    if overlap < 0 or overlap >= chunk_size:
        overlap = min(DEFAULT_OVERLAP, max(0, chunk_size // 4))

    if not text.strip():
        return []

    # Kısa metin tek chunk
    if len(text) <= chunk_size:
        return [
            TextChunk(
                index=1,
                total=1,
                text=text,
                start_page=_page_at(text, 0) or 1 if _detect_pages(text) else None,
                end_page=_detect_pages(text),
            )
        ]

    raw_spans: list[tuple[int, int]] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)

        # Doğal bir kesim noktası ara (paragraf > satır > boşluk)
        if end < length:
            window = text[start:end]
            split_at = _best_split(window)
            if split_at > 0:
                end = start + split_at

        raw_spans.append((start, end))

        if end >= length:
            break
        start = max(end - overlap, start + 1)

    total = len(raw_spans)
    chunks: list[TextChunk] = []
    has_pages = _detect_pages(text) is not None

    for i, (s, e) in enumerate(raw_spans, start=1):
        piece = text[s:e].strip()
        if not piece:
            continue
        chunks.append(
            TextChunk(
                index=i,
                total=total,
                text=piece,
                start_page=_page_at(text, s) if has_pages else None,
                end_page=_page_at(text, e - 1) if has_pages else None,
            )
        )

    # index/total'i temizlenmiş listeye göre yeniden numaralandır
    final_total = len(chunks)
    for new_index, chunk in enumerate(chunks, start=1):
        chunk.index = new_index
        chunk.total = final_total

    return chunks


def _best_split(window: str) -> int:
    """Pencere içinde sondan en iyi kesim noktasını döndürür (0 = bulunamadı)."""
    # En az %60'ından sonra kesmeye çalış ki çok küçük chunk olmasın
    floor = int(len(window) * 0.6)

    for separator in ("\n\n", "\n", ". ", " "):
        idx = window.rfind(separator)
        if idx >= floor:
            return idx + len(separator)
    return 0
