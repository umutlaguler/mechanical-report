"""Rapor oluşturucu.

Çıkarılan mekanik maddeleri temizler (deduplicate), kategorilere göre gruplar,
düşük güvenli maddeleri işaretler ve nihai FinalMechanicalReport üretir.
Yönetici özeti için opsiyonel olarak OpenAI çağrısı yapılır; AI kullanılamazsa
deterministik bir özet üretilir.
"""

from __future__ import annotations

import json
import re

from .ai_extractor import FinalMechanicalReport, MechanicalItem
from .config import LOW_CONFIDENCE_THRESHOLD, MECHANICAL_CATEGORIES
from .utils import logger, truncate

# Kategori normalizasyonu için anahtar kelime eşlemesi
_CATEGORY_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("Tank / Kazan / Kapak", ("tank", "kazan", "kapak", "gövde", "hermetik")),
    (
        "Dalga Duvar / Radyatör / Soğutma",
        ("dalga duvar", "corrugated", "radyatör", "soğut", "fan", "soğutma yüzey"),
    ),
    (
        "Sac Kalınlığı ve Malzeme",
        ("sac", "kalınlık", "malzeme", "çelik", "alüminyum", "paslanmaz", "karkas", "konstrüksiyon"),
    ),
    ("Kaynak ve Sızdırmazlık", ("kaynak", "sızdırmaz", "weld")),
    (
        "Conta / Flanş / Bağlantı Elemanları",
        ("conta", "flanş", "cıvata", "somun", "bağlantı eleman", "gasket"),
    ),
    (
        "Basınç / Vakum / Emniyet Donanımları",
        ("basınç", "vakum", "patlama", "yırtılma", "emniyet valf", "pressure"),
    ),
    (
        "Yağ Doldurma / Boşaltma / Vanalar",
        ("yağ", "vana", "doldurma", "boşaltma", "oil", "seviye gösterge"),
    ),
    (
        "Boya / Kaplama / Korozyon",
        ("boya", "kaplama", "korozyon", "kumlama", "galvaniz", "ral", "mikron", "yüzey hazırl"),
    ),
    (
        "Taşıma / Kaldırma / Tekerlek / Şasi",
        ("taşıma", "kaldırma", "halka", "tekerlek", "şasi", "ray", "kaldırma halkası"),
    ),
    (
        "Buşing / Kablo Kutusu / Mahfaza",
        ("buşing", "kablo kutusu", "mahfaza", "muhafaza", "izolatör", "bara", "topraklama", "ip "),
    ),
    (
        "Ölçüler / Ağırlıklar / Toleranslar",
        ("ölçü", "boyut", "ağırlık", "tolerans", "maksimum", "dış boyut"),
    ),
    ("Mekanik Testler", ("test", "deney", "muayene")),
    ("Ambalaj / Sevkiyat / Montaj", ("ambalaj", "sevkiyat", "montaj", "paket", "nakliye")),
]


def _normalize_category(raw_category: str, item: MechanicalItem) -> str:
    """AI'nın verdiği kategoriyi standart kategori listesine eşler."""
    text = f"{raw_category} {item.title} {item.requirement}".lower()

    # Önce ham kategori bilinen bir kategoriyle örtüşüyor mu?
    for known in MECHANICAL_CATEGORIES:
        if raw_category and raw_category.strip().lower() == known.lower():
            return known

    for category, keywords in _CATEGORY_KEYWORDS:
        if any(kw in text for kw in keywords):
            return category

    return "Genel Mekanik Kapsam"


def _dedup_key(item: MechanicalItem) -> str:
    """Bir maddenin benzersizlik anahtarını üretir."""
    norm = re.sub(r"\s+", " ", f"{item.title} {item.requirement}".lower()).strip()
    return norm[:160]


def deduplicate_items(items: list[MechanicalItem]) -> list[MechanicalItem]:
    """Benzer maddeleri birleştirir; en yüksek confidence'lı olanı tutar."""
    best: dict[str, MechanicalItem] = {}
    for item in items:
        key = _dedup_key(item)
        if not key:
            continue
        existing = best.get(key)
        if existing is None or item.confidence > existing.confidence:
            best[key] = item
    return list(best.values())


def process_items(
    items: list[MechanicalItem], min_confidence: float = 0.0
) -> list[MechanicalItem]:
    """Maddeleri temizler: kategori normalize eder, dedup yapar, filtreler, sıralar."""
    cleaned: list[MechanicalItem] = []
    for item in items:
        item.category = _normalize_category(item.category, item)
        cleaned.append(item)

    cleaned = deduplicate_items(cleaned)

    if min_confidence > 0:
        cleaned = [it for it in cleaned if it.confidence >= min_confidence]

    # Kategori sırası + confidence'a göre sırala
    order = {cat: i for i, cat in enumerate(MECHANICAL_CATEGORIES)}
    cleaned.sort(
        key=lambda it: (order.get(it.category, 99), -it.confidence, it.title.lower())
    )
    return cleaned


def group_by_category(
    items: list[MechanicalItem],
) -> dict[str, list[MechanicalItem]]:
    """Maddeleri kategoriye göre gruplar (kategori sırası korunur)."""
    grouped: dict[str, list[MechanicalItem]] = {}
    for cat in MECHANICAL_CATEGORIES:
        members = [it for it in items if it.category == cat]
        if members:
            grouped[cat] = members
    # Listede olmayan beklenmedik kategoriler
    for item in items:
        if item.category not in grouped and item.category not in MECHANICAL_CATEGORIES:
            grouped.setdefault(item.category, []).append(item)
    return grouped


def low_confidence_items(
    items: list[MechanicalItem], threshold: float = LOW_CONFIDENCE_THRESHOLD
) -> list[MechanicalItem]:
    """Eşik altındaki düşük güvenli maddeleri döndürür."""
    return [it for it in items if it.confidence < threshold]


def average_confidence(items: list[MechanicalItem]) -> float:
    """Ortalama güven skoru."""
    if not items:
        return 0.0
    return sum(it.confidence for it in items) / len(items)


# --------------------------------------------------------------------------- #
# Nihai rapor üretimi
# --------------------------------------------------------------------------- #


def _build_grouped_details_text(
    grouped: dict[str, list[MechanicalItem]],
) -> dict[str, list[str]]:
    """grouped_details alanı için metin satırları üretir."""
    details: dict[str, list[str]] = {}
    for category, members in grouped.items():
        lines: list[str] = []
        for it in members:
            piece = it.requirement.strip()
            if it.value_or_limit:
                piece += f" — {it.value_or_limit}"
            if it.related_standard:
                piece += f" ({it.related_standard})"
            lines.append(piece)
        details[category] = lines
    return details


def _build_checklist(items: list[MechanicalItem]) -> list[str]:
    """Maddelerden kontrol listesi üretir."""
    checklist: list[str] = []
    seen: set[str] = set()
    for it in items:
        base = it.title.strip() or truncate(it.requirement, 60)
        line = base
        if it.value_or_limit:
            line += f": {it.value_or_limit}"
        key = line.lower()
        if key not in seen:
            seen.add(key)
            checklist.append(line)
    return checklist


def _deterministic_summary(
    items: list[MechanicalItem], grouped: dict[str, list[MechanicalItem]]
) -> str:
    """AI olmadan deterministik yönetici özeti."""
    if not items:
        return "Belgede mekanik mühendisliği ilgilendiren belirgin bir madde tespit edilmedi."
    top_categories = sorted(grouped.items(), key=lambda kv: -len(kv[1]))[:5]
    cats_text = ", ".join(f"{cat} ({len(m)})" for cat, m in top_categories)
    return (
        f"Belgeden toplam {len(items)} mekanik madde, {len(grouped)} kategoride çıkarıldı. "
        f"En yoğun kategoriler: {cats_text}. "
        "Maddeler tank/kazan konstrüksiyonu, sac kalınlığı ve malzeme, kaynak ve sızdırmazlık, "
        "boya/galvaniz, taşıma donanımları, ölçü/ağırlık ve mekanik testler başlıklarında "
        "yoğunlaşmaktadır."
    )


def _ai_summary(
    items: list[MechanicalItem],
    grouped: dict[str, list[MechanicalItem]],
    document_title: str,
    api_key: str,
    model: str,
) -> tuple[str, list[str]]:
    """OpenAI ile yönetici özeti ve eksik/belirsiz noktalar üretir."""
    from openai import OpenAI

    digest_lines: list[str] = []
    for cat, members in grouped.items():
        digest_lines.append(f"## {cat}")
        for it in members[:12]:
            val = f" [{it.value_or_limit}]" if it.value_or_limit else ""
            digest_lines.append(f"- {it.title}: {truncate(it.requirement, 140)}{val}")
    digest = "\n".join(digest_lines)[:12000]

    system = (
        "Sen kıdemli bir mekanik mühendissin. Sana bir trafo/ekipman teknik şartnamesinden "
        "çıkarılmış mekanik maddelerin özeti veriliyor. Yöneticiye yönelik kısa ve teknik bir "
        "özet (executive_summary) ve şartnamede eksik veya belirsiz kalan mekanik noktaların "
        "listesini (missing_or_unclear_points) üret. Sadece JSON döndür: "
        '{"executive_summary": str, "missing_or_unclear_points": [str]}'
    )
    user = f"Belge başlığı: {document_title}\n\nMekanik maddeler:\n{digest}"

    try:
        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        data = json.loads(completion.choices[0].message.content or "{}")
        summary = str(data.get("executive_summary", "")).strip()
        missing = [str(x).strip() for x in data.get("missing_or_unclear_points", []) if str(x).strip()]
        if not summary:
            summary = _deterministic_summary(items, grouped)
        return summary, missing
    except Exception as exc:  # noqa: BLE001 - özet AI'sı kritik değil
        logger.warning("AI özeti üretilemedi, deterministik özete geçiliyor: %s", exc)
        return _deterministic_summary(items, grouped), []


def guess_document_title(document_text: str, fallback: str = "Teknik Şartname") -> str:
    """Belge metninden tahmini bir başlık çıkarır."""
    if not document_text:
        return fallback
    for raw_line in document_text.split("\n"):
        line = raw_line.strip()
        if line.startswith("--- PAGE"):
            continue
        if len(line) >= 8 and not line.lower().startswith("chunk"):
            return truncate(line, 120)
    return fallback


def build_report(
    items: list[MechanicalItem],
    document_text: str,
    min_confidence: float = 0.0,
    use_ai_summary: bool = False,
    api_key: str | None = None,
    model: str = "gpt-4.1-mini",
    document_title: str | None = None,
) -> tuple[FinalMechanicalReport, list[MechanicalItem]]:
    """Maddelerden nihai raporu üretir.

    Returns:
        (FinalMechanicalReport, işlenmiş madde listesi) ikilisi.
    """
    processed = process_items(items, min_confidence=min_confidence)
    grouped = group_by_category(processed)

    title = document_title or guess_document_title(document_text)
    grouped_details = _build_grouped_details_text(grouped)
    checklist = _build_checklist(processed)

    missing_points: list[str] = []
    if use_ai_summary and api_key and processed:
        summary, missing_points = _ai_summary(
            processed, grouped, title, api_key, model
        )
    else:
        summary = _deterministic_summary(processed, grouped)

    # Düşük güvenli maddeleri belirsizlik notu olarak ekle
    low = low_confidence_items(processed)
    if low:
        missing_points.append(
            f"{len(low)} madde düşük güven skoruyla işaretlendi; "
            "şartnameden manuel doğrulama önerilir."
        )

    report = FinalMechanicalReport(
        document_title=title,
        executive_summary=summary,
        grouped_details=grouped_details,
        checklist=checklist,
        missing_or_unclear_points=missing_points,
    )
    return report, processed
