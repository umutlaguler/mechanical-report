"""Rapor oluşturucu (disiplin-parametrik).

Çıkarılan maddeleri disipline göre temizler (deduplicate), kategorilere göre
gruplar, düşük güvenli maddeleri işaretler ve disiplin başına FinalReport üretir.
Yönetici özeti için opsiyonel olarak OpenAI çağrısı yapılır; AI kullanılamazsa
deterministik bir özet üretilir.
"""

from __future__ import annotations

import json
import re

from .ai_extractor import FinalReport, SpecItem
from .config import (
    AI_TEMPERATURE,
    LOW_CONFIDENCE_THRESHOLD,
    categories_for,
    discipline_label,
)
from .utils import logger, truncate

# Kategori normalizasyonu için anahtar kelime eşlemesi (disiplin -> liste)
_MECHANICAL_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
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
        ("buşing", "bushing", "kablo kutusu", "mahfaza", "muhafaza", "izolatör", "bara", "topraklama", "ip "),
    ),
    (
        "Ölçüler / Ağırlıklar / Toleranslar",
        ("ölçü", "boyut", "ağırlık", "tolerans", "maksimum", "dış boyut"),
    ),
    ("Mekanik Testler", ("test", "deney", "muayene")),
    ("Ambalaj / Sevkiyat / Montaj", ("ambalaj", "sevkiyat", "montaj", "paket", "nakliye")),
]

_ELECTRICAL_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    (
        "Güç / Gerilim Sınıfı / Frekans",
        ("kva", "mva", "gerilim", "voltage", "frekans", "frequency", "anma güc", "kv ", "güç"),
    ),
    (
        "Sargı / Bağlantı Grubu / Vektör",
        ("sargı", "winding", "bağlantı grubu", "vektör", "vector", "yıldız", "üçgen", "dyn", "ynd"),
    ),
    (
        "OLTC / Kademe Değiştirici",
        ("oltc", "detc", "kademe", "tap changer", "tap-changer", "kademe değiştir"),
    ),
    (
        "Bushing / İzolatör Bağlantıları",
        ("bushing", "buşing", "izolatör", "insulator", "porselen", "epoksi", "composite"),
    ),
    (
        "Kablo Kutusu / Busbar / Terminaller",
        ("kablo kutusu", "cable box", "busbar", "bara", "terminal", "nötr çıkış", "plug-in", "connector"),
    ),
    (
        "Koruma / Topraklama / Yıldırımlık",
        ("koruma", "röle", "buchholz", "topraklama", "earthing", "grounding", "parafudr", "yıldırımlık", "arrester"),
    ),
    (
        "Yalıtım / İzolasyon Seviyesi (BIL)",
        ("bil", "yalıtım seviye", "izolasyon", "li ", "darbe gerilim", "test gerilim", "li/ac"),
    ),
    (
        "Yardımcı Donanım / Sensör / İzleme",
        ("sensör", "termometre", "sıcaklık", "monitoring", "izleme", "gösterge", "transduser", "ct ", "vt "),
    ),
    (
        "Kayıplar / Empedans / Verimlilik",
        ("kayıp", "loss", "empedans", "impedance", "uk", "verimlilik", "efficiency", "ısınma"),
    ),
    ("Elektriksel Testler", ("test", "deney", "rutin test", "tip test", "routine", "type test")),
]

_KEYWORDS_BY_DISCIPLINE = {
    "mekanik": _MECHANICAL_KEYWORDS,
    "elektrik": _ELECTRICAL_KEYWORDS,
}


def _default_category(discipline: str) -> str:
    """Disiplinin 'genel' fallback kategorisi."""
    return categories_for(discipline)[0]


def _normalize_category(raw_category: str, item: SpecItem, discipline: str) -> str:
    """AI'nın verdiği kategoriyi ilgili disiplinin standart listesine eşler."""
    known_categories = categories_for(discipline)
    text = f"{raw_category} {item.title} {item.requirement}".lower()

    # Önce ham kategori bilinen bir kategoriyle örtüşüyor mu?
    for known in known_categories:
        if raw_category and raw_category.strip().lower() == known.lower():
            return known

    for category, keywords in _KEYWORDS_BY_DISCIPLINE.get(discipline, []):
        if any(kw in text for kw in keywords):
            return category

    return _default_category(discipline)


def _dedup_key(item: SpecItem) -> str:
    """Bir maddenin benzersizlik anahtarını üretir."""
    norm = re.sub(r"\s+", " ", f"{item.title} {item.requirement}".lower()).strip()
    return norm[:160]


def deduplicate_items(items: list[SpecItem]) -> list[SpecItem]:
    """Benzer maddeleri birleştirir; en yüksek confidence'lı olanı tutar."""
    best: dict[str, SpecItem] = {}
    for item in items:
        key = _dedup_key(item)
        if not key:
            continue
        existing = best.get(key)
        if existing is None or item.confidence > existing.confidence:
            best[key] = item
    return list(best.values())


def process_items(
    items: list[SpecItem], discipline: str, min_confidence: float = 0.0
) -> list[SpecItem]:
    """Maddeleri temizler: kategori normalize eder, dedup yapar, filtreler, sıralar."""
    cleaned: list[SpecItem] = []
    for item in items:
        item.category = _normalize_category(item.category, item, discipline)
        cleaned.append(item)

    cleaned = deduplicate_items(cleaned)

    if min_confidence > 0:
        cleaned = [it for it in cleaned if it.confidence >= min_confidence]

    # Kategori sırası + confidence'a göre sırala
    order = {cat: i for i, cat in enumerate(categories_for(discipline))}
    cleaned.sort(
        key=lambda it: (order.get(it.category, 99), -it.confidence, it.title.lower())
    )
    return cleaned


def group_by_category(
    items: list[SpecItem], discipline: str
) -> dict[str, list[SpecItem]]:
    """Maddeleri kategoriye göre gruplar (kategori sırası korunur)."""
    known_categories = categories_for(discipline)
    grouped: dict[str, list[SpecItem]] = {}
    for cat in known_categories:
        members = [it for it in items if it.category == cat]
        if members:
            grouped[cat] = members
    # Listede olmayan beklenmedik kategoriler
    for item in items:
        if item.category not in grouped and item.category not in known_categories:
            grouped.setdefault(item.category, []).append(item)
    return grouped


def low_confidence_items(
    items: list[SpecItem], threshold: float = LOW_CONFIDENCE_THRESHOLD
) -> list[SpecItem]:
    """Eşik altındaki düşük güvenli maddeleri döndürür."""
    return [it for it in items if it.confidence < threshold]


def average_confidence(items: list[SpecItem]) -> float:
    """Ortalama güven skoru."""
    if not items:
        return 0.0
    return sum(it.confidence for it in items) / len(items)


# --------------------------------------------------------------------------- #
# Nihai rapor üretimi
# --------------------------------------------------------------------------- #


def _build_grouped_details_text(
    grouped: dict[str, list[SpecItem]],
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


def _build_checklist(items: list[SpecItem]) -> list[str]:
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
    items: list[SpecItem], grouped: dict[str, list[SpecItem]], discipline: str
) -> str:
    """AI olmadan deterministik yönetici özeti."""
    label = discipline_label(discipline)
    if not items:
        return f"Belgede {label.lower()} mühendisliğini ilgilendiren belirgin bir madde tespit edilmedi."
    top_categories = sorted(grouped.items(), key=lambda kv: -len(kv[1]))[:5]
    cats_text = ", ".join(f"{cat} ({len(m)})" for cat, m in top_categories)
    return (
        f"Belgeden toplam {len(items)} {label.lower()} madde, {len(grouped)} kategoride çıkarıldı. "
        f"En yoğun kategoriler: {cats_text}."
    )


def _ai_summary(
    items: list[SpecItem],
    grouped: dict[str, list[SpecItem]],
    document_title: str,
    discipline: str,
    api_key: str,
    model: str,
) -> tuple[str, list[str]]:
    """OpenAI ile yönetici özeti ve eksik/belirsiz noktalar üretir."""
    from openai import OpenAI

    label = discipline_label(discipline)
    digest_lines: list[str] = []
    for cat, members in grouped.items():
        digest_lines.append(f"## {cat}")
        for it in members[:12]:
            val = f" [{it.value_or_limit}]" if it.value_or_limit else ""
            digest_lines.append(f"- {it.title}: {truncate(it.requirement, 140)}{val}")
    digest = "\n".join(digest_lines)[:12000]

    system = (
        f"Sen kıdemli bir {label.lower()} mühendisisin. Sana bir trafo/ekipman teknik "
        f"şartnamesinden çıkarılmış {label.lower()} maddelerin özeti veriliyor. Yöneticiye "
        "yönelik kısa ve teknik bir özet (executive_summary) ve şartnamede eksik veya belirsiz "
        f"kalan {label.lower()} noktaların listesini (missing_or_unclear_points) üret. "
        "Çıktının tamamı TÜRKÇE olsun. Sadece JSON döndür: "
        '{"executive_summary": str, "missing_or_unclear_points": [str]}'
    )
    user = f"Belge başlığı: {document_title}\n\n{label} maddeler:\n{digest}"

    try:
        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=max(AI_TEMPERATURE, 0.2),
        )
        data = json.loads(completion.choices[0].message.content or "{}")
        summary = str(data.get("executive_summary", "")).strip()
        missing = [str(x).strip() for x in data.get("missing_or_unclear_points", []) if str(x).strip()]
        if not summary:
            summary = _deterministic_summary(items, grouped, discipline)
        return summary, missing
    except Exception as exc:  # noqa: BLE001 - özet AI'sı kritik değil
        logger.warning("AI özeti üretilemedi, deterministik özete geçiliyor: %s", exc)
        return _deterministic_summary(items, grouped, discipline), []


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
    items: list[SpecItem],
    discipline: str,
    document_text: str,
    min_confidence: float = 0.0,
    use_ai_summary: bool = False,
    api_key: str | None = None,
    model: str = "gpt-4.1-mini",
    document_title: str | None = None,
) -> tuple[FinalReport, list[SpecItem]]:
    """Tek bir disiplin için nihai raporu üretir.

    Returns:
        (FinalReport, işlenmiş madde listesi) ikilisi.
    """
    # Yalnızca bu disipline ait maddeler
    discipline_items = [it for it in items if (it.discipline or "mekanik") == discipline]

    processed = process_items(discipline_items, discipline, min_confidence=min_confidence)
    grouped = group_by_category(processed, discipline)

    title = document_title or guess_document_title(document_text)
    grouped_details = _build_grouped_details_text(grouped)
    checklist = _build_checklist(processed)

    missing_points: list[str] = []
    if use_ai_summary and api_key and processed:
        summary, missing_points = _ai_summary(
            processed, grouped, title, discipline, api_key, model
        )
    else:
        summary = _deterministic_summary(processed, grouped, discipline)

    # Düşük güvenli maddeleri belirsizlik notu olarak ekle
    low = low_confidence_items(processed)
    if low:
        missing_points.append(
            f"{len(low)} madde düşük güven skoruyla işaretlendi; "
            "şartnameden manuel doğrulama önerilir."
        )

    report = FinalReport(
        discipline=discipline,
        document_title=title,
        executive_summary=summary,
        grouped_details=grouped_details,
        checklist=checklist,
        missing_or_unclear_points=missing_points,
    )
    return report, processed


def build_reports(
    items: list[SpecItem],
    disciplines: list[str],
    document_text: str,
    min_confidence: float = 0.0,
    use_ai_summary: bool = False,
    api_key: str | None = None,
    model: str = "gpt-4.1-mini",
    document_title: str | None = None,
) -> dict[str, tuple[FinalReport, list[SpecItem]]]:
    """Her disiplin için ayrı rapor üretir.

    Returns:
        {disiplin: (FinalReport, işlenmiş maddeler)} sözlüğü (disiplin sırası korunur).
    """
    results: dict[str, tuple[FinalReport, list[SpecItem]]] = {}
    for discipline in disciplines:
        results[discipline] = build_report(
            items,
            discipline=discipline,
            document_text=document_text,
            min_confidence=min_confidence,
            use_ai_summary=use_ai_summary,
            api_key=api_key,
            model=model,
            document_title=document_title,
        )
    return results
