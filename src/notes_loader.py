"""Model için domain notlarını Excel'den okuyan modül.

`model için notlar.xlsx` dosyası, mekanik/elektrik kullanıcılardan gelen
geri bildirimleri ekipman bazlı tutar. Her satır bir ekipmanı, eş anlamlılarını
ve "neyi kontrol et / nasıl raporla" talimatını içerir. Bu notlar, AI çıkarımı
sırasında sistem prompt'una enjekte edilir; böylece model bu ekipmanları
şartnamede arar ve nota göre değerlendirir.

Excel yapısı esnek okunur (geleceğe dönük):
- Her sayfa gezilir; başlık satırı ("Ekipman" + "Eş Anlamlılar"/"Not" içeren) bulunur.
- Disiplin tespiti: (a) varsa "Disiplin"/"Discipline" kolonu; yoksa (b) sayfa adı
  veya üstteki başlık hücresi ("MEKANİK..."→mekanik, "ELEKTRİK..."→elektrik);
  bulunamazsa fallback "mekanik".
- "KAPAK" gibi tek hücreli satırlar bölüm (section) başlığı sayılır.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .config import DISCIPLINES, NOTES_XLSX_PATH, discipline_label
from .utils import logger


@dataclass
class EquipmentNote:
    """Tek bir ekipman için domain notu."""

    discipline: str
    section: str
    equipment: str
    synonyms: list[str] = field(default_factory=list)
    note: str = ""


# --------------------------------------------------------------------------- #
# Yardımcılar
# --------------------------------------------------------------------------- #
def _cell(value) -> str:
    """Hücre değerini temiz string'e çevirir."""
    if value is None:
        return ""
    return str(value).strip()


def _nonempty(row: tuple) -> list[str]:
    """Satırdaki boş olmayan hücre metinlerini döndürür."""
    return [c for c in (_cell(v) for v in row) if c]


def _detect_discipline(text: str) -> str | None:
    """Bir metinden disiplini tahmin eder (mekanik/elektrik)."""
    low = text.lower()
    if "elektr" in low:  # elektrik / electrical
        return "elektrik"
    if "mekan" in low or "mechanic" in low:  # mekanik / mechanical
        return "mekanik"
    return None


def _split_synonyms(raw: str) -> list[str]:
    """';' (veya ',') ile ayrılmış eş anlamlıları listeye çevirir."""
    if not raw:
        return []
    parts: list[str] = []
    for chunk in raw.replace("\n", ";").split(";"):
        chunk = chunk.strip().strip(",").strip()
        if chunk:
            parts.append(chunk)
    # tekilleştir, sırayı koru
    seen: set[str] = set()
    unique: list[str] = []
    for p in parts:
        key = p.lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def _find_header(row: tuple) -> dict[str, int] | None:
    """Bir satır başlık satırı mı? Öyleyse kolon -> index eşlemesi döndürür."""
    mapping: dict[str, int] = {}
    for idx, value in enumerate(row):
        text = _cell(value).lower()
        if not text:
            continue
        if "ekipman" in text or "equipment" in text:
            mapping["equipment"] = idx
        elif "anlam" in text or "synonym" in text or "eş" in text:
            mapping["synonyms"] = idx
        elif text == "not" or "not" == text.split()[0] or "note" in text:
            mapping["note"] = idx
        elif "disiplin" in text or "discipline" in text:
            mapping["discipline"] = idx
    if "equipment" in mapping and ("synonyms" in mapping or "note" in mapping):
        return mapping
    return None


# --------------------------------------------------------------------------- #
# Ana yükleyici
# --------------------------------------------------------------------------- #
def _load_from_path(path: Path) -> list[EquipmentNote]:
    """Excel'i okuyup EquipmentNote listesi üretir."""
    try:
        import openpyxl
    except ImportError as exc:  # pragma: no cover
        logger.warning("openpyxl kurulu değil, notlar okunamadı: %s", exc)
        return []

    if not path.exists():
        logger.warning("Not dosyası bulunamadı: %s", path)
        return []

    try:
        workbook = openpyxl.load_workbook(path, data_only=True, read_only=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Not dosyası okunamadı (%s): %s", path, exc)
        return []

    notes: list[EquipmentNote] = []

    for worksheet in workbook.worksheets:
        sheet_discipline = _detect_discipline(worksheet.title)
        current_discipline = sheet_discipline
        current_section = ""
        header: dict[str, int] | None = None

        for row in worksheet.iter_rows(values_only=True):
            cells = _nonempty(row)
            if not cells:
                continue

            # Başlık satırı?
            maybe_header = _find_header(row)
            if maybe_header is not None:
                header = maybe_header
                continue

            # Henüz başlık görülmediyse: bu satır disiplin başlığı ya da section olabilir
            if header is None:
                if len(cells) == 1:
                    disc = _detect_discipline(cells[0])
                    if disc is not None:
                        current_discipline = disc
                    else:
                        current_section = cells[0]
                continue

            # Tek hücreli satır = bölüm (section) başlığı
            if len(cells) == 1:
                disc = _detect_discipline(cells[0])
                if disc is not None:
                    current_discipline = disc
                else:
                    current_section = cells[0]
                continue

            # Veri satırı
            equipment = _cell(row[header["equipment"]]) if header.get("equipment") is not None else ""
            if not equipment:
                continue

            synonyms = (
                _split_synonyms(_cell(row[header["synonyms"]]))
                if header.get("synonyms") is not None
                else []
            )
            note = _cell(row[header["note"]]) if header.get("note") is not None else ""

            row_discipline = current_discipline
            if header.get("discipline") is not None:
                explicit = _detect_discipline(_cell(row[header["discipline"]]))
                if explicit is not None:
                    row_discipline = explicit

            notes.append(
                EquipmentNote(
                    discipline=row_discipline or "mekanik",
                    section=current_section,
                    equipment=equipment,
                    synonyms=synonyms,
                    note=note,
                )
            )

    workbook.close()
    logger.info("Domain notları yüklendi: %d ekipman (%s)", len(notes), path.name)
    return notes


@lru_cache(maxsize=1)
def load_equipment_notes() -> tuple[EquipmentNote, ...]:
    """Varsayılan Excel'den notları yükler (cache'li). Tuple döner ki hashable olsun."""
    return tuple(_load_from_path(NOTES_XLSX_PATH))


def notes_for_discipline(discipline: str) -> list[EquipmentNote]:
    """Belirli bir disipline ait notları döndürür."""
    return [n for n in load_equipment_notes() if n.discipline == discipline]


def build_notes_prompt_block(discipline: str) -> str:
    """Disiplinin notlarını sistem prompt'una eklenecek metne çevirir.

    Not yoksa boş string döner (prompt'a hiçbir şey eklenmez).
    """
    notes = notes_for_discipline(discipline)
    if not notes:
        return ""

    label = discipline_label(discipline)
    lines: list[str] = [
        f"{label.upper()} DOMAIN KONTROL NOTLARI (kurum içi geri bildirimlerden):",
        "Aşağıdaki ekipmanları şartnamede (eş anlamlılarıyla birlikte, hangi dilde "
        "olursa olsun) ara. Bulursan, verilen NOT talimatına göre değerlendir ve "
        "ilgili maddeleri raporla. Talimatta belirtilen kontrol noktalarını "
        "(tip, standart, ölçü, yön, IP sınıfı vb.) ayrıca not et.",
        "",
    ]

    # Bölüm (section) bazlı grupla, sırayı koru
    current_section = object()  # sentinel
    for note in notes:
        if note.section != current_section:
            current_section = note.section
            if current_section:
                lines.append(f"[{current_section}]")
        syn = ""
        if note.synonyms:
            syn = " (eş anlamlılar: " + "; ".join(note.synonyms) + ")"
        if note.note:
            lines.append(f"- {note.equipment}{syn}: {note.note}")
        else:
            lines.append(f"- {note.equipment}{syn}")
    lines.append("")
    return "\n".join(lines)
