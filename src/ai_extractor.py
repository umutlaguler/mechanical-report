"""Yapay zeka tabanlı disiplin-parametrik madde çıkarıcı.

Mekanik ve elektrik disiplinlerini ayrı ayrı işler. Her disiplin için kendi
sistem prompt'u + `model için notlar.xlsx`'ten gelen domain notları kullanılır.
Şartname herhangi bir dilde olabilir; dil otomatik algılanır ve rapor Türkçe
üretilir (yalnız source_quote orijinal dilde korunur).

OpenAI structured output kullanır. SDK sürümüne göre üç yöntem sırayla denenir:
1. client.responses.parse(... text_format=...)   (yeni Responses API)
2. client.beta.chat.completions.parse(... response_format=...)  (parse helper)
3. chat.completions + JSON mode + manuel Pydantic parse (her zaman çalışan fallback)

Her chunk bağımsız işlenir; bir chunk hata verirse loglanır ve diğerleri devam eder.
"""

from __future__ import annotations

import json
from typing import Callable, Optional

from pydantic import BaseModel, Field, ValidationError

from .config import AI_TEMPERATURE, categories_for, discipline_label
from .notes_loader import build_notes_prompt_block
from .text_chunker import TextChunk
from .utils import logger

# --------------------------------------------------------------------------- #
# Pydantic modelleri (structured output şeması)
# --------------------------------------------------------------------------- #


class SpecItem(BaseModel):
    """Tek bir mühendislik gerekliliği/maddesi (mekanik veya elektrik)."""

    discipline: str = Field(
        default="mekanik", description="Maddenin disiplini: 'mekanik' veya 'elektrik'"
    )
    category: str = Field(description="Maddenin ait olduğu kategori")
    title: str = Field(description="Kısa başlık")
    requirement: str = Field(description="Gerekliliğin açık ifadesi")
    value_or_limit: Optional[str] = Field(
        default=None, description="Varsa sayısal değer veya sınır (ör. 3 mm, 16 bar, 36 kV)"
    )
    related_standard: Optional[str] = Field(
        default=None, description="Varsa ilgili standart (ör. TS EN 60076, IEC 60137, EN 50180)"
    )
    relevance: str = Field(
        description="Bu maddenin ilgili disiplin (mekanik/elektrik) açısından önemi"
    )
    source_quote: Optional[str] = Field(
        default=None, description="Şartnameden ORİJİNAL dilde birebir alıntı"
    )
    page_or_section: Optional[str] = Field(
        default=None, description="Sayfa veya bölüm bilgisi"
    )
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="0-1 arası güven skoru"
    )


# Geriye dönük uyumluluk için eski isim
MechanicalItem = SpecItem


class ChunkExtractionResult(BaseModel):
    """Bir chunk'tan çıkarılan tüm maddeler."""

    items: list[SpecItem] = Field(default_factory=list)


class FinalReport(BaseModel):
    """Bir disiplin için, tüm chunk'lar birleştirildikten sonraki nihai rapor."""

    discipline: str = "mekanik"
    document_title: str = ""
    executive_summary: str = ""
    grouped_details: dict[str, list[str]] = Field(default_factory=dict)
    checklist: list[str] = Field(default_factory=list)
    missing_or_unclear_points: list[str] = Field(default_factory=list)


# Geriye dönük uyumluluk için eski isim
FinalMechanicalReport = FinalReport


# --------------------------------------------------------------------------- #
# Promptlar
# --------------------------------------------------------------------------- #

# Her iki disipline ortak, çok dillilik talimatı
_MULTILINGUAL_BLOCK = """ÇOK DİLLİLİK: Şartname herhangi bir dilde yazılmış olabilir \
(Türkçe, İngilizce, Almanca, Fransızca, İspanyolca, Rusça, Çince, Arapça vb.). \
Belgenin dilini otomatik algıla ve içeriği eksiksiz anla. ÇIKTIDAKİ TÜM ALANLARI \
(category, title, requirement, value_or_limit, related_standard, relevance, page_or_section) \
TÜRKÇE üret. SADECE `source_quote` alanı şartnamedeki ORİJİNAL dildeki birebir alıntı olarak kalsın."""

_COMMON_FIELDS = """Her madde için şunları doldur: discipline ("{discipline}"), category, title, \
requirement, value_or_limit (yoksa null), related_standard (yoksa null), relevance, \
source_quote (mümkünse orijinal dilde birebir alıntı), \
page_or_section (metindeki PAGE/CHUNK bilgisinden çıkar), confidence (0-1).

İlgili madde yoksa boş bir items listesi döndür. Sadece istenen JSON yapısında cevap ver."""


_MECHANICAL_PROMPT = """Sen teknik şartnameleri analiz eden kıdemli bir MEKANİK mühendissin. \
Görevin, verilen şartname metninden SADECE mekanik mühendisliği ilgilendiren maddeleri çıkarmaktır.

Yakalaman gereken konular (örnekler):
Tank, kazan, kapak, gövde, şasi, karkas, konstrüksiyon, sac kalınlığı, malzeme kalitesi, \
çelik, paslanmaz çelik, alüminyum, galvaniz, kaynak, çift kaynak, sızdırmazlık, conta, flanş, \
cıvata, somun, basınç, vakum, iç basınç, yırtılma/patlama basıncı, emniyet valfi, yağ doldurma/boşaltma, \
vana, yağ seviyesi, soğutma yüzeyi, radyatör, dalga duvar (corrugated wall), fan, doğal soğutma, \
boya, kaplama, korozyon, kumlama, yüzey hazırlığı, RAL renk kodları, boya kalınlığı (mikron), \
kaldırma halkası, taşıma, tekerlek, ray açıklığı, ambalaj, sevkiyat, ölçüler, ağırlık, toleranslar, \
maksimum boyutlar, mekanik testler (sızdırmazlık/basınç/boya/galvaniz testi), IP koruma sınıfı, \
kablo kutusu (mekanik mahfaza yönü), muhafaza/mahfaza, montaj detayları, mekanik aksesuarlar.

Çıkarmaman gereken konular:
Saf elektriksel değerler (gerilim seviyesi, kısa devre empedansı, sargı bağlantı grubu, \
koruma rölesi gibi) mekanik tasarımı etkilemiyorsa dahil etme.

ANCAK elektriksel bir madde mekanik tasarımı etkiliyorsa dahil et. Örnekler: buşing yerleşimi, \
minimum açıklıklar, kablo kutusunun mekanik mahfazası, izolatör montajı, soğutma sistemi mekaniği."""


_ELECTRICAL_PROMPT = """Sen teknik şartnameleri analiz eden kıdemli bir ELEKTRİK mühendissin. \
Görevin, verilen şartname metninden SADECE elektrik mühendisliği ilgilendiren maddeleri çıkarmaktır.

Yakalaman gereken konular (örnekler):
Anma gücü (kVA/MVA), gerilim sınıfı/seviyesi, frekans, sargı bağlantı grubu ve vektör grubu, \
yıldız/üçgen bağlantı, kademe değiştirici (OLTC/DETC), kademe aralığı, yalıtım seviyesi ve BIL/LI/AC \
test gerilimleri, kısa devre empedansı (%Uk), yük ve boş yük kayıpları, verimlilik, ısınma sınırı, \
yalıtım sınıfı, izolasyon koordinasyonu, bushing elektriksel değerleri (akım, gerilim, BIL), \
kablo kutusu/busbar elektriksel bağlantısı, terminal ve nötr çıkışı, topraklama ve nötr topraklaması, \
yıldırımlık/parafudr, koruma röleleri ve sensörler (Buchholz, termometre, yağ/sargı sıcaklık, basınç rölesi), \
izleme/monitoring, akım/gerilim transformatörleri, elektriksel testler (rutin/tip/özel testler).

Çıkarmaman gereken konular:
Saf mekanik konular (sac kalınlığı, boya/galvaniz, kaynak, taşıma donanımı, ambalaj gibi) \
elektriksel bir gereklilik içermiyorsa dahil etme.

ANCAK mekanik bir madde elektriksel performans/güvenlik için kritikse dahil et. Örnekler: \
buşing minimum açıklıkları, topraklama terminali yerleşimi, kablo kutusu giriş yönü ve IP sınıfı."""


_DISCIPLINE_PROMPTS = {
    "mekanik": _MECHANICAL_PROMPT,
    "elektrik": _ELECTRICAL_PROMPT,
}


def build_system_prompt(discipline: str) -> str:
    """Disipline özel sistem prompt'unu (çok dillilik + domain notları dahil) üretir."""
    base = _DISCIPLINE_PROMPTS.get(discipline, _MECHANICAL_PROMPT)
    categories = categories_for(discipline)
    categories_text = ", ".join(categories)

    parts = [base, "", _MULTILINGUAL_BLOCK, ""]

    notes_block = build_notes_prompt_block(discipline)
    if notes_block:
        parts.append(notes_block)

    parts.append(f"Kategori olarak mümkünse şunlardan birini kullan: {categories_text}.")
    parts.append("")
    parts.append(_COMMON_FIELDS.format(discipline=discipline))
    return "\n".join(parts)


def _build_user_prompt(chunk_text_with_header: str, discipline: str) -> str:
    label = discipline_label(discipline)
    return (
        f"Aşağıdaki şartname parçasından {label} mühendisliğini ilgilendiren tüm maddeleri çıkar.\n"
        "Metindeki '--- PAGE X ---' ve '[CHUNK i/n]' ifadelerini page_or_section alanını "
        "doldurmak için kullan.\n\n"
        "=== ŞARTNAME PARÇASI BAŞLANGICI ===\n"
        f"{chunk_text_with_header}\n"
        "=== ŞARTNAME PARÇASI SONU ==="
    )


# --------------------------------------------------------------------------- #
# OpenAI istemcisi
# --------------------------------------------------------------------------- #


class AIExtractionError(Exception):
    """AI çıkarımı sırasında oluşan hata."""


def _make_client(api_key: str):
    """OpenAI istemcisini oluşturur."""
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise AIExtractionError(
            "openai paketi kurulu değil. 'pip install openai' çalıştırın."
        ) from exc
    return OpenAI(api_key=api_key)


def _extract_one_chunk(
    client, model: str, chunk: TextChunk, discipline: str
) -> ChunkExtractionResult:
    """Tek bir chunk'ı verilen disipline göre işler."""
    system_prompt = build_system_prompt(discipline)
    user_prompt = _build_user_prompt(chunk.with_header(), discipline)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # 1) Yeni Responses API parse helper
    try:
        if hasattr(client, "responses") and hasattr(client.responses, "parse"):
            completion = client.responses.parse(
                model=model,
                input=messages,
                text_format=ChunkExtractionResult,
            )
            parsed = getattr(completion, "output_parsed", None)
            if isinstance(parsed, ChunkExtractionResult):
                return _tag_discipline(parsed, discipline)
    except Exception as exc:  # noqa: BLE001 - bir sonraki yönteme düş
        logger.debug("responses.parse kullanılamadı, fallback: %s", exc)

    # 2) chat.completions.parse helper (beta)
    try:
        beta = getattr(client, "beta", None)
        parse_fn = getattr(
            getattr(getattr(beta, "chat", None), "completions", None), "parse", None
        )
        if parse_fn is not None:
            completion = parse_fn(
                model=model,
                messages=messages,
                response_format=ChunkExtractionResult,
            )
            parsed = completion.choices[0].message.parsed
            if isinstance(parsed, ChunkExtractionResult):
                return _tag_discipline(parsed, discipline)
    except Exception as exc:  # noqa: BLE001 - JSON fallback'e düş
        logger.debug("chat.completions.parse kullanılamadı, fallback: %s", exc)

    # 3) JSON mode fallback (her zaman çalışır)
    result = _extract_with_json_mode(client, model, messages)
    return _tag_discipline(result, discipline)


def _tag_discipline(result: ChunkExtractionResult, discipline: str) -> ChunkExtractionResult:
    """Tüm maddelere disiplin etiketini garanti eder."""
    for item in result.items:
        item.discipline = discipline
    return result


def _extract_with_json_mode(client, model: str, messages: list[dict]) -> ChunkExtractionResult:
    """response_format=json_object ile çıktı alıp manuel parse eder."""
    schema_hint = (
        "\n\nÇıktıyı şu JSON şemasında ver: "
        '{"items": [{"discipline": str, "category": str, "title": str, "requirement": str, '
        '"value_or_limit": str|null, "related_standard": str|null, '
        '"relevance": str, "source_quote": str|null, '
        '"page_or_section": str|null, "confidence": float}]}'
    )
    json_messages = [dict(m) for m in messages]
    json_messages[-1]["content"] += schema_hint

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=json_messages,
            response_format={"type": "json_object"},
            temperature=AI_TEMPERATURE,
        )
        content = completion.choices[0].message.content or "{}"
    except Exception as exc:
        raise AIExtractionError(f"OpenAI API hatası: {exc}") from exc

    return _parse_json_to_result(content)


def _parse_json_to_result(content: str) -> ChunkExtractionResult:
    """Ham JSON string'ini ChunkExtractionResult'a güvenli çevirir."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Bazen model metin arasında JSON döndürür; ilk { ... } bloğunu ayıkla
        start, end = content.find("{"), content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise AIExtractionError("Model geçerli JSON döndürmedi.")
        try:
            data = json.loads(content[start : end + 1])
        except json.JSONDecodeError as exc:
            raise AIExtractionError(f"JSON parse hatası: {exc}") from exc

    if isinstance(data, list):
        data = {"items": data}
    if not isinstance(data, dict):
        return ChunkExtractionResult(items=[])

    try:
        return ChunkExtractionResult.model_validate(data)
    except ValidationError:
        # Madde madde kurtarmayı dene
        items: list[SpecItem] = []
        for raw in data.get("items", []) or []:
            try:
                items.append(SpecItem.model_validate(raw))
            except ValidationError as exc:
                logger.warning("Geçersiz madde atlandı: %s", exc)
        return ChunkExtractionResult(items=items)


# --------------------------------------------------------------------------- #
# Genel çıkarım API'si
# --------------------------------------------------------------------------- #


def extract_items(
    chunks: list[TextChunk],
    api_key: str,
    disciplines: list[str],
    model: str = "gpt-4.1-mini",
    progress_callback: Callable[[str, int, int, int], None] | None = None,
) -> tuple[list[SpecItem], list[str]]:
    """Tüm chunk'ları her disiplin için işler.

    Args:
        chunks: İşlenecek metin parçaları.
        api_key: OpenAI API anahtarı.
        disciplines: İşlenecek disiplinler (ör. ["mekanik", "elektrik"]).
        model: Kullanılacak model.
        progress_callback: (disiplin, işlenen, toplam, bulunan madde) ile çağrılır.

    Returns:
        (tüm maddeler [disiplin etiketli], hata mesajları) ikilisi.
    """
    if not api_key:
        raise AIExtractionError("OpenAI API anahtarı bulunamadı.")
    if not chunks or not disciplines:
        return [], []

    client = _make_client(api_key)
    all_items: list[SpecItem] = []
    errors: list[str] = []
    total = len(chunks)

    for discipline in disciplines:
        label = discipline_label(discipline)
        found_in_discipline = 0
        for i, chunk in enumerate(chunks, start=1):
            try:
                result = _extract_one_chunk(client, model, chunk, discipline)
                all_items.extend(result.items)
                found_in_discipline += len(result.items)
                logger.info(
                    "[%s] Chunk %d/%d işlendi, %d madde bulundu.",
                    label, i, total, len(result.items),
                )
            except Exception as exc:  # noqa: BLE001 - tek chunk hatası akışı durdurmaz
                msg = f"[{label}] Chunk {i}/{total} işlenemedi: {exc}"
                logger.error(msg)
                errors.append(msg)
            finally:
                if progress_callback is not None:
                    progress_callback(discipline, i, total, found_in_discipline)

    return all_items, errors


def extract_mechanical_items(
    chunks: list[TextChunk],
    api_key: str,
    model: str = "gpt-4.1-mini",
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> tuple[list[SpecItem], list[str]]:
    """Geriye dönük uyumluluk sarmalayıcısı: yalnızca mekanik disiplini işler."""
    cb = None
    if progress_callback is not None:
        def cb(_disc, done, total, found):  # noqa: ANN001
            progress_callback(done, total, found)

    return extract_items(
        chunks, api_key=api_key, disciplines=["mekanik"], model=model, progress_callback=cb
    )
