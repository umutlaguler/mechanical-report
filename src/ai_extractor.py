"""Yapay zeka tabanlı mekanik madde çıkarıcı.

OpenAI structured output kullanır. SDK sürümüne göre üç yöntem sırayla denenir:
1. client.responses.parse(... text_format=...)   (yeni Responses API)
2. client.beta.chat.completions.parse(... response_format=...)  (parse helper)
3. chat.completions + JSON mode + manuel Pydantic parse (her zaman çalışan fallback)

Her chunk bağımsız işlenir; bir chunk hata verirse loglanır ve diğerleri devam eder.
"""

from __future__ import annotations

import json
from typing import Callable

from pydantic import BaseModel, Field, ValidationError

from .text_chunker import TextChunk
from .utils import logger

# --------------------------------------------------------------------------- #
# Pydantic modelleri (structured output şeması)
# --------------------------------------------------------------------------- #


class MechanicalItem(BaseModel):
    """Tek bir mekanik gereklilik/madde."""

    category: str = Field(description="Maddenin ait olduğu mekanik kategori")
    title: str = Field(description="Kısa başlık")
    requirement: str = Field(description="Gerekliliğin açık ifadesi")
    value_or_limit: str | None = Field(
        default=None, description="Varsa sayısal değer veya sınır (ör. 3 mm, 16 bar)"
    )
    related_standard: str | None = Field(
        default=None, description="Varsa ilgili standart (ör. TS EN 60076, ISO 1461)"
    )
    mechanical_relevance: str = Field(
        description="Bu maddenin mekanik mühendislik açısından önemi"
    )
    source_quote: str | None = Field(
        default=None, description="Şartnameden birebir alıntı"
    )
    page_or_section: str | None = Field(
        default=None, description="Sayfa veya bölüm bilgisi"
    )
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="0-1 arası güven skoru"
    )


class ChunkExtractionResult(BaseModel):
    """Bir chunk'tan çıkarılan tüm maddeler."""

    items: list[MechanicalItem] = Field(default_factory=list)


class FinalMechanicalReport(BaseModel):
    """Tüm chunk'lar birleştirildikten sonraki nihai rapor."""

    document_title: str
    executive_summary: str
    grouped_details: dict[str, list[str]] = Field(default_factory=dict)
    checklist: list[str] = Field(default_factory=list)
    missing_or_unclear_points: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Promptlar
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """Sen teknik şartnameleri analiz eden kıdemli bir mekanik mühendissin. \
Görevin, verilen şartname metninden SADECE mekanik mühendisliği ilgilendiren maddeleri çıkarmaktır.

Yakalaman gereken konular (örnekler):
Tank, kazan, kapak, gövde, şasi, karkas, konstrüksiyon, sac kalınlığı, malzeme kalitesi, \
çelik, paslanmaz çelik, alüminyum, galvaniz, kaynak, çift kaynak, sızdırmazlık, conta, flanş, \
cıvata, somun, basınç, vakum, iç basınç, yırtılma/patlama basıncı, emniyet valfi, yağ doldurma/boşaltma, \
vana, yağ seviyesi, soğutma yüzeyi, radyatör, dalga duvar (corrugated wall), fan, doğal soğutma, \
boya, kaplama, korozyon, kumlama, yüzey hazırlığı, RAL renk kodları, boya kalınlığı (mikron), \
kaldırma halkası, taşıma, tekerlek, ray açıklığı, ambalaj, sevkiyat, ölçüler, ağırlık, toleranslar, \
maksimum boyutlar, mekanik testler (sızdırmazlık/basınç/boya/galvaniz testi), IP koruma sınıfı, \
kablo kutusu, muhafaza/mahfaza, montaj detayları, mekanik aksesuarlar.

Çıkarmaman gereken konular:
Saf elektriksel değerler, kısa devre empedansı, gerilim seviyesi gibi mekanik bağlantısı olmayan \
parametreler, rutin elektrik deneyleri, koruma rölesi gibi mekanik olmayan elektriksel konular.

ANCAK elektriksel bir madde mekanik tasarımı etkiliyorsa dahil et. Örnekler: buşing yerleşimi, \
minimum açıklıklar, kablo kutusu, izolatör montajı, bara bağlantı düzeni, topraklama terminali, soğutma sistemi.

Her madde için şunları doldur: category, title, requirement, value_or_limit (yoksa null), \
related_standard (yoksa null), mechanical_relevance, source_quote (mümkünse birebir alıntı), \
page_or_section (metindeki PAGE/CHUNK bilgisinden çıkar), confidence (0-1).

Mekanik madde yoksa boş bir items listesi döndür. Sadece istenen JSON yapısında cevap ver."""


def _build_user_prompt(chunk_text_with_header: str) -> str:
    return (
        "Aşağıdaki şartname parçasından mekanik mühendisliği ilgilendiren tüm maddeleri çıkar.\n"
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


def _extract_one_chunk(client, model: str, chunk: TextChunk) -> ChunkExtractionResult:
    """Tek bir chunk'ı işler ve ChunkExtractionResult döndürür."""
    user_prompt = _build_user_prompt(chunk.with_header())
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
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
                return parsed
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
                return parsed
    except Exception as exc:  # noqa: BLE001 - JSON fallback'e düş
        logger.debug("chat.completions.parse kullanılamadı, fallback: %s", exc)

    # 3) JSON mode fallback (her zaman çalışır)
    return _extract_with_json_mode(client, model, messages)


def _extract_with_json_mode(client, model: str, messages: list[dict]) -> ChunkExtractionResult:
    """response_format=json_object ile çıktı alıp manuel parse eder."""
    schema_hint = (
        "\n\nÇıktıyı şu JSON şemasında ver: "
        '{"items": [{"category": str, "title": str, "requirement": str, '
        '"value_or_limit": str|null, "related_standard": str|null, '
        '"mechanical_relevance": str, "source_quote": str|null, '
        '"page_or_section": str|null, "confidence": float}]}'
    )
    json_messages = [dict(m) for m in messages]
    json_messages[-1]["content"] += schema_hint

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=json_messages,
            response_format={"type": "json_object"},
            temperature=0.1,
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
        items: list[MechanicalItem] = []
        for raw in data.get("items", []) or []:
            try:
                items.append(MechanicalItem.model_validate(raw))
            except ValidationError as exc:
                logger.warning("Geçersiz madde atlandı: %s", exc)
        return ChunkExtractionResult(items=items)


def extract_mechanical_items(
    chunks: list[TextChunk],
    api_key: str,
    model: str = "gpt-4.1-mini",
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> tuple[list[MechanicalItem], list[str]]:
    """Tüm chunk'ları işler.

    Args:
        chunks: İşlenecek metin parçaları.
        api_key: OpenAI API anahtarı.
        model: Kullanılacak model.
        progress_callback: (işlenen, toplam, o ana kadar bulunan madde) ile çağrılır.

    Returns:
        (tüm maddeler, hata mesajları) ikilisi.
    """
    if not api_key:
        raise AIExtractionError("OpenAI API anahtarı bulunamadı.")
    if not chunks:
        return [], []

    client = _make_client(api_key)
    all_items: list[MechanicalItem] = []
    errors: list[str] = []
    total = len(chunks)

    for i, chunk in enumerate(chunks, start=1):
        try:
            result = _extract_one_chunk(client, model, chunk)
            all_items.extend(result.items)
            logger.info(
                "Chunk %d/%d işlendi, %d madde bulundu.", i, total, len(result.items)
            )
        except Exception as exc:  # noqa: BLE001 - tek chunk hatası tüm akışı durdurmaz
            msg = f"Chunk {i}/{total} işlenemedi: {exc}"
            logger.error(msg)
            errors.append(msg)
        finally:
            if progress_callback is not None:
                progress_callback(i, total, len(all_items))

    return all_items, errors
