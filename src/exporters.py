"""Rapor dışa aktarma modülü: JSON, Excel, Markdown ve PDF."""

from __future__ import annotations

import io
import json
from datetime import datetime

import pandas as pd

from .ai_extractor import FinalMechanicalReport, MechanicalItem
from .utils import logger

ITEM_COLUMNS = [
    "category",
    "title",
    "requirement",
    "value_or_limit",
    "related_standard",
    "mechanical_relevance",
    "source_quote",
    "page_or_section",
    "confidence",
]

_COLUMN_LABELS = {
    "category": "Kategori",
    "title": "Başlık",
    "requirement": "Gereklilik",
    "value_or_limit": "Değer / Sınır",
    "related_standard": "Standart",
    "mechanical_relevance": "Mekanik Önem",
    "source_quote": "Kaynak Alıntı",
    "page_or_section": "Sayfa / Bölüm",
    "confidence": "Güven",
}


def items_to_dataframe(items: list[MechanicalItem]) -> pd.DataFrame:
    """Maddeleri pandas DataFrame'e çevirir (sabit kolon sırasıyla)."""
    if not items:
        return pd.DataFrame(columns=ITEM_COLUMNS)
    rows = [item.model_dump() for item in items]
    df = pd.DataFrame(rows)
    for col in ITEM_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[ITEM_COLUMNS]


# --------------------------------------------------------------------------- #
# JSON
# --------------------------------------------------------------------------- #
def export_json(report: FinalMechanicalReport, items: list[MechanicalItem]) -> bytes:
    """Rapor + maddeleri JSON bytes olarak döndürür."""
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "report": report.model_dump(),
        "items": [item.model_dump() for item in items],
        "item_count": len(items),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


# --------------------------------------------------------------------------- #
# Excel
# --------------------------------------------------------------------------- #
def export_excel(items: list[MechanicalItem], report: FinalMechanicalReport | None = None) -> bytes:
    """Maddeleri ve checklist'i çok sayfalı Excel olarak döndürür."""
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    df = items_to_dataframe(items)
    df_display = df.rename(columns=_COLUMN_LABELS)

    checklist_rows = report.checklist if report else []
    df_checklist = pd.DataFrame(
        {"#": range(1, len(checklist_rows) + 1), "Kontrol Maddesi": checklist_rows}
    )

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_display.to_excel(writer, sheet_name="Mechanical Items", index=False)
        df_checklist.to_excel(writer, sheet_name="Checklist", index=False)

        header_font = Font(bold=True, color="FFFFFF")
        header_fill = PatternFill("solid", fgColor="2F5496")
        wrap = Alignment(vertical="top", wrap_text=True)

        widths = {
            "Mechanical Items": [20, 26, 50, 18, 20, 40, 40, 16, 10],
            "Checklist": [6, 70],
        }

        for sheet_name, ws in writer.sheets.items():
            # Header stili
            for cell in ws[1]:
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = Alignment(vertical="center", horizontal="center")
            # Kolon genişlikleri
            for col_idx, width in enumerate(widths.get(sheet_name, []), start=1):
                ws.column_dimensions[get_column_letter(col_idx)].width = width
            # Gövde hücrelerinde metin sarma
            for row in ws.iter_rows(min_row=2):
                for cell in row:
                    cell.alignment = wrap
            ws.freeze_panes = "A2"

    buffer.seek(0)
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def export_markdown(report: FinalMechanicalReport, items: list[MechanicalItem]) -> str:
    """Raporu Markdown metni olarak döndürür."""
    lines: list[str] = []
    lines.append("# Mekanik Şartname Analiz Raporu")
    lines.append("")
    lines.append(f"**Belge:** {report.document_title}  ")
    lines.append(
        f"**Oluşturulma:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  "
    )
    lines.append(f"**Toplam mekanik madde:** {len(items)}")
    lines.append("")

    lines.append("## Yönetici Özeti")
    lines.append("")
    lines.append(report.executive_summary or "—")
    lines.append("")

    lines.append("## Kategori Bazlı Detaylar")
    lines.append("")
    if report.grouped_details:
        for category, details in report.grouped_details.items():
            lines.append(f"### {category}")
            for detail in details:
                lines.append(f"- {detail}")
            lines.append("")
    else:
        lines.append("_Kategori detayı bulunamadı._")
        lines.append("")

    lines.append("## Detaylı Mekanik Maddeler")
    lines.append("")
    if items:
        lines.append(
            "| Kategori | Başlık | Gereklilik | Değer/Sınır | Standart | Sayfa | Güven |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for it in items:
            lines.append(
                "| {cat} | {title} | {req} | {val} | {std} | {pg} | {conf:.2f} |".format(
                    cat=_md_cell(it.category),
                    title=_md_cell(it.title),
                    req=_md_cell(it.requirement),
                    val=_md_cell(it.value_or_limit),
                    std=_md_cell(it.related_standard),
                    pg=_md_cell(it.page_or_section),
                    conf=it.confidence,
                )
            )
        lines.append("")
    else:
        lines.append("_Madde bulunamadı._")
        lines.append("")

    lines.append("## Checklist")
    lines.append("")
    if report.checklist:
        for c in report.checklist:
            lines.append(f"- [ ] {c}")
    else:
        lines.append("_Checklist boş._")
    lines.append("")

    lines.append("## Eksik / Belirsiz Noktalar")
    lines.append("")
    if report.missing_or_unclear_points:
        for m in report.missing_or_unclear_points:
            lines.append(f"- {m}")
    else:
        lines.append("_Belirgin eksik/belirsiz nokta tespit edilmedi._")
    lines.append("")

    return "\n".join(lines)


def _md_cell(value: str | None) -> str:
    """Markdown tablo hücresi için metni güvenli hale getirir."""
    if not value:
        return "—"
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #
def export_pdf(report: FinalMechanicalReport, items: list[MechanicalItem]) -> bytes:
    """Raporu PDF bytes olarak döndürür (reportlab).

    reportlab kurulu değilse anlaşılır bir hata yükseltir.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "reportlab kurulu değil. PDF export için 'pip install reportlab' çalıştırın."
        ) from exc

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        title="Mekanik Şartname Analiz Raporu",
    )

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=styles["Title"], fontSize=18, spaceAfter=8)
    h2 = ParagraphStyle(
        "H2", parent=styles["Heading2"], fontSize=13, textColor=colors.HexColor("#2F5496")
    )
    body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=9, leading=13, alignment=TA_LEFT)
    small = ParagraphStyle("Small", parent=styles["Normal"], fontSize=7.5, leading=10)

    elements: list = []
    elements.append(Paragraph("Mekanik Şartname Analiz Raporu", h1))
    elements.append(
        Paragraph(
            f"<b>Belge:</b> {_pdf_text(report.document_title)} &nbsp;&nbsp; "
            f"<b>Tarih:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;&nbsp; "
            f"<b>Toplam madde:</b> {len(items)}",
            body,
        )
    )
    elements.append(Spacer(1, 8))

    elements.append(Paragraph("Yönetici Özeti", h2))
    elements.append(Paragraph(_pdf_text(report.executive_summary) or "—", body))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph("Kategori Bazlı Detaylar", h2))
    if report.grouped_details:
        for category, details in report.grouped_details.items():
            elements.append(Paragraph(f"<b>{_pdf_text(category)}</b>", body))
            for detail in details:
                elements.append(Paragraph(f"• {_pdf_text(detail)}", body))
            elements.append(Spacer(1, 4))
    else:
        elements.append(Paragraph("—", body))
    elements.append(Spacer(1, 6))

    elements.append(Paragraph("Detaylı Mekanik Maddeler", h2))
    if items:
        header = ["Kategori", "Başlık", "Gereklilik", "Değer", "Standart", "Güven"]
        table_data = [[Paragraph(f"<b>{h}</b>", small) for h in header]]
        for it in items:
            table_data.append(
                [
                    Paragraph(_pdf_text(it.category), small),
                    Paragraph(_pdf_text(it.title), small),
                    Paragraph(_pdf_text(it.requirement), small),
                    Paragraph(_pdf_text(it.value_or_limit) or "—", small),
                    Paragraph(_pdf_text(it.related_standard) or "—", small),
                    Paragraph(f"{it.confidence:.2f}", small),
                ]
            )
        table = Table(
            table_data,
            colWidths=[28 * mm, 30 * mm, 55 * mm, 22 * mm, 25 * mm, 12 * mm],
            repeatRows=1,
        )
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2F5496")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B0B0B0")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F4F8")]),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        elements.append(table)
    else:
        elements.append(Paragraph("Madde bulunamadı.", body))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph("Checklist", h2))
    if report.checklist:
        for c in report.checklist:
            elements.append(Paragraph(f"☐ {_pdf_text(c)}", body))
    else:
        elements.append(Paragraph("—", body))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph("Eksik / Belirsiz Noktalar", h2))
    if report.missing_or_unclear_points:
        for m in report.missing_or_unclear_points:
            elements.append(Paragraph(f"• {_pdf_text(m)}", body))
    else:
        elements.append(Paragraph("Belirgin eksik/belirsiz nokta tespit edilmedi.", body))

    try:
        doc.build(elements)
    except Exception as exc:
        logger.error("PDF oluşturulamadı: %s", exc)
        raise RuntimeError(f"PDF oluşturulamadı: {exc}") from exc

    buffer.seek(0)
    return buffer.getvalue()


def _pdf_text(value: str | None) -> str:
    """reportlab Paragraph için XML kaçışı uygular."""
    if not value:
        return ""
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .strip()
    )
