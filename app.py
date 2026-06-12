"""Mekanik Şartname Analizörü — Streamlit uygulaması.

Çalıştırma:
    streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.ai_extractor import (
    AIExtractionError,
    FinalMechanicalReport,
    MechanicalItem,
    extract_mechanical_items,
)
from src.config import (
    AVAILABLE_MODELS,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    LOW_CONFIDENCE_THRESHOLD,
    MAX_CHUNK_SIZE,
    MIN_CHUNK_SIZE,
    UPLOADER_TYPES,
    get_api_key,
    get_default_model,
)
from src.document_reader import (
    DocumentReadError,
    read_document_from_uploaded_file,
    read_document_from_url,
)
from src.exporters import (
    export_excel,
    export_json,
    export_markdown,
    export_pdf,
    items_to_dataframe,
)
from src.report_builder import (
    average_confidence,
    build_report,
    group_by_category,
    low_confidence_items,
)
from src.text_chunker import chunk_text
from src.utils import safe_filename, truncate

# --------------------------------------------------------------------------- #
# Sayfa ayarları + hafif stil
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Mekanik Şartname Analizörü",
    page_icon="⚙️",
    layout="wide",
)

st.markdown(
    """
    <style>
        .main { background-color: #f7f9fc; }
        .block-container { padding-top: 2rem; }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e3e8ef;
            border-radius: 12px;
            padding: 14px 16px;
            box-shadow: 0 1px 2px rgba(16,24,40,0.04);
        }
        .stTabs [data-baseweb="tab-list"] { gap: 6px; }
        .stTabs [data-baseweb="tab"] {
            background: #ffffff;
            border-radius: 8px 8px 0 0;
            padding: 8px 16px;
        }
        h1, h2, h3 { color: #1f2a44; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
if "analysis" not in st.session_state:
    st.session_state.analysis = None  # dict: report, items, errors, doc_chars


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar() -> dict:
    """Sidebar'ı çizer ve ayarları döndürür."""
    st.sidebar.header("⚙️ Ayarlar")

    api_key = get_api_key()
    if api_key:
        st.sidebar.success("OpenAI API anahtarı bulundu (.env).")
    else:
        st.sidebar.error("API anahtarı yok.")
        st.sidebar.caption(
            "Proje kökünde `.env` dosyası oluşturup `OPENAI_API_KEY=...` ekleyin "
            "veya aşağıya yapıştırın."
        )
    manual_key = st.sidebar.text_input(
        "API anahtarı (opsiyonel)",
        type="password",
        help="Girilirse .env yerine bu anahtar kullanılır. Hiçbir yere kaydedilmez.",
    )
    effective_key = manual_key.strip() or api_key

    st.sidebar.divider()
    st.sidebar.subheader("Analiz Modu")
    st.sidebar.selectbox(
        "Mod",
        ["Mekanik Mühendislik Analizi"],
        index=0,
        help="Şimdilik yalnızca mekanik analiz aktiftir. Yapı yeni modlara açıktır.",
    )

    default_model = get_default_model()
    model = st.sidebar.selectbox(
        "Model",
        AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(default_model)
        if default_model in AVAILABLE_MODELS
        else 0,
    )

    st.sidebar.divider()
    st.sidebar.subheader("Chunking")
    chunk_size = st.sidebar.slider(
        "Chunk boyutu (karakter)",
        MIN_CHUNK_SIZE,
        MAX_CHUNK_SIZE,
        DEFAULT_CHUNK_SIZE,
        step=500,
    )
    overlap = st.sidebar.slider(
        "Overlap (karakter)", 0, 2000, DEFAULT_OVERLAP, step=100
    )

    st.sidebar.divider()
    st.sidebar.subheader("Filtre & Özet")
    min_confidence = st.sidebar.slider(
        "Minimum güven skoru", 0.0, 1.0, 0.0, step=0.05
    )
    use_ai_summary = st.sidebar.checkbox(
        "AI ile yönetici özeti üret", value=True
    )

    st.sidebar.divider()
    st.sidebar.subheader("Export Seçenekleri")
    exports = {
        "json": st.sidebar.checkbox("JSON", value=True),
        "excel": st.sidebar.checkbox("Excel", value=True),
        "markdown": st.sidebar.checkbox("Markdown", value=True),
        "pdf": st.sidebar.checkbox("PDF", value=True),
    }

    return {
        "api_key": effective_key,
        "model": model,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "min_confidence": min_confidence,
        "use_ai_summary": use_ai_summary,
        "exports": exports,
    }


# --------------------------------------------------------------------------- #
# Belge okuma
# --------------------------------------------------------------------------- #
def read_input_document(uploaded_file, url: str) -> str | None:
    """Yüklenen dosya veya URL'den metni okur. Hata olursa UI'da gösterir."""
    try:
        if uploaded_file is not None:
            with st.spinner("Dosya okunuyor..."):
                return read_document_from_uploaded_file(uploaded_file)
        if url.strip():
            with st.spinner("URL'den belge indiriliyor..."):
                return read_document_from_url(url.strip())
    except DocumentReadError as exc:
        st.error(f"📄 Belge okunamadı: {exc}")
        return None
    except Exception as exc:  # noqa: BLE001
        st.error(f"Beklenmeyen hata: {exc}")
        return None

    st.warning("Lütfen bir dosya yükleyin veya bir URL girin.")
    return None


# --------------------------------------------------------------------------- #
# Analiz akışı
# --------------------------------------------------------------------------- #
def run_analysis(text: str, settings: dict) -> None:
    """Metni chunk'lar, AI ile işler, rapor üretir ve session'a yazar."""
    if not settings["api_key"]:
        st.error("OpenAI API anahtarı gerekli. Sidebar'dan girin veya .env tanımlayın.")
        return

    chunks = chunk_text(text, settings["chunk_size"], settings["overlap"])
    if not chunks:
        st.error("Belgeden işlenecek metin çıkarılamadı.")
        return

    st.info(f"Belge {len(chunks)} parçaya bölündü. Analiz başlıyor...")
    progress_bar = st.progress(0.0)
    status = st.empty()

    def on_progress(done: int, total: int, found: int) -> None:
        progress_bar.progress(done / total)
        status.markdown(
            f"🔄 Chunk **{done}/{total}** işlendi — şimdiye kadar **{found}** madde bulundu."
        )

    try:
        items, errors = extract_mechanical_items(
            chunks,
            api_key=settings["api_key"],
            model=settings["model"],
            progress_callback=on_progress,
        )
    except AIExtractionError as exc:
        progress_bar.empty()
        st.error(f"🤖 AI hatası: {exc}")
        return
    except Exception as exc:  # noqa: BLE001
        progress_bar.empty()
        st.error(f"Beklenmeyen analiz hatası: {exc}")
        return

    progress_bar.progress(1.0)

    if errors:
        with st.expander(f"⚠️ {len(errors)} chunk işlenirken hata oluştu (detaylar)"):
            for err in errors:
                st.write(f"- {err}")

    if not items:
        st.warning(
            "Mekanik mühendisliği ilgilendiren madde bulunamadı. "
            "Belge içeriğini veya minimum güven filtresini kontrol edin."
        )

    with st.spinner("Rapor oluşturuluyor..."):
        report, processed = build_report(
            items,
            document_text=text,
            min_confidence=settings["min_confidence"],
            use_ai_summary=settings["use_ai_summary"],
            api_key=settings["api_key"],
            model=settings["model"],
        )

    st.session_state.analysis = {
        "report": report,
        "items": processed,
        "errors": errors,
        "doc_chars": len(text),
        "chunk_count": len(chunks),
    }
    status.empty()
    st.success("✅ Analiz tamamlandı.")


# --------------------------------------------------------------------------- #
# Sonuç görünümü
# --------------------------------------------------------------------------- #
def render_results(settings: dict) -> None:
    """Analiz sonuçlarını sekmeler halinde gösterir."""
    analysis = st.session_state.analysis
    report: FinalMechanicalReport = analysis["report"]
    items: list[MechanicalItem] = analysis["items"]

    grouped = group_by_category(items)
    low = low_confidence_items(items)

    st.divider()
    st.subheader(f"📋 {report.document_title}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mekanik Madde", len(items))
    c2.metric("Kategori", len(grouped))
    c3.metric("Ort. Güven", f"{average_confidence(items):.2f}")
    c4.metric("Düşük Güvenli", len(low))

    tab_summary, tab_cat, tab_table, tab_check, tab_json, tab_export = st.tabs(
        ["📝 Özet", "🗂️ Kategoriler", "📊 Tablo", "✅ Checklist", "🧾 Ham JSON", "⬇️ Export"]
    )

    with tab_summary:
        st.markdown("### Yönetici Özeti")
        st.write(report.executive_summary or "—")
        st.markdown("### Eksik / Belirsiz Noktalar")
        if report.missing_or_unclear_points:
            for m in report.missing_or_unclear_points:
                st.markdown(f"- {m}")
        else:
            st.caption("Belirgin eksik/belirsiz nokta tespit edilmedi.")

    with tab_cat:
        if not grouped:
            st.info("Gösterilecek kategori yok.")
        for category, members in grouped.items():
            with st.expander(f"{category}  ·  {len(members)} madde", expanded=False):
                for it in members:
                    conf_icon = "🟢" if it.confidence >= LOW_CONFIDENCE_THRESHOLD else "🟡"
                    st.markdown(f"**{conf_icon} {it.title}** · güven {it.confidence:.2f}")
                    st.markdown(f"- **Gereklilik:** {it.requirement}")
                    if it.value_or_limit:
                        st.markdown(f"- **Değer/Sınır:** {it.value_or_limit}")
                    if it.related_standard:
                        st.markdown(f"- **Standart:** {it.related_standard}")
                    st.markdown(f"- **Mekanik önem:** {it.mechanical_relevance}")
                    if it.page_or_section:
                        st.markdown(f"- **Sayfa/Bölüm:** {it.page_or_section}")
                    if it.source_quote:
                        st.caption(f"“{truncate(it.source_quote, 240)}”")
                    st.divider()

    with tab_table:
        df = items_to_dataframe(items)
        if df.empty:
            st.info("Tabloda gösterilecek madde yok.")
        else:
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "confidence": st.column_config.ProgressColumn(
                        "Güven", min_value=0.0, max_value=1.0, format="%.2f"
                    )
                },
            )

    with tab_check:
        st.markdown("### Mekanik Checklist")
        if report.checklist:
            for i, c in enumerate(report.checklist):
                st.checkbox(c, key=f"check_{i}", value=False)
        else:
            st.caption("Checklist boş.")

    with tab_json:
        st.json(
            {
                "report": report.model_dump(),
                "items": [it.model_dump() for it in items],
            }
        )

    with tab_export:
        render_export_buttons(report, items, settings)


def render_export_buttons(
    report: FinalMechanicalReport, items: list[MechanicalItem], settings: dict
) -> None:
    """Seçili export formatları için indirme butonları üretir."""
    exports = settings["exports"]
    base = safe_filename(report.document_title, "mekanik_rapor")

    st.markdown("### İndirilebilir Çıktılar")
    cols = st.columns(4)

    if exports.get("json"):
        with cols[0]:
            st.download_button(
                "⬇️ JSON",
                data=export_json(report, items),
                file_name=f"{base}.json",
                mime="application/json",
                use_container_width=True,
            )

    if exports.get("excel"):
        with cols[1]:
            try:
                excel_bytes = export_excel(items, report)
                st.download_button(
                    "⬇️ Excel",
                    data=excel_bytes,
                    file_name=f"{base}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Excel oluşturulamadı: {exc}")

    if exports.get("markdown"):
        with cols[2]:
            st.download_button(
                "⬇️ Markdown",
                data=export_markdown(report, items).encode("utf-8"),
                file_name=f"{base}.md",
                mime="text/markdown",
                use_container_width=True,
            )

    if exports.get("pdf"):
        with cols[3]:
            try:
                pdf_bytes = export_pdf(report, items)
                st.download_button(
                    "⬇️ PDF",
                    data=pdf_bytes,
                    file_name=f"{base}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as exc:  # noqa: BLE001
                st.warning(f"PDF oluşturulamadı: {exc}")

    if not any(exports.values()):
        st.info("Sidebar'dan en az bir export formatı seçin.")


# --------------------------------------------------------------------------- #
# Ana akış
# --------------------------------------------------------------------------- #
def main() -> None:
    settings = render_sidebar()

    st.title("⚙️ Mekanik Şartname Analizörü")
    st.markdown(
        "Teknik şartnamelerinizi yükleyin; uygulama **mekanik mühendisliği ilgilendiren** "
        "tüm maddeleri yapay zeka ile çıkarır, kategorilere ayırır ve indirilebilir rapor üretir."
    )

    col_file, col_url = st.columns(2)
    with col_file:
        uploaded_file = st.file_uploader(
            "📎 Şartname dosyası yükle",
            type=UPLOADER_TYPES,
            help="Desteklenen: PDF, DOCX, TXT, MD, HTML",
        )
    with col_url:
        url = st.text_input(
            "🌐 veya bir URL girin",
            placeholder="https://ornek.com/sartname.pdf",
            help="PDF/HTML URL'si. HTML içinde PDF linki varsa otomatik bulunur.",
        )

    analyze = st.button(
        "🔍 Şartnameyi Analiz Et", type="primary", use_container_width=True
    )

    if analyze:
        text = read_input_document(uploaded_file, url)
        if text:
            run_analysis(text, settings)

    if st.session_state.analysis is not None:
        render_results(settings)
    else:
        st.caption(
            "Henüz analiz yapılmadı. Bir dosya yükleyin veya URL girin, "
            "ardından **Şartnameyi Analiz Et**'e basın."
        )


if __name__ == "__main__":
    main()
