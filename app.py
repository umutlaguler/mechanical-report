"""Şartname Analizörü (Mekanik + Elektrik) — Streamlit uygulaması.

Çalıştırma:
    streamlit run app.py

NOT: API anahtarı ve tüm YZ ayarları (model, chunking, sıcaklık, özet) koddan
yönetilir; arayüzde son kullanıcıya açılmaz. Anahtar yalnızca .env'den okunur.
"""

from __future__ import annotations

import streamlit as st

from src.ai_extractor import (
    AIExtractionError,
    FinalReport,
    SpecItem,
    extract_items,
)
from src.config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    DISCIPLINES,
    LOW_CONFIDENCE_THRESHOLD,
    USE_AI_SUMMARY,
    UPLOADER_TYPES,
    discipline_label,
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
    build_reports,
    group_by_category,
    low_confidence_items,
)
from src.text_chunker import chunk_text
from src.utils import safe_filename, truncate

# --------------------------------------------------------------------------- #
# Sayfa ayarları + hafif stil
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Şartname Analizörü",
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
    st.session_state.analysis = None  # dict: bundle, errors, doc_chars, chunk_count


# --------------------------------------------------------------------------- #
# Sidebar (yalnızca bilgilendirme — YZ ayarı yok)
# --------------------------------------------------------------------------- #
def render_sidebar() -> dict:
    """Sidebar'ı çizer ve görüntü tercihlerini döndürür (YZ ayarı içermez)."""
    st.sidebar.header("⚙️ Bilgi")

    if get_api_key():
        st.sidebar.success("OpenAI API anahtarı bulundu (.env).")
    else:
        st.sidebar.error(
            "API anahtarı yok. Proje kökündeki `.env` dosyasına "
            "`OPENAI_API_KEY=...` ekleyin."
        )

    st.sidebar.caption(
        "Analiz kapsamı: **Mekanik + Elektrik**. Model ve analiz ayarları "
        "uygulama tarafından yönetilir."
    )

    st.sidebar.divider()
    st.sidebar.subheader("Görünüm")
    min_confidence = st.sidebar.slider(
        "Minimum güven skoru (görüntü filtresi)", 0.0, 1.0, 0.0, step=0.05,
        help="Yalnızca tabloda/gösterimde filtreler; analiz tüm maddeleri çıkarır.",
    )

    return {"min_confidence": min_confidence}


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
def run_analysis(text: str) -> None:
    """Metni chunk'lar, her disiplin için AI ile işler, raporlar üretir."""
    api_key = get_api_key()
    if not api_key:
        st.error("OpenAI API anahtarı gerekli. Proje kökündeki `.env` dosyasına ekleyin.")
        return

    model = get_default_model()

    chunks = chunk_text(text, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP)
    if not chunks:
        st.error("Belgeden işlenecek metin çıkarılamadı.")
        return

    st.info(
        f"Belge {len(chunks)} parçaya bölündü. "
        f"Disiplinler: {', '.join(discipline_label(d) for d in DISCIPLINES)}. Analiz başlıyor..."
    )
    progress_bar = st.progress(0.0)
    status = st.empty()

    total_steps = len(chunks) * len(DISCIPLINES)
    step_state = {"completed_disciplines": 0}

    def on_progress(discipline: str, done: int, total: int, found: int) -> None:
        overall = (step_state["completed_disciplines"] * total + done) / total_steps
        progress_bar.progress(min(overall, 1.0))
        status.markdown(
            f"🔄 **{discipline_label(discipline)}** — chunk **{done}/{total}** "
            f"işlendi, **{found}** madde bulundu."
        )
        if done == total:
            step_state["completed_disciplines"] += 1

    try:
        items, errors = extract_items(
            chunks,
            api_key=api_key,
            disciplines=DISCIPLINES,
            model=model,
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
            "İlgili madde bulunamadı. Belge içeriğini kontrol edin."
        )

    with st.spinner("Raporlar oluşturuluyor..."):
        bundle = build_reports(
            items,
            disciplines=DISCIPLINES,
            document_text=text,
            min_confidence=0.0,
            use_ai_summary=USE_AI_SUMMARY,
            api_key=api_key,
            model=model,
        )

    st.session_state.analysis = {
        "bundle": bundle,
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
    """Analiz sonuçlarını disiplin sekmeleri halinde gösterir."""
    analysis = st.session_state.analysis
    bundle: dict = analysis["bundle"]

    title = ""
    for report, _items in bundle.values():
        if report.document_title:
            title = report.document_title
            break

    st.divider()
    st.subheader(f"📋 {title or 'Teknik Şartname'}")

    # Üst seviye disiplin sekmeleri
    disc_tabs = st.tabs([f"{discipline_label(d)}" for d in bundle.keys()])
    for tab, (discipline, (report, items)) in zip(disc_tabs, bundle.items()):
        with tab:
            render_discipline(discipline, report, items, settings)

    st.divider()
    st.subheader("⬇️ Export")
    render_export_buttons(bundle)


def render_discipline(
    discipline: str, report: FinalReport, items: list[SpecItem], settings: dict
) -> None:
    """Tek bir disiplinin sonuçlarını alt sekmelerde gösterir."""
    min_conf = settings.get("min_confidence", 0.0)
    if min_conf > 0:
        items = [it for it in items if it.confidence >= min_conf]

    grouped = group_by_category(items, discipline)
    low = low_confidence_items(items)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{discipline_label(discipline)} Madde", len(items))
    c2.metric("Kategori", len(grouped))
    c3.metric("Ort. Güven", f"{average_confidence(items):.2f}")
    c4.metric("Düşük Güvenli", len(low))

    tab_summary, tab_cat, tab_table, tab_check, tab_json = st.tabs(
        ["📝 Özet", "🗂️ Kategoriler", "📊 Tablo", "✅ Checklist", "🧾 Ham JSON"]
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
                    st.markdown(f"- **Önem:** {it.relevance}")
                    if it.page_or_section:
                        st.markdown(f"- **Sayfa/Bölüm:** {it.page_or_section}")
                    if it.source_quote:
                        st.caption(f"“{truncate(it.source_quote, 240)}” (orijinal dil)")
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
        st.markdown("### Checklist")
        if report.checklist:
            for i, c in enumerate(report.checklist):
                st.checkbox(c, key=f"check_{discipline}_{i}", value=False)
        else:
            st.caption("Checklist boş.")

    with tab_json:
        st.json(
            {
                "report": report.model_dump(),
                "items": [it.model_dump() for it in items],
            }
        )


def render_export_buttons(bundle: dict) -> None:
    """Her iki disiplini içeren dosyalar için indirme butonları üretir."""
    title = ""
    for report, _items in bundle.values():
        if report.document_title:
            title = report.document_title
            break
    base = safe_filename(title, "sartname_rapor")

    st.markdown("### İndirilebilir Çıktılar (Mekanik + Elektrik birlikte)")
    cols = st.columns(4)

    with cols[0]:
        st.download_button(
            "⬇️ JSON",
            data=export_json(bundle),
            file_name=f"{base}.json",
            mime="application/json",
            use_container_width=True,
        )

    with cols[1]:
        try:
            st.download_button(
                "⬇️ Excel",
                data=export_excel(bundle),
                file_name=f"{base}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Excel oluşturulamadı: {exc}")

    with cols[2]:
        st.download_button(
            "⬇️ Markdown",
            data=export_markdown(bundle).encode("utf-8"),
            file_name=f"{base}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with cols[3]:
        try:
            st.download_button(
                "⬇️ PDF",
                data=export_pdf(bundle),
                file_name=f"{base}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as exc:  # noqa: BLE001
            st.warning(f"PDF oluşturulamadı: {exc}")


# --------------------------------------------------------------------------- #
# Ana akış
# --------------------------------------------------------------------------- #
def main() -> None:
    settings = render_sidebar()

    st.title("⚙️ Şartname Analizörü — Mekanik + Elektrik")
    st.markdown(
        "Teknik şartnamelerinizi yükleyin; uygulama **mekanik ve elektrik mühendisliğini "
        "ilgilendiren** tüm maddeleri yapay zeka ile çıkarır, kategorilere ayırır ve "
        "indirilebilir rapor üretir. Şartname **herhangi bir dilde** olabilir; rapor Türkçe üretilir."
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
            run_analysis(text)

    if st.session_state.analysis is not None:
        render_results(settings)
    else:
        st.caption(
            "Henüz analiz yapılmadı. Bir dosya yükleyin veya URL girin, "
            "ardından **Şartnameyi Analiz Et**'e basın."
        )


if __name__ == "__main__":
    main()
