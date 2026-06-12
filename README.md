# ⚙️ Mechanical Spec Analyzer (Mekanik Şartname Analizörü)

Teknik şartnameleri okuyup **yalnızca mekanik mühendisliği ilgilendiren** maddeleri yapay zeka
ile çıkaran, kategorilere ayıran ve indirilebilir rapor üreten bir Streamlit uygulaması.

Trafo/ekipman şartnamelerinde tank, kazan, kapak, sac kalınlığı, kaynak, sızdırmazlık, conta,
basınç, emniyet valfi, yağ vanaları, boya/galvaniz, taşıma donanımları, ölçü/ağırlık, mekanik
testler, kablo kutusu/mahfaza, IP sınıfı ve montaj gibi başlıklara odaklanır.

---

## Özellikler

- 📎 **Çoklu girdi:** PDF, DOCX, TXT, MD, HTML dosyası **veya** URL.
- 🌐 **Akıllı URL okuma:** Doğrudan PDF, HTML sayfası ya da HTML içindeki ilk PDF linkini bulup indirir.
- 🧩 **Chunking:** Uzun metinleri sayfa ve overlap koruyarak parçalar (`--- PAGE X ---` işaretleri).
- 🤖 **Structured output:** OpenAI ile Pydantic şemasına uygun JSON madde çıkarımı.
- 🗂️ **Kategori bazlı analiz:** 14 mekanik kategoriye otomatik gruplama.
- 📊 **Zengin sonuç ekranı:** metrik kartları, özet, kategori expander'ları, tablo, checklist, ham JSON.
- ⬇️ **Export:** JSON, Excel (çok sayfalı), Markdown ve PDF.
- 🛡️ **Dayanıklılık:** Bir chunk hata verse bile analiz durmaz; hata loglanır.

---

## Kurulum

Python 3.11+ önerilir.

```bash
cd mechanical-spec-analyzer
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## .env Ayarı

`.env.example` dosyasını `.env` olarak kopyalayın ve anahtarınızı girin:

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4.1-mini   # opsiyonel
```

> API anahtarı koda yazılmaz. `.env` dosyası `.gitignore` ile dışlanmıştır.
> Anahtarı sidebar'dan geçici olarak da girebilirsiniz (hiçbir yere kaydedilmez).

## Çalıştırma

```bash
streamlit run app.py
```

Tarayıcıda açılan arayüzde:

1. Bir dosya yükleyin **veya** URL girin.
2. Sidebar'dan model, chunk boyutu, overlap ve minimum güven filtresini ayarlayın.
3. **🔍 Şartnameyi Analiz Et** butonuna basın.
4. Sonuçları sekmelerde inceleyin: Özet, Kategoriler, Tablo, Checklist, Ham JSON.
5. **Export** sekmesinden JSON / Excel / Markdown / PDF indirin.

---

## Desteklenen Dosya Tipleri

| Tip  | Açıklama                          |
|------|-----------------------------------|
| PDF  | PyMuPDF ile metin + sayfa bilgisi |
| DOCX | python-docx (paragraf + tablo)    |
| TXT  | UTF-8 düz metin                   |
| MD   | Markdown düz metin                |
| HTML | BeautifulSoup ile temizlenmiş metin |
| URL  | PDF / HTML / HTML içindeki PDF linki |

## Çıktılar

- **JSON** — tam rapor + tüm maddeler.
- **Excel** — `Mechanical Items` ve `Checklist` sayfaları (biçimlendirilmiş).
- **Markdown** — yönetici özeti, kategori detayları, madde tablosu, checklist, eksik noktalar.
- **PDF** — reportlab ile biçimlendirilmiş rapor.

---

## Proje Yapısı

```
mechanical-spec-analyzer/
├── app.py                 # Streamlit arayüzü
├── requirements.txt
├── .env.example
├── README.md
├── src/
│   ├── config.py          # ayarlar, kategoriler, .env okuma
│   ├── document_reader.py # PDF/DOCX/TXT/MD/HTML/URL okuma
│   ├── text_chunker.py    # sayfa-duyarlı chunking
│   ├── ai_extractor.py    # Pydantic modelleri + OpenAI çıkarımı
│   ├── report_builder.py  # dedup, gruplama, özet, rapor
│   ├── exporters.py       # JSON / Excel / Markdown / PDF
│   └── utils.py           # logging, temizleme, geçici dosya
├── outputs/
└── sample_files/
```

## Bilinen Sınırlamalar

- **Taranmış (görüntü) PDF'ler** OCR içermez; metin çıkmaz. OCR'lı PDF kullanın.
- Çıkarım kalitesi seçilen modele ve şartname diline bağlıdır.
- Çok büyük belgeler çok sayıda API çağrısı (ve maliyet) gerektirir.
- URL indirme 60 MB ile sınırlıdır; kimlik doğrulama gerektiren sayfalar desteklenmez.
- AI çıktısı doğrulanmalıdır; düşük güven skorlu maddeler 🟡 ile işaretlenir.

## Mimari Notlar

- OpenAI çağrısı üç yöntemi sırayla dener: `responses.parse` → `beta.chat.completions.parse`
  → `chat.completions` + JSON mode. Böylece farklı SDK sürümlerinde çalışır.
- Yeni analiz modları `config.py` ve `ai_extractor.py` üzerinden kolayca eklenebilir.
