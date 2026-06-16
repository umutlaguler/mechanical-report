# ⚙️ Şartname Analizörü (Mekanik + Elektrik)

Teknik şartnameleri okuyup **hem mekanik hem elektrik mühendisliğini ilgilendiren** maddeleri
yapay zeka ile çıkaran, kategorilere ayıran ve indirilebilir rapor üreten bir Streamlit uygulaması.

Trafo/ekipman şartnamelerinde:

- **Mekanik:** tank, kazan, kapak, sac kalınlığı, kaynak, sızdırmazlık, conta, basınç, emniyet valfi,
  yağ vanaları, boya/galvaniz, taşıma donanımları, ölçü/ağırlık, mekanik testler, kablo kutusu mahfazası,
  IP sınıfı ve montaj.
- **Elektrik:** anma gücü/gerilim sınıfı, sargı bağlantı grubu, OLTC/kademe değiştirici, BIL/yalıtım seviyesi,
  bushing/izolatör elektriksel değerleri, kablo kutusu/busbar/terminaller, koruma/topraklama, kayıplar/empedans,
  sensör/izleme donanımı, elektriksel testler.

Her çalıştırmada **mekanik ve elektrik ayrı bölümler** halinde analiz edilir.

---

## Özellikler

- 📎 **Çoklu girdi:** PDF, DOCX, TXT, MD, HTML dosyası **veya** URL.
- 🌍 **Çok dillilik:** Şartname herhangi bir dilde olabilir (TR/EN dışında Almanca, Fransızca, vb.).
  Dil otomatik algılanır; **rapor Türkçe** üretilir, `source_quote` orijinal dilde korunur.
- 📚 **Domain notları:** `model için notlar.xlsx` dosyasındaki kurum içi geri bildirimler (ekipman + eş
  anlamlılar + talimat) modele otomatik enjekte edilir.
- 🧩 **Chunking:** Uzun metinleri sayfa ve overlap koruyarak parçalar (`--- PAGE X ---` işaretleri).
- 🤖 **Structured output:** OpenAI ile Pydantic şemasına uygun JSON madde çıkarımı.
- 🗂️ **Kategori bazlı analiz:** Mekanik ve elektrik için ayrı kategori setlerine otomatik gruplama.
- 📊 **Zengin sonuç ekranı:** Mekanik/Elektrik sekmeleri; her birinde metrik kartları, özet, kategori
  expander'ları, tablo, checklist, ham JSON.
- ⬇️ **Export:** JSON, Excel (disiplin başına sayfa), Markdown ve PDF — her biri iki disiplini de içerir.
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

> **API anahtarı ve tüm YZ ayarları yalnızca koddan/.env'den yönetilir; arayüzde son kullanıcıya
> gösterilmez.** `.env` dosyası `.gitignore` ile dışlanmıştır.

## Model için domain notları (`model için notlar.xlsx`)

Kurum içi geri bildirimler bu Excel'den okunur ve modele enjekte edilir. Yapı:

| Ekipman | Eş Anlamlılar | Not |
|---------|---------------|-----|
| HV Bushing | High Voltage Bushing;Porcelain Bushing;... | HV bushing talebi kontrol edilmelidir... |

- `Eş Anlamlılar` `;` ile ayrılır. `Not` "neyi kontrol et / nasıl raporla" talimatıdır.
- **Disiplin** belirleme (esnek):
  - Sayfa başlığı / üst hücre "MEKANİK..." → mekanik, "ELEKTRİK..." → elektrik; **veya**
  - Tabloya bir `Disiplin` sütunu ekleyin (mekanik/elektrik); **veya**
  - Ayrı bir sayfa açın (ör. adı "Elektrik" olan sayfa).
- `KAPAK` gibi tek hücreli satırlar **bölüm başlığı** sayılır.
- Elektrik notları eklemek için: yeni satırları `Disiplin=elektrik` ile ya da "ELEKTRİK İÇİN NOTLAR"
  başlıklı yeni bir sayfada girin. Kod otomatik okur (yeniden başlatma yeterli).

## Çalıştırma

```bash
streamlit run app.py
```

Tarayıcıda açılan arayüzde:

1. Bir dosya yükleyin **veya** URL girin.
2. **🔍 Şartnameyi Analiz Et** butonuna basın.
3. Sonuçları **Mekanik** ve **Elektrik** sekmelerinde inceleyin: Özet, Kategoriler, Tablo, Checklist, Ham JSON.
4. **Export** bölümünden JSON / Excel / Markdown / PDF indirin (her biri iki disiplini içerir).

> Arayüzde model/chunking gibi YZ ayarı yoktur; bunlar `src/config.py` içinde yönetilir.

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

- **JSON** — `disciplines` ağacı altında her disiplin için rapor + maddeler.
- **Excel** — disiplin başına `… Maddeler` ve `… Checklist` sayfaları (biçimlendirilmiş).
- **Markdown** — Mekanik ve Elektrik bölümleri; her birinde özet, kategori detayları, tablo, checklist.
- **PDF** — reportlab ile iki bölümlü biçimlendirilmiş rapor.

---

## Proje Yapısı

```
mechanical-spec-analyzer/
├── app.py                 # Streamlit arayüzü (Mekanik/Elektrik sekmeleri)
├── requirements.txt
├── .env.example
├── README.md
├── model için notlar.xlsx # kurum içi domain notları (modele enjekte edilir)
├── src/
│   ├── config.py          # disiplinler, kategoriler, YZ sabitleri, .env okuma
│   ├── notes_loader.py    # Excel domain notlarını okuma + prompt bloğu
│   ├── document_reader.py # PDF/DOCX/TXT/MD/HTML/URL okuma
│   ├── text_chunker.py    # sayfa-duyarlı chunking
│   ├── ai_extractor.py    # Pydantic modelleri + disiplin promptları + OpenAI çıkarımı
│   ├── report_builder.py  # dedup, gruplama, özet, disiplin başına rapor
│   ├── exporters.py       # JSON / Excel / Markdown / PDF (iki disiplinli)
│   └── utils.py           # logging, temizleme, geçici dosya
├── outputs/
└── sample_files/
```

## Bilinen Sınırlamalar

- **Taranmış (görüntü) PDF'ler** OCR içermez; metin çıkmaz. OCR'lı PDF kullanın.
- İki disiplin = chunk başına iki API çağrısı; büyük belgelerde maliyet artar.
- Çıkarım kalitesi seçilen modele ve şartname diline bağlıdır.
- URL indirme 60 MB ile sınırlıdır; kimlik doğrulama gerektiren sayfalar desteklenmez.
- AI çıktısı doğrulanmalıdır; düşük güven skorlu maddeler 🟡 ile işaretlenir.

## Mimari Notlar

- OpenAI çağrısı üç yöntemi sırayla dener: `responses.parse` → `beta.chat.completions.parse`
  → `chat.completions` + JSON mode. Böylece farklı SDK sürümlerinde çalışır.
- Yeni disiplinler `config.py` (`DISCIPLINES`, kategoriler) ve `ai_extractor.py` (prompt) üzerinden eklenebilir.
- Geriye dönük uyumluluk: `MechanicalItem`/`FinalMechanicalReport` isimleri `SpecItem`/`FinalReport`
  için alias olarak korunur.
