# 🏪 Brand & Product Analysis Pipeline

**Automatische Erkennung von Marken und Produkttypen aus Supermarktregalen**

Diese Pipeline analysiert Bilder von Supermarktregalen und identifiziert automatisch:
- 🏷️ **Marken** (Brands) wie Nivea, L'Oreal, Dove, etc.
- 📦 **Produkttypen** wie Lotion, Shampoo, Snacks, Drinks, etc.  
- 📊 **Produktvielfalt** pro Marke (wie viele verschiedene Produktarten)

## 🚀 Schnellstart

### 1. Installation

```bash
# Clone/Download der Dateien
# Erstelle Python Virtual Environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Installiere Requirements  
pip install -r requirements_brand_analysis.txt
```

### 2. Bilder vorbereiten

```bash
# Erstelle Bilder-Ordner
mkdir images

# Kopiere Ihre Supermarkt-Bilder hinein
cp /path/to/your/photos/*.jpg images/
```

**Unterstützte Formate:** JPG, PNG, WebP, HEIC

### 3. Pipeline ausführen

```bash
# Konfiguration prüfen
python run_brand_analysis.py --config-check

# Testlauf (ohne echte API-Calls)
python run_brand_analysis.py --dry-run

# Vollständige Analyse starten
python run_brand_analysis.py
```

## 📊 Ergebnisse

Die Pipeline erstellt folgende Outputs in `./brand_analysis_output/`:

### Excel-Report (`brand_type_summary.xlsx`)
- **Brand_Type_Summary**: Hauptergebnisse mit Marke, Typ und Produktanzahl
- **Cluster_Details**: Detaillierte Cluster-Informationen  
- **Top_Brands**: Ranking der Marken nach Produktvielfalt
- **Top_Product_Types**: Ranking der Produktkategorien

### Beispielbilder (`./examples/`)
- Repräsentative Bilder für jeden erkannten Cluster
- Dateibenennung: `cluster_XXX_Brand_Type_ImageID.jpg`

### Zwischenergebnisse (`./intermediate/`)
- JSON-Dateien mit detaillierten Analysedaten
- Embeddings und Clustering-Ergebnisse
- Debug-Informationen

## ⚙️ Konfiguration

Alle Parameter können in `config_brand_analysis.py` angepasst werden:

```python
# Wichtige Einstellungen
CONFIDENCE_THRESHOLD = 0.75        # Mindest-Vertrauen für Ergebnisse
DUPLICATE_SCORE_THRESHOLD = 0.8    # Schwelle für Duplikat-Erkennung  
CLUSTERING_EPS = 0.2              # Clustering-Sensitivität
ENABLE_SEGMENTATION = True         # Automatische Bild-Segmentierung
```

## 🔧 Technische Details

### Pipeline-Architektur

1. **📷 Bild-Vorbereitung**
   - Automatische Größenanpassung 
   - Optionale Segmentierung hoher Bilder
   - Format-Konvertierung

2. **🤖 Whole-Image-Analyse**
   - Google Gemini Vision API
   - Erkennung von Marken und Produkttypen
   - Konfidenz-Bewertung

3. **🔍 OCR & Logo-Erkennung**
   - Google Vision API für Texterkennung
   - Logo-Detection
   - Token-Bereinigung (Preise, etc.)

4. **🔀 Ergebnis-Fusion**
   - Kombiniert Vision + OCR Ergebnisse
   - Konfidenz-Anpassung basierend auf Übereinstimmungen
   - Widerspruchs-Erkennung

5. **🎯 Multimodale Embeddings**
   - CLIP Vision-Language Model
   - Fusion von Bild- und Text-Embeddings
   - Normalisierte Feature-Vektoren

6. **🔍 Clustering & Duplikat-Erkennung**
   - DBSCAN-Clustering ähnlicher Produkte
   - Jaccard + Cosine Similarity
   - Union-Find für Gruppierung

7. **📊 Aggregation & Export**
   - Cluster-Metadaten-Extraktion
   - Brand-Typ-Zusammenfassung
   - Excel-Export mit mehreren Sheets

### Verwendete APIs & Modelle

- **Google Gemini 1.5 Pro**: Whole-Image Vision Analysis
- **Google Vision API**: OCR + Logo Detection  
- **CLIP ViT-B/32**: Multimodale Embeddings
- **DBSCAN**: Clustering Algorithm

## 📈 Typische Ergebnisse

Für ~200 Supermarktbilder erwarten Sie:

- **50-100 verschiedene Marken** erkannt
- **20-30 Produktkategorien** identifiziert  
- **200-500 Produktarten** unterschieden
- **Verarbeitung in 10-30 Minuten** (je nach Bildanzahl)

### Beispiel-Output:

| Brand | Type | Product Kinds | Avg Confidence | Example Image |
|-------|------|---------------|----------------|---------------|
| Nivea | lotion | 8 | 0.89 | cluster_001_nivea_lotion_IMG_001.jpg |
| L'Oreal | shampoo | 5 | 0.92 | cluster_015_loreal_shampoo_IMG_045.jpg |
| Colgate | toothpaste | 3 | 0.95 | cluster_032_colgate_toothpaste_IMG_089.jpg |

## 🛠️ Troubleshooting

### Häufige Probleme:

**"Keine Bilder gefunden"**
```bash
# Prüfe Bilder-Ordner
ls -la images/
# Unterstützte Formate: .jpg, .jpeg, .png, .webp, .heic
```

**"API Key fehlt"**
```python
# In config_brand_analysis.py:
GOOGLE_API_KEY = "Ihr_Google_API_Key_hier"
```

**"Requirements fehlen"**
```bash
# Reinstall
pip install -r requirements_brand_analysis.txt

# Oder einzeln:
pip install torch sentence-transformers faiss-cpu
```

**"Zu wenige Cluster"**
```python
# In config_brand_analysis.py anpassen:
CLUSTERING_EPS = 0.15  # Kleinerer Wert = mehr Cluster
MIN_SAMPLES = 1        # Weniger restriktiv
```

**"Zu viele falsche Erkennungen"**
```python
# Höhere Schwellenwerte:
CONFIDENCE_THRESHOLD = 0.85
DUPLICATE_SCORE_THRESHOLD = 0.9
```

## 📞 Support

Bei Fragen oder Problemen:

1. **Logs prüfen**: `./brand_analysis_output/pipeline.log`
2. **Konfiguration validieren**: `python run_brand_analysis.py --config-check`
3. **Dry-Run testen**: `python run_brand_analysis.py --dry-run`
4. **Debug-Modus**: Setze `LOG_LEVEL = "DEBUG"` in der Konfiguration

## 🎯 Optimierung für Ihre Daten

### Für bessere Ergebnisse:

1. **Bildqualität**: Hohe Auflösung, gute Beleuchtung
2. **Marken-Liste erweitern**: Fügen Sie lokale Marken in `KNOWN_BRANDS` hinzu
3. **Parameter-Tuning**: Experimentieren Sie mit `CLUSTERING_EPS` und Schwellenwerten
4. **Segmentierung**: Aktivieren bei hohen Regalbillidern für bessere Abdeckung

### Skalierung:

- **Große Bildmengen**: Pipeline unterstützt Batch-Verarbeitung
- **Performance**: Nutzen Sie GPU für schnellere Embeddings (`torch.cuda`)
- **Cache**: Aktivieren Sie Caching für wiederholte Analysen

---

**Viel Erfolg mit Ihrer Supermarkt-Analyse! 🛒📊**