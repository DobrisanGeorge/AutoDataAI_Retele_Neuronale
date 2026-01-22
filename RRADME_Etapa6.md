# README – Etapa 6: Analiza Performanței, Optimizarea și Concluzii Finale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Dobrisan Andrei George  
**Link Repository GitHub:** [Adaugă Link-ul Tău Aici]  
**Data predării:** 15.01.2026

---
## Scopul Etapei 6

Această etapă corespunde punctelor **7. Analiza performanței și optimizarea parametrilor**, **8. Analiza și agregarea rezultatelor** și **9. Formularea concluziilor finale** din lista de 9 etape.

**Obiectiv principal:** Maturizarea completă a Sistemului cu Inteligență Artificială (SIA) prin trecerea la o arhitectură avansată (EfficientNet), implementarea detecției multiple (Multi-Label) și finalizarea aplicației software pentru scenarii industriale.

---

## PREREQUISITE – Verificare Etapa 5 (OBLIGATORIU)

**Înainte de a începe Etapa 6, verificați că aveți din Etapa 5:**

- [x] **Model antrenat** (versiunea MobileNetV2 inițială)
- [x] **Metrici baseline** raportate: Accuracy ~70%
- [x] **Tabel hiperparametri** completat
- [x] **UI funcțional** (versiunea dark mode)
- [x] **State Machine** implementat

---

## Cerințe

### 1. Actualizarea Aplicației Software în Etapa 6 

**CERINȚĂ CENTRALĂ:** Trecerea de la un clasificator simplu la un sistem expert de evaluare a daunelor.

### Tabel Modificări Aplicație Software

| **Componenta** | **Stare Etapa 5 (Baseline)** | **Modificare Etapa 6 (Final)** | **Justificare** |
|----------------|-------------------|------------------------|-----------------|
| **Arhitectură Model** | MobileNetV2 (Transfer Learning) | **EfficientNetB0** (Fine-Tuning) | Capacitate superioară de extragere a texturilor fine (zgârieturi). Accuracy +12%. |
| **Tip Problemă** | Multi-Class (Softmax) | **Multi-Label (Sigmoid)** | O mașină poate avea simultan "Far Spart" și "Bară Zgâriată". Softmax suprima daunele secundare. |
| **Rezoluție Input** | 224 x 224 px | **260 x 260 px** | Rezoluția nativă EfficientNetB0 permite observarea detaliilor mici. |
| **Business Logic** | Doar afișare clasă | **Calcul Severitate & Daună Totală** | Implementarea unui algoritm care însumează scorurile daunelor pentru a recomanda "Total Loss". |
| **UI / UX** | Streamlit Standard | **Enterprise Dark Mode** | Contrast maxim pentru vizibilitate în condiții de service/atelier. |
| **Modularizare** | Preprocesare în `app.py` | `src/preprocessing/transformers.py` | Cod curat, reutilizabil între antrenare și inferență. |

---

### 2. Tabel Experimente de Optimizare

Am realizat o serie de experimente pentru a crește performanța de la baseline-ul din Etapa 5.

| **Exp#** | **Modificare față de Baseline (MobileNetV2)** | **Accuracy** | **F1-score** | **Timp antrenare** | **Observații** |
|----------|------------------------------------------|--------------|--------------|-------------------|----------------|
| **Baseline** | MobileNetV2, 224px, Softmax | 0.72 | 0.68 | 15 min | Model rapid, dar ratează detalii fine și daune multiple. |
| **Exp 1** | MobileNetV2 + Class Weights | 0.75 | 0.71 | 15 min | A ajutat la echilibrarea claselor rare (ex: `glass_shatter`). |
| **Exp 2** | EfficientNetB0 (Frozen), 260px | 0.79 | 0.76 | 25 min | Arhitectura mai bună a adus un salt major în performanță. |
| **Exp 3** | Exp 2 + Augmentare Agresivă (Zoom 0.3) | 0.81 | 0.78 | 28 min | Modelul a devenit mai robust la poze necentrate. |
| **Exp 4** | **EfficientNetB0 + Fine-Tuning + Multi-Label** | **0.88** | **0.85** | **45 min** | **BEST.** Dezghețarea ultimelor 20 straturi și trecerea la Binary Crossentropy a permis detecția simultană. |

**Justificare alegere configurație finală (Exp 4):**
Am ales **EfficientNetB0 cu Fine-Tuning și Sigmoid Output** deoarece în contextul asigurărilor este critic să nu omitem nicio daună. Trecerea la Multi-Label ne permite să identificăm toate avariile dintr-o poză, iar Fine-Tuning-ul a adaptat filtrele convoluționale pentru texturi specifice de metal și sticlă spartă, nu doar forme generice.

---

### 3. Analiza Detaliată a Performanței

### 3.1 Interpretare Confusion Matrix

*(Deoarece folosim Multi-Label, matricea de confuzie se analizează per clasă sau global)*

**Clase cu performanță ridicată:**
* **Head Lamp / Tail Lamp:** Features foarte distincte (sticlă colorată, reflexii, contrast mare față de caroserie). Precision > 90%.
* **Glass Shatter:** Textura de "pânză de păianjen" este unică și ușor de învățat de EfficientNet.

**Clase cu performanță slabă:**
* **Bumper Scratch vs Door Scratch:** Modelul confundă uneori locația. Deși detectează corect "Scratch", greșește elementul de caroserie dacă poza este prea zoomed-in (nu vede conturul ușii).
* **Dent (Îndoitură):** Defectele care depind doar de deformarea reflexiei luminii sunt greu de detectat în poze 2D statice fără informație de adâncime.

### 3.2 Analiza a 5 Exemple Greșite (Failure Analysis)

| **Index** | **True Label** | **Predicted** | **Confidence** | **Cauză probabilă** | **Soluție propusă** |
|-----------|----------------|---------------|----------------|---------------------|---------------------|
| #1 | `door_scratch` | `reflection` (False Negative) | 0.25 | Zgârietura este foarte fină și se suprapune cu reflexia unui copac. | Augmentare cu zgomot și variații de contrast pentru a învăța diferența textură/reflexie. |
| #2 | `bumper_dent` | `bumper_scratch` | 0.65 | Îndoitura are vopseaua sărită, arătând ca o zgârietură mare. | Etichetare mai precisă (Multi-label: ambele sunt corecte tehnic). |
| #3 | `head_lamp` | `no_damage` | 0.40 | Farul este doar crăpat fin, nu spart complet. | Colectare date cu "hairline cracks" (fisuri fine). |
| #4 | `glass_shatter` | `interior` | 0.55 | Poza este făcută din interiorul mașinii spre exterior. | Augmentare cu Flip Orizontal/Vertical și poze din interior în training. |
| #5 | `mud/dirt` | `bumper_scratch` | 0.72 | Noroiul uscat pe bară seamănă cu o zonă de impact. | Antrenarea unei clase negative "Dirty/Mud" pentru a reduce FP. |

---

### 4. Concluzii Finale și Lecții Învățate

### 4.1 Evaluare sintetică

Proiectul **AutoClaim AI** a reușit să demonstreze viabilitatea utilizării Deep Learning pentru automatizarea constatărilor auto.
* **Obiectiv atins:** Sistemul poate clasifica corect daunele majore și medii cu o acuratețe de ~85-88%.
* **Inovație:** Implementarea algoritmului de "Daună Totală" pe baza unui scor de severitate ponderat oferă valoare de business imediată.

### 4.2 Limitări Identificate

1.  **Dependența de lumină:** Performanța scade drastic în poze de noapte sau în garaje subterane slab iluminate.
2.  **Context spațial:** Modelul nu știe "unde" este dauna pe mașină (nu face Object Detection / Bounding Box), ci doar "că există". Pentru un deviz precis, ar fi nevoie de localizare (YOLO).
3.  **Murdăria:** Mașinile murdare generează alarme false (False Positives).

### 4.3 Lecții Învățate

1.  **Data is King:** Trecerea de la 2000 la 4000 de imagini (prin generare sintetică în Etapa 4) a avut un impact mai mare asupra performanței decât ajustarea hiperparametrilor.
2.  **Arhitectura contează:** MobileNetV2 este bun pentru mobil, dar EfficientNetB0 este mult mai capabil să vadă texturi fine (zgârieturi) necesare în acest domeniu.
3.  **Multi-Label este obligatoriu:** În lumea reală, accidentele nu produc o singură daună izolată. Trecerea la `Sigmoid` a fost decizia arhitecturală cheie.

### 4.4 Plan Post-Feedback (Înainte de Examen)

1.  Re-antrenare finală cu toate datele disponibile (inclusiv setul de validare) pentru maximizarea performanței.
2.  Curățarea codului și adăugarea de comentarii explicative (docstrings) în toate modulele.
3.  Exportarea modelului și testarea pe un dispozitiv mobil (opțional, dacă timpul permite).

---

## Structura Repository-ului la Finalul Etapei 6

```bash

AutoDataAI/

├── data/
│   ├── processed/
│   ├── raw/                                # Dataset original + Sintetic
│   ├── train/                              # 70%
│   ├── validation/                         # 15%
│   └── test/                               # 15%
├── docs/
│   ├── datasets/
│   ├── state_machine.png
│   └── screenshots/                        # Dovezi funcționare UI
├── models/
│   ├── damage_model.h5                     # Model Optimizat (EfficientNet)
│   └── classes.txt
├── results/
│   ├── training_history.csv
│   ├── test_metrics.json
│   └── training_plot_pro.png               # Grafice antrenare
├── src/
│   ├── data_acquisition/
│   │   ├── organize_dataset.py
│   │   ├── split_data.py
│   │   └── generate_synthetic_data.py      # Generator date sintetice
│   ├── preprocessing/
│   │   └── transformers.py                 # Modul preprocesare partajat
│   ├── neural_network/
│   │   ├── train_model.py                  # Script antrenare EfficientNet
│   │   └── train_model_bck.py
│   └── app/
│       ├── app.py                          # Interfața Enterprise Finală
│       └── app_bck.py
├── .gitignore
├── requirements.txt
├── README.md                               # Documentație Generală
├── README_Etapa3
├── README_Etapa4
├── README_Etapa5
└── README_Etapa6_Optimizare_Concluzii.md   # Acest fișier

```