# 1. Identificare Proiect

| Câmp | Valoare |
|------|---------|
| **Student** | **Dobrisan Andrei George** |
| **Grupa / Specializare** | **[Completeaza Aici Grupa]** / Informatică Industrială |
| **Disciplina** | Rețele Neuronale |
| **Instituție** | POLITEHNICA București – FIIR |
| **Link Repository GitHub** | **[Adauga Aici Link-ul Tau]** |
| **Acces Repository** | Public |
| **Stack Tehnologic** | Python (TensorFlow/Keras, Streamlit, OpenCV) |
| **Domeniul Industrial de Interes (DII)** | Automotive / Asigurări Auto |
| **Tip Rețea Neuronală** | **CNN (EfficientNetB0)** + Multi-Label Classification |

### Rezultate Cheie (Versiunea Finală vs Etapa 6)

| Metric | Țintă Minimă | Rezultat Etapa 6 | Rezultat Final | Îmbunătățire | Status |
|--------|--------------|------------------|----------------|--------------|--------|
| Accuracy (Test Set) | ≥70% | 88.40% | **88.40%** | +16.4% (vs Baseline) | [✓] |
| F1-Score (Macro) | ≥0.65 | 0.85 | **0.85** | +0.17 (vs Baseline) | [✓] |
| Latență Inferență | < 100 ms | 35 ms | **35 ms** | -13 ms | [✓] |
| Contribuție Date Originale | ≥40% | 50% | **50%** | - | [✓] |
| Nr. Experimente Optimizare | ≥4 | 4 | **4** | - | [✓] |

### Declarație de Originalitate & Politica de Utilizare AI

**Acest proiect reflectă munca, gândirea și deciziile mele proprii.**

Utilizarea asistenților de inteligență artificială (ChatGPT, Claude, Grok, GitHub Copilot etc.) este **permisă și încurajată** ca unealtă de dezvoltare – pentru explicații, generare de idei, sugestii de cod, debugging, structurarea documentației sau rafinarea textelor.

**Nu este permis** să preiau:
- cod, arhitectură RN sau soluție luată aproape integral de la un asistent AI fără modificări și raționamente proprii semnificative,
- dataset-uri publice fără contribuție proprie substanțială (minimum 40% din observațiile finale – conform cerinței obligatorii Etapa 4),
- conținut esențial care nu poartă amprenta clară a propriei mele înțelegeri.

**Confirmare explicită (bifez doar ce este adevărat):**

| Nr. | Cerință | Confirmare |
|-----|-------------------------------------------------------------------------|------------|
| 1 | Modelul RN a fost antrenat **de la zero** (weights inițializate random sau fine-tuning pe arhitectură standard, **NU** model gata descărcat) | [x] DA |
| 2 | Minimum **40% din date sunt contribuție originală** (generate/achiziționate/etichetate de mine) | [x] DA |
| 3 | Codul este propriu sau sursele externe sunt **citate explicit** în Bibliografie | [x] DA |
| 4 | Arhitectura, codul și interpretarea rezultatelor reprezintă **muncă proprie** (AI folosit doar ca tool, nu ca sursă integrală de cod/dataset) | [x] DA |
| 5 | Pot explica și justifica **fiecare decizie importantă** cu argumente proprii | [x] DA |

**Semnătură student (prin completare):** *Dobrisan Andrei George*

---

## 2. Descrierea Nevoii și Soluția SIA

### 2.1 Nevoia Reală / Studiul de Caz

În industria asigurărilor auto, evaluarea daunelor este un proces manual, lent și subiectiv. Inspectorii trebuie să analizeze mii de fotografii zilnic, ceea ce duce la erori umane, inconsecvență în estimări și întârzieri majore pentru clienți. Există o nevoie critică de automatizare a triajului inițial pentru a separa cazurile de "Daună Totală" de cele reparabile, reducând timpul de așteptare de la zile la secunde.

### 2.2 Beneficii Măsurabile Urmărite

1. **Reducerea timpului de triaj:** De la 15-30 minute (uman) la < 1 secundă (AI).
2. **Consistență:** Eliminarea subiectivității umane în clasificarea severității daunelor (acuratețe țintă >85%).
3. **Prioritizare automată:** Detectarea instantanee a cazurilor "Total Loss" pentru a accelera procesarea dosarelor critice.
4. **Reducerea costurilor operaționale:** Scăderea numărului de inspecții fizice necesare pentru daunele minore cu 40%.

### 2.3 Tabel: Nevoie → Soluție SIA → Modul Software

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul** | **Modul software responsabil** | **Metric măsurabil** |
|---------------------------|--------------------------|--------------------------------|----------------------|
| Triaj rapid al dosarelor de daună | Clasificare automată a imaginilor încărcate de client | **Neural Network (EfficientNet)** | Timp inferență < 50ms |
| Estimarea costului reparației | Identificarea tipului de avarie (ex: Far spart vs Zgârietură) | **Model Multi-Label** | Acuratețe > 85% |
| Alertă rapidă pentru epave | Calcul automat al unui "Scor de Severitate" | **Web Service / UI Logic** | Recall > 90% pe clase critice |

---

## 3. Dataset și Contribuție Originală

### 3.1 Sursa și Caracteristicile Datelor

| Caracteristică | Valoare |
|----------------|---------|
| **Origine date** | Mixt: Dataset Public (Kaggle) + Generare Sintetică Proprie |
| **Sursa concretă** | Kaggle "Car Damage Assessment" + Script propriu de augmentare fizică |
| **Număr total observații finale (N)** | ~4,000 imagini |
| **Număr features** | Imagine RGB (260 x 260 px) - Rezoluție optimizată EfficientNet |
| **Tipuri de date** | Imagini nestructurate |
| **Format fișiere** | JPG / PNG |
| **Perioada colectării/generării** | Noiembrie 2025 - Ianuarie 2026 |

### 3.2 Contribuția Originală (minim 40% OBLIGATORIU)

| Câmp | Valoare |
|------|---------|
| **Total observații finale (N)** | 4000 |
| **Observații originale (M)** | 2000 |
| **Procent contribuție originală** | **50%** |
| **Tip contribuție** | Date sintetice prin metode avansate (Augmentare fizică) |
| **Locație cod generare** | `src/data_acquisition/generate_synthetic_data.py` |
| **Locație date originale** | `data/raw/` (fișiere prefixate cu `syn_`) |

**Descriere metodă generare/achiziție:**

Am dezvoltat un script Python (`generate_synthetic_data.py`) care simulează condiții reale, dificile, de captură a imaginilor, specifice clienților care fac poze cu telefonul în grabă. Scriptul nu aplică doar rotații simple, ci transformări fizice relevante:
* **Noise Injection (Sare și Piper):** Simulează senzorii camerelor slabe în lumină scăzută (ISO mare).
* **Motion Blur:** Simulează mișcarea mâinii în timpul fotografierii.
* **Brightness Shift:** Simulează supraexpunerea (soare puternic) sau subexpunerea (garaj).
Aceste date au dublat dimensiunea setului de date și au crescut robustețea modelului cu ~15%.

### 3.3 Preprocesare și Split Date

| Set | Procent | Număr Observații |
|-----|---------|------------------|
| Train | 70% | ~2800 |
| Validation | 15% | ~600 |
| Test | 15% | ~600 |

**Preprocesări aplicate:**
- **Resize inteligent (LANCZOS):** Redimensionare la **260x260** (input nativ EfficientNetB0) păstrând aspect ratio.
- **Normalizare:** Specifică EfficientNet (range-uri interne Keras `preprocess_input`).
- **Structură:** Organizare în foldere pe clase (`No_Damage`, `Minor`, `Major`, etc.).

**Referințe fișiere:** `src/preprocessing/transformers.py`, `src/data_acquisition/split_data.py`

---

## 4. Arhitectura SIA și State Machine

### 4.1 Cele 3 Module Software

| Modul | Tehnologie | Funcționalitate Principală | Locație în Repo |
|-------|------------|---------------------------|-----------------|
| **Data Acquisition** | Python (OpenCV) | Generare date sintetice și organizare dataset | `src/data_acquisition/` |
| **Neural Network** | TensorFlow / Keras | Model EfficientNetB0 Multi-Label antrenat | `src/neural_network/` |
| **Web Service / UI** | Streamlit | Interfață Enterprise cu calcul severitate | `src/app/` |

### 4.2 State Machine

**Locație diagramă:** `docs/state_machine.png`

**Stări principale și descriere:**

| Stare | Descriere | Condiție Intrare | Condiție Ieșire |
|-------|-----------|------------------|-----------------|
| `IDLE` | Așteptare utilizator | Start aplicație | Upload imagine |
| `PRE-PROCESARE` | Resize 260px și normalizare | Imagine validă | Tensor (1,260,260,3) |
| `INFERENȚĂ` | Rulare model EfficientNetB0 | Tensor disponibil | Vector probabilități |
| `CALCUL SEVERITATE` | Algoritm ponderare daune | Predicții Multi-Label | Scor Severitate (0-20) |
| `AFIȘARE REZULTAT` | Afișare deviz estimativ | Scor < 12 | Reset |
| `ALERTĂ TOTALĂ` | Card roșu și notificare | Scor >= 12 | Reset |

**Justificare alegere arhitectură State Machine:**

Am ales o arhitectură bazată pe **procesare secvențială cu ramificare decizională finală**. Motivul este natura critică a aplicației: înainte de a afișa orice rezultat utilizatorului, sistemul trebuie să calculeze un "Scor de Severitate" compus. Nu este suficientă o simplă clasificare; sistemul trebuie să decidă dacă vehiculul este "Daună Totală" (Total Loss) pe baza unei sume ponderate a daunelor detectate (ex: Airbag sărit + Lonjeron îndoit = Scor Critic).

### 4.3 Actualizări State Machine în Etapa 6

| Componentă Modificată | Valoare Etapa 5 | Valoare Etapa 6 | Justificare Modificare |
|----------------------|-----------------|-----------------|------------------------|
| **Tip Model** | Single-Class (Softmax) | **Multi-Label (Sigmoid)** | Realitatea implică daune multiple simultan. |
| **Logicǎ Decizie** | Threshold simplu (0.5) | **Calcul Severitate Ponderat** | Nevoie de business pentru triaj "Total Loss". |
| **Stare nouă** | N/A | `CALCUL SEVERITATE` | Agregarea rezultatelor multiple într-un singur scor. |

---

## 5. Modelul RN – Antrenare și Optimizare

### 5.1 Arhitectura Rețelei Neuronale

Input (260, 260, 3) → EfficientNetB0 (Pre-trained ImageNet, Frozen except last 20 layers) → GlobalAveragePooling2D → BatchNormalization → Dropout(0.5) → Dense(256, ReLU) → Dropout(0.5) → Dense(num_classes, Sigmoid) <-- Multi-Label Head Output: Vector probabilități independente per clasă

**Justificare alegere arhitectură:**

Am trecut de la MobileNetV2 (Etapa 5) la **EfficientNetB0**. Deși MobileNet este rapid, EfficientNet oferă un balans mult mai bun între acuratețe și parametri, fiind capabil să extragă texturi fine (zgârieturi) pe care modelele mai simple le ratau. Am folosit activarea **Sigmoid** în ultimul strat pentru a permite **Multi-Label Classification** (o mașină poate avea *simultan* bara zgâriată și farul spart), spre deosebire de Softmax care forța o singură clasă.

### 5.2 Hiperparametri Finali (Model Optimizat - Etapa 6)

| Hiperparametru | Valoare Finală | Justificare Alegere |
|----------------|----------------|---------------------|
| Learning Rate | 0.0001 (1e-4) | LR mic pentru fine-tuning stabil al straturilor pre-antrenate. |
| Batch Size | 16 | EfficientNet necesită memorie VRAM mare; 16 asigură stabilitate. |
| Epochs | 12 (cu Early Stopping) | Previne overfitting-ul; modelul a convers rapid. |
| Optimizer | Adam | Standardul industriei pentru convergență rapidă. |
| Loss Function | Binary Crossentropy | Obligatoriu pentru Multi-Label (tratează fiecare clasă independent). |
| Regularizare | Dropout 0.5 | Valoare ridicată pentru a forța rețeaua să învețe trăsături robuste. |

### 5.3 Experimente de Optimizare (minim 4 experimente)

| Exp# | Modificare față de Baseline | Accuracy | F1-Score | Timp Antrenare | Observații |
|------|----------------------------|----------|----------|----------------|------------|
| **Baseline** | MobileNetV2, 224px | 72.0% | 0.68 | 15 min | Rapid, dar ratează detalii fine. |
| Exp 1 | EfficientNetB0, LR 0.001 | 75.4% | 0.71 | 20 min | Overfitting rapid din cauza LR mare. |
| Exp 2 | EfficientNetB0, Dropout 0.3 | 82.1% | 0.79 | 22 min | Performanță bună, dar încă instabil. |
| **FINAL** | **EfficientNetB0, LR 1e-4, Drop 0.5** | **88.4%** | **0.85** | **35 min** | **Optim.** Dropout mare și LR mic au stabilizat fine-tuning-ul. |

**Justificare alegere model final:**
Configurația finală a oferit cel mai bun echilibru. Deși timpul de antrenare a crescut, saltul de acuratețe de la 72% la 88% este critic pentru o aplicație industrială unde erorile costă bani. Dropout-ul de 0.5 a fost esențial pentru generalizarea pe setul de test.

**Referințe fișiere:** `results/optimization_experiments.csv`, `models/optimized_model.h5`

---

## 6. Performanță Finală și Analiză Erori

### 6.1 Metrici pe Test Set (Model Optimizat)

| Metric | Valoare | Target Minim | Status |
|--------|---------|--------------|--------|
| **Accuracy** | **88.40%** | ≥70% | [✓] |
| **F1-Score (Macro)** | **0.85** | ≥0.65 | [✓] |
| **Test Loss** | **0.3120** | - | - |

**Îmbunătățire față de Baseline (Etapa 5):**

| Metric | Etapa 5 (Baseline) | Etapa 6 (Optimizat) | Îmbunătățire |
|--------|-------------------|---------------------|--------------|
| Accuracy | 72.00% | 88.40% | **+16.4%** |
| F1-Score | 0.68 | 0.85 | **+0.17** |

**Referință fișier:** `results/final_metrics.json`

### 6.2 Confusion Matrix

**Locație:** `docs/confusion_matrix_optimized.png`

**Interpretare:**
* **Performanță Top:** Clasele "Glass Shatter" și "Head Lamp" sunt detectate cu precizie >90% datorită texturilor unice (sticlă spartă, reflexii).
* **Puncte Slabe:** Există o ușoară confuzie între "Bumper Scratch" și "Door Scratch" (modelul recunoaște zgârietura, dar uneori greșește elementul de caroserie dacă poza este prea zoomed-in).

### 6.3 Analiza Top 5 Erori (Failure Analysis)

Date extrase din `results/error_analysis.csv`:

| # | Input | Predicție RN | Clasă Reală | Cauză Probabilă | Implicație Industrială |
|---|-------|--------------|-------------|-----------------|------------------------|
| 1 | `img_402.jpg` (Vopsea neagră) | `Scratch` | `Dent` | Reflexia luminii pe vopsea neagră interpretată ca zgârietură. | Deviz incorect (vopsire vs îndreptare). |
| 2 | `img_115.jpg` (Parbriz) | `No Damage` | `Glass Shatter` | Fisura era foarte fină ("hairline") și rezoluția a pierdut detaliul. | Daună neplătită clientului. |
| 3 | `syn_99.jpg` (Zoom mare) | `Bumper` | `Door` | Augmentarea cu Zoom a tăiat contextul vizual (roata/mânerul). | Identificare greșită a piesei. |
| 4 | `img_003.jpg` (Murdar) | `Smudge` | `Scratch` | Mașina era murdară; modelul a crezut că zgârietura e noroi. | Respingere eronată dosar. |
| 5 | `img_88.jpg` (Stop spate) | `Tail Lamp` | `Head Lamp` | Formă similară, confuzie cauzată de filtrul de culoare roșie. | Comandă piesă greșită. |

### 6.4 Validare în Context Industrial

Rezultatele indică faptul că modelul este **viabil pentru triaj asistat**. Cu o acuratețe de 88%, sistemul poate automatiza aprobarea daunelor evidente, lăsând cazurile ambigue (cele 12% erori) pentru inspectorii umani. Aceasta ar reduce volumul de muncă manuală cu aproximativ 70-80%.

---

## 7. Aplicația Software Finală

### 7.1 Modificări Implementate în Etapa 6

| Componentă | Stare Etapa 5 | Modificare Etapa 6 | Justificare |
|------------|---------------|-------------------|-------------|
| **Model încărcat** | `trained_model.h5` | `optimized_model.h5` | Acuratețe superioară (+16.4%). |
| **Tip Detecție** | Single-Class | **Multi-Label** | Realism: daunele nu sunt mutu-exclusive. |
| **UI Logic** | Afișare Clasă | **Calcul Severitate** | Nevoie de business (detectare Total Loss). |
| **Design** | Basic | **Enterprise Dark Mode** | Vizibilitate mai bună în service-uri. |

### 7.2 Screenshot UI cu Model Optimizat

**Locație:** `docs/screenshots/inference_optimized.png`

Screenshot-ul demonstrează interfața finală rulând modelul optimizat. Se observă imaginea încărcată, lista de daune detectate cu scorurile de încredere și, dacă este cazul, alerta roșie de "DAUNĂ TOTALĂ".

### 7.3 Demonstrație Funcțională End-to-End

**Locație dovadă:** `docs/demo/` (sau capturi în `docs/screenshots`)

**Fluxul demonstrat:**
1.  **Input:** Utilizatorul încarcă o imagine cu o mașină lovită frontal.
2.  **Procesare:** Sistemul redimensionează imaginea și rulează EfficientNetB0.
3.  **Inferență:** Modelul detectează: `Head Lamp (98%)`, `Bumper Dent (85%)`.
4.  **Decizie:** Scorul de severitate calculat este 14 (>12).
5.  **Output:** UI afișează Cardul Roșu "DAUNĂ TOTALĂ".

---

## 8. Structura Repository-ului Final

```bash

CAR-DAMAGE-ASSESSMENT-RETELE-NEURONALE/
│
├── DOBRISAN_Andrei_George_633AB_README_Proiect_RN.md   # ← ACEST FIȘIER (Final)
├── README.md                                           # README General
├── README_Etapa3.md                                    # Documentație Etapa 3
├── README_Etapa4.md                                    # Documentație Etapa 4
├── README_Etapa5.md                                    # Documentație Etapa 5
├── README_Etapa6.md                                    # Documentație Etapa 6
│
├── config/
│   └── optimized_config.yaml                           # Configurare model final
│
├── data/
│   ├── processed/                                      # Date procesate
│   ├── raw/                                            # Date originale + sintetice
│   ├── test/                                           # Set Test (15%)
│   ├── train/                                          # Set Train (70%)
│   └── validation/                                     # Set Validation (15%)
│
├── docs/
│   ├── datasets/
│   ├── screenshots/
│   ├── confusion_matrix_optimized.png                  # Matricea de confuzie finală
│   ├── learning_curves_final.png                       # Grafice loss/accuracy
│   └── State_Machine.png                               # Diagrama de stări
│
├── models/
│   ├── classes.txt                                     # Lista claselor
│   ├── damage_model.h5                                 # Model Baseline (Etapa 5)
│   └── optimized_model.h5                              # Model Final Optimizat (Etapa 6)
│
├── results/
│   ├── error_analysis.csv                              # Analiza erorilor (Top 5)
│   ├── final_metrics.json                              # Metrici finale JSON
│   ├── optimization_experiments.csv                    # Tabel experimente
│   ├── test_metrics.json                               # Metrici baseline
│   ├── training_history.csv                            # Istoric antrenare
│   └── training_plot.png                               # Plot antrenare baseline
│
├── src/
│   ├── app/
│   │   └── app.py                                      # Aplicația Streamlit Finală
│   ├── data_acquisition/
│   │   ├── generate_synthetic_data.py                  # Script generare date (Contribuție)
│   │   ├── organize_dataset.py
│   │   └── split_data.py
│   ├── neural_network/
│   │   ├── train_model.py                              # Script antrenare vechi
│   │   └── train_optimized.py                          # Script antrenare FINAL
│   ├── preprocessing/
│   │   └── transformers.py                             # Procesare imagini
│   └── visualisation/
│       └── generate_plots.py                           # Generare grafice documentație
│
├── .gitignore
└── requirements.txt

```

## 9. Concluzii și Lecții Învățate

### 9.1 Evaluare Finală
- Proiectul a atins toate obiectivele tehnice, depășind pragul de 70% acuratețe (88.4% final). Sistemul este capabil să funcționeze ca un asistent de triaj în timp real.

### 9.2 Lecții Învățate
- Datele sunt critice: Generarea datelor sintetice a avut cel mai mare impact asupra performanței (+10%).

- Arhitectura: Trecerea la EfficientNet și Multi-Label (Sigmoid) a rezolvat problema daunelor multiple simultane.

- Optimizare: Monitorizarea val_loss și folosirea Early Stopping au prevenit overfitting-ul.

## 10. Concluzii și Discuții

### 10.1 Evaluare Performanță vs Obiective Inițiale

| Obiectiv Definit | Target | Realizat | Status |
|------------------|--------|----------|--------|
| Automatizare Triaj | < 1 min | < 1 sec | [✓] |
| Accuracy pe test set | ≥70% | 88.40% | [✓] |
| F1-Score pe test set | ≥0.65 | 0.85 | [✓] |

### 10.2 Ce NU Funcționează – Limitări Cunoscute

1.  **Dependența de lumină:** Performanța scade în condiții nocturne sau de iluminare artificială slabă (garaje subterane).
2.  **Context spațial:** Modelul identifică dauna, dar nu localizează exact piesa (nu face Object Detection / Bounding Box).
3.  **Murdăria:** Noroiul dens este uneori clasificat fals ca "Zgârietură" sau "Rugină".

### 10.3 Lecții Învățate (Top 5)

1.  **Datele bat Modelul:** Generarea datelor sintetice (augmentarea cu zgomot) a avut un impact mai mare asupra performanței (+10%) decât schimbarea arhitecturii (+6%).
2.  **Multi-Label este esențial:** În lumea reală, problemele nu sunt mutu-exclusive. Trecerea la `Sigmoid` a fost decizia cheie pentru realism.
3.  **Optimizarea iterativă:** Monitorizarea `Validation Loss` a salvat modelul de overfitting prin Early Stopping.
4.  **Arhitectura contează:** EfficientNet a fost mult mai capabil să vadă texturi fine (zgârieturi) decât MobileNetV2.
5.  **Calculul Severității:** Transformarea output-ului tehnic (probabilități) într-un metric de business (Scor Severitate) este ceea ce dă valoare aplicației.

---