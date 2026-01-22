# 📘 README – Etapa 4: Arhitectura Completă a Aplicației SIA bazată pe Rețele Neuronale

## 📌 Metadate Proiect

- **Disciplina:** Rețele Neuronale  
- **Instituție:** Universitatea POLITEHNICA din București – FIIR  
- **Student:** Dobrisan Andrei George  
- **Tema proiect:** Clasificare a Daunelor de Caroserie Auto folosind MobileNetV2  
- **Link Repository GitHub:** https://github.com/DobrisanGeorge/Car-Damage-Assessment-Retele-Neuronale  
- **Data:** 04.12.2025  

---

## 0. Scopul Etapei 4

Această etapă corespunde punctului **5. Dezvoltarea arhitecturii aplicației software bazată pe RN** din lista de 9 etape (slide 2 – *RN Specificații proiect.pdf*).

Obiectivul este să livrez un **schelet complet și funcțional** al sistemului cu inteligență artificială (**SIA**) pentru **Clasificarea Daunelor de Caroserie Auto**, folosind **MobileNetV2** ca model RN.

În acest stadiu:

- Modelul RN este **definit și compilat**, eventual cu ponderi inițiale (ex. ImageNet)
- Pipeline-ul este complet, de la **date → preprocess → RN → output UI**
- Sistemul poate fi rulat cap-coadă, chiar dacă performanța modelului nu este încă optimă

### ✔️ Ce trebuie să funcționeze în Etapa 4

- Toate cele **3 module principale**:
  - Modul 1 – Data Logging / Acquisition
  - Modul 2 – Neural Network (MobileNetV2)
  - Modul 3 – Web Service / UI (Streamlit)
- Codul rulează fără erori, minimal:
  - Scripturile de organizare a datelor creează structura `train/validation/test`
  - Modelul MobileNetV2 poate fi **definit, compilat și salvat** (`damage_model.h5`)
  - Aplicația Streamlit pornește, permite **upload de imagine** și afișează **o clasificare**

### ❌ Ce NU este obligatoriu în Etapa 4

- Model RN antrenat complet pe multe epoci
- Hiperparametri optimizați
- Acuratețe mare pe setul de test
- UI complexă și foarte polished

### Notă Anti-Plagiat

Modelul în această etapă este **NEANTRENAT sau minimal antrenat**. Arhitectura este construită de la zero în repository-ul propriu, demonstrând înțelegerea pipeline-ului și a structurii SIA.

---

## 1. Tabel – Nevoie Reală → Soluție SIA → Modul Software

Tabelul leagă nevoile reale identificate (în contextul asigurărilor auto) de soluția oferită prin SIA și modulul software responsabil.

| Nevoie reală                 | Cum o rezolvă SIA-ul                                | Modul software              |
|------------------------------|-----------------------------------------------------|-----------------------------|
| Evaluare automată a daunelor | Clasifică imaginea în `No`, `Minor`, `Major Damage` | **MobileNetV2 + Streamlit** |
| Reducerea erorilor umane     | Oferă scor de încredere pentru fiecare predicție    | **Inference Module**        |
| Integrare în flux digital    | Returnează rezultate structurate (JSON + text)      | **Web Service (Streamlit)** |
| Trasabilitate decizii        | Salvează log-uri cu imagine + rezultat              | **Data Logging**            |


**Observație:**  
Metricile sunt **măsurabile** (timp de răspuns, scor de încredere), iar fiecare nevoie este asociată cu modul(e) software clar(e) din arhitectură.

---

## 2. Contribuția Originală (≥ 40%) la Setul de Date

Conform cerinței, **minimum 40%** din totalul observațiilor finale utilizate în `data/train`, `data/validation`, `data/test` trebuie să fie **originale** (prelucrate, etichetate sau generate de mine).

### 2.1. Statistica setului de date

- **Total observații finale (după Etapa 3 + Etapa 4):** `5000` imagini  
- **Observații originale (contribuția mea):** `2000` imagini  
- **Procent contribuție originală:** `2000 / 5000 = 40%` ✅

### 2.2. Tipul contribuției mele

- [ ] Date generate prin simulare fizică  
- [ ] Date achiziționate cu senzori proprii  
- [x] Etichetare / adnotare manuală  
- [ ] Date sintetice prin metode avansate (GAN, augmentare sofisticată etc.)

### 2.3. Descriere detaliată

Am pornit de la un set de date brut provenit din surse publice, de tipul **Car Damage Assessment (Kaggle)**. Acest dataset avea:

- Imagini etichetate generic sau incorect  
- Imagini irelevante (non-auto, poze cu interior, background-uri fără mașini)  
- Lipsă de echilibru între clase (`No_Damage`, `Minor_Damage`, `Major_Damage`)

Contribuția mea concretă:

1. **Curățare și filtrare:**
   - Eliminarea imaginilor irelevante sau corupte
   - Eliminarea duplicatelor evidente
   - Verificarea manuală a unui subset semnificativ de imagini

2. **Etichetare/Adnotare manuală (2000 imagini):**
   - Re-etichetarea imaginilor cu daună clară drept `Minor_Damage` sau `Major_Damage`
   - Etichetarea imaginilor fără daună vizibilă drept `No_Damage`
   - Corectarea etichetelor greșite din datasetul inițial

3. **Completare cu noi imagini pentru `No_Damage`:**
   - Am extras un subset de imagini de mașini fără daună vizibilă din surse publice (ex. Google Images cu filtre de licență potrivite pentru uz academic)
   - Am etichetat manual aceste imagini ca `No_Damage` pentru **echilibrarea claselor**

### 2.4. Locația codului pentru contribuție

- **Script principal de organizare:**  
  `src/data_acquisition/organize_dataset.py`  
  - Curăță, redenumește și organizează imaginile în foldere pe clase  
- **Script de split train/val/test:**  
  `src/data_acquisition/split_data.py`  
  - Împarte dataset-ul final în `train/validation/test` (ex: 70% / 15% / 15%)

### 2.5. Dovezi în repo

- Structura finală a folderelor în `data/raw/` și `data/generated/`  
- Log-ul de rulare al scriptului `organize_dataset.py` (poate fi salvat ca `.txt`)  
- Raport/scurt script care afișează distribuția finală pe clase

---

## 3. Diagrama State Machine a Sistemului (OBLIGATORIE)

- **Fișier:** `docs/state_machine.png`  
- **Format:** PNG (poate fi creat în draw.io, PowerPoint, etc.)  

Diagrama acoperă fluxul complet al aplicației SIA pentru clasificarea daunelor auto.

### 3.1. Stări definite

State Machine-ul urmează logica unei aplicații de **Clasificare la cerere**:

1. **IDLE (Așteptare)**  
   - Aplicația Streamlit este pornită și așteaptă ca utilizatorul să încarce o imagine.

2. **VALIDATE_INPUT (Validare input)**  
   - Se verifică:
     - dacă fișierul încărcat este imagine (`.jpg`, `.jpeg`, `.png`)
     - mărimea fișierului
     - dacă poate fi citit de OpenCV / PIL

3. **PREPROCESS (Preprocesare imagine)**  
   - Operații:
     - redimensionare la `224 x 224` pixeli
     - conversie în tensor (NumPy array)
     - normalizare [0, 1] sau scalare specifică MobileNetV2
     - adăugare dimensiune batch `(1, 224, 224, 3)`

4. **INFERENCE (Inferență RN)**  
   - Imaginea preprocesată este dată ca input în modelul `MobileNetV2`  
   - Se obține un vector de probabilități pentru clasele:
     - `No_Damage`
     - `Minor_Damage`
     - `Major_Damage`

5. **DISPLAY_RESULT (Afișare rezultat)**  
   - Se determină clasa cu probabilitatea maximă  
   - Se afișează în UI:
     - clasa prezisă
     - scorul de încredere (ex: 0.87 → 87%)
   - Se afișează, eventual, și distribuția completă a probabilităților

6. **ERROR (Eroare)**  
   - Se ajunge aici dacă:
     - fișierul încărcat nu este imagine
     - fișierul este corupt
     - apare o excepție la preprocesare sau inferență  
   - Sistemul afișează un mesaj de eroare și revine în **IDLE**

### 3.2. Justificarea State Machine-ului ales

Am ales un model de tip **„Clasificare la cerere (On-Demand Classification)”** deoarece:

- În contextul **asigurărilor auto**, utilizatorul (client sau inspector) inițiază procesul prin încărcarea unei imagini.
- Nu este un sistem care rulează continuu în timp real, ci unul reacțional: **primesc input → procesez → dau rezultat**.
- Starea **ERROR** este esențială pentru a trata robust input-uri invalide și a evita crash-uri ale aplicației.
- Separarea în stări **VALIDATE_INPUT**, **PREPROCESS**, **INFERENCE**, **DISPLAY_RESULT** reflectă pipeline-ul standard al unui SIA modern.

---

## 4. Scheletul Complet al celor 3 Module (Conform Cursului – Slide 7)

Profesorul cere explicit 3 module:

1. Data Logging / Acquisition  
2. Neural Network Module  
3. Web Service / UI  

### 4.1. Tabel de sinteză module

| Modul                      | Tehnologii / Locație Python | LabVIEW (dacă e cazul)   | Cerință minimă la predare                          |
|----------------------------|-----------------------------|--------------------------|----------------------------------------------------|
| Data Logging / Acquisition | `src/data_acquisition/`     | VI-uri opționale         | Generează folderele cu imagini organizate pe clase |
| Neural Network Module      | `src/neural_network/`       | RN în LabVIEW (opțional) | Definește și salvează modelul `damage_model.h5`    |
| Web Service / UI           | `src/app/app.py`            | WebVI (opțional) | UI funcțional: upload + rezultat + scor                    |


---

### 4.2. Modul 1 – Data Logging / Acquisition

**Locație principală cod:**  
`src/data_acquisition/`

**Fișiere tipice:**

- `organize_dataset.py`
- `split_data.py`
- (opțional) `inspect_distribution.py` – pentru a afisa distribuția pe clase

**Funcționalități:**

- Citește imaginile din `data/raw/` și, eventual, `data/generated/`
- Curăță datele (elimină corupte/duplicat – script sau manual + script)
- Organizează imaginile pe clase:
  - `No_Damage/`
  - `Minor_Damage/`
  - `Major_Damage/`
- Face split-ul în seturi:
  - `data/train/`
  - `data/validation/`
  - `data/test/`

**Cerințe îndeplinite în Etapa 4:**

- [x] Codul `organize_dataset.py` rulează fără erori (`python src/data_acquisition/organize_dataset.py`)  
- [x] Structura de foldere `train/validation/test` este creată și populată  
- [x] Este respectată regula **≥ 40% contribuție originală**  
- [x] Scriptul conține comentarii explicative (ex. logica split-ului 70/15/15)

---

### 4.3. Modul 2 – Neural Network Module (MobileNetV2)

**Locație principală cod:**  
`src/neural_network/`

**Fișiere tipice:**

- `model_definition.py` – definește arhitectura MobileNetV2:
  - încărcare bază MobileNetV2 (cu sau fără ponderi ImageNet)
  - adăugare layer(e) fully-connected pentru 3 clase
- `train_model.py` – script de training minimal:
  - încarcă datele din `data/train`, `data/validation`
  - compilează modelul
  - rulează 1–2 epoci de test (nu antrenare serioasă)
  - salvează modelul în `models/damage_model.h5`

**Caracteristici model:**

- **Input shape:** `(224, 224, 3)`  
- **Backbone:** `tf.keras.applications.MobileNetV2`  
- **Output:** softmax cu 3 neuroni (pentru cele 3 clase)  
- **Loss:** `categorical_crossentropy` (dacă se folosesc one-hot labels)  
- **Optimizer:** `adam` (sau similar)  

**Cerințe îndeplinite în Etapa 4:**

- [x] Arhitectura RN este definită și compilată fără erori  
- [x] Modelul poate fi salvat și reîncărcat (`damage_model.h5`)  
- [x] Există un minim de documentație în cod (de ce MobileNetV2, de ce transfer learning)  
- [x] Nu este necesar un training complet – doar verificarea pipeline-ului  

---

### 4.4. Modul 3 – Web Service / UI (Streamlit)

**Locație principală cod:**  
`src/app/app.py`  
(sau direct în rădăcină `app.py`, important e să fie documentat în README)

**Tehnologie:**  
- **Streamlit** pentru interfață web simplă

**Funcționalități:**

- Încarcă modelul `damage_model.h5` (sau o versiune neantrenată)  
- Permite utilizatorului să încarce o imagine (file uploader)  
- Apelează funcțiile de:
  - `VALIDATE_INPUT`
  - `PREPROCESS`
  - `INFERENCE`  
- Afișează:
  - clasa prezisă (`No_Damage`, `Minor_Damage`, `Major_Damage`)
  - scorul de încredere
  - (opțional) un grafic cu distribuția probabilităților

**Cerințe îndeplinite în Etapa 4:**

- [x] `streamlit run app.py` pornește fără erori  
- [x] UI afişează un rezultat pentru imaginea încărcată (chiar dacă modelul nu e încă performant)  
- [x] Există un screenshot demo în `docs/screenshots/ui_demo.png`  

---

## 5. Structura Repository-ului la Finalul Etapei 4

Structura recomandată (adaptată proiectului de clasificare daune auto):

```bash
proiect-rn-Dobrisan-Andrei-George/
├── data/
│   ├── raw/               # Imaginile brute și fișierele originale
│   ├── processed/         # (Opțional) imagini redimensionate/normalize
│   ├── generated/         # Date originale (dacă sunt separate)
│   ├── train/             # Structura finală pentru antrenare
│   ├── validation/
│   └── test/
├── src/
│   ├── data_acquisition/  # MODUL 1: organize_dataset.py, split_data.py
│   ├── preprocessing/     # Funcții comune de preprocess (ex. preprocess_image.py)
│   ├── neural_network/    # MODUL 2: model_definition.py, train_model.py
│   └── app/               # MODUL 3: app.py (Streamlit)
├── docs/
│   ├── state_machine.png  # Diagrama State Machine (OBLIGATORIU)
│   └── screenshots/
│       └── ui_demo.png    # Screenshot UI Streamlit
├── models/
│   └── damage_model.h5    # Model MobileNetV2 (neantrenat sau minimal antrenat)
├── config/
│   └── config.yaml        # (Opțional) Config pentru paths, parametri
├── README.md              # README general proiect
├── README_Etapa3.md       # README specific Etapei 3
├── README_Etapa4_Arhitectura_SIA.md  # Acest fișier
└── requirements.txt       # Dependințe Python (tensorflow, streamlit, etc.)
