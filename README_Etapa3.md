**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Dobrisan Andrei George
**Data:** 2025.11.20  

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Introducere

Acest document descrie activitățile realizate în **Etapa 3**, în care se analizează și se preprocesează setul de date necesar proiectului „AutoDataAI". Scopul etapei este pregătirea corectă a datelor pentru instruirea modelului RN, respectând bunele practici privind calitatea, consistența și reproductibilitatea datelor.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

##  1. Structura Repository-ului Github (versiunea Etapei 3)

```
CarVisionAI/
├── README.md
├── docs/
│   └── datasets/          # descriere seturi de date, surse, diagrame
├── data/
│   ├── raw/               # date brute
│   ├── processed/         # date curățate și transformate
│   ├── train/             # set de instruire
│   ├── validation/        # set de validare
│   └── test/              # set de testare
├── src/
│   ├── preprocessing/     # funcții pentru preprocesare
│   ├── data_acquisition/  # generare / achiziție date (dacă există)
│   └── neural_network/    # implementarea RN (în etapa următoare)
├── config/                # fișiere de configurare
└── requirements.txt       # dependențe Python
```

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

##  2. Descrierea Setului de Date

### 2.1 Sursa datelor

* **Origine:** Set de date public (Kaggle - Car Damage Assessment)
* **Modul de achiziție:** Fișier extern (Descărcare)
* **Perioada / condițiile colectării:** Colecție de imagini sub diverse condiții de iluminare și unghiuri, reprezentând daune în lumea reală.

### 2.2 Caracteristicile dataset-ului

* **Număr total de observații:**  ≈1500 Imagini.
* **Număr de caracteristici (features):** Imagini RGB (224×224×3 pixeli)
* **Tipuri de date:**  Imagini
* **Format fișiere:**  PNG / JPEG

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### 2.3 Descrierea fiecărei caracteristici

## Caracteristicile Damage Detection

| Caracteristică (Clasă) | Descriere                                                | Domeniu valori            |
|------------------------|----------------------------------------------------------|---------------------------|
| **No_Damage**          | Vehiculul nu prezintă daune vizibile.                    | Clasificare (0 sau 1)     |
| **Minor_Damage**       | Daune superficiale (zgârieturi ușoare, îndoituri mici).  | Clasificare (0 sau 1)     |
| **Major_Damage**       | Daune semnificative (elemente structurale afectate).     | Clasificare (0 sau 1)     |

**Tip:** Categorial (Clasificare)
**Format:** One-Hot Encoding (la ieșirea RN)

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Fișier recomandat:**  `data/README.md`

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

##  3. Analiza Exploratorie a Datelor (EDA) – Sintetic

### 3.1 Statistici descriptive aplicate

* **Distribuția pe clase:** Numărul exact de imagini în fiecare clasă (Ex: Major_Damage vs. No_Damage) pentru a identifica dezechilibrele.
* **Dimensiunea imaginilor:** Verificarea rezoluției medii și a raportului de aspect (aspect ratio) al imaginilor brute.
* **Verificarea imaginilor corupte:** Identificarea fișierelor care nu pot fi deschise sau nu au extensia corectă.

### 3.2 Analiza calității datelor

* **Consistența claselor:** Verificarea manuală a unui eșantion aleatoriu din fiecare folder pentru a asigura că etichetele corespund daunei vizuale.
* **Detectarea imaginilor inutilizabile:** Eliminarea imaginilor extrem de neclare (blur), întunecate sau care nu conțin mașini.

### 3.3 Probleme identificate

* **Dezechilibru de clasă (Class Imbalance):** Clasa "Daună Majoră" are semnificativ mai puține imagini decât "Fără Daună", ceea ce poate duce la un model părtinitor.
* **Variație coloristică:** Imaginile provin din surse diverse, având balanțe de alb și expuneri foarte diferite.
* **Variație de încadrare:** Unele imagini sunt prim-planuri (macro) ale zgârieturii, altele sunt cadre largi cu toată mașina.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 4. Preprocesarea Datelor

### 4.1 Curățarea datelor

- **Eliminarea duplicatelor:** Utilizarea hash-urilor pentru identificarea și ștergerea imaginilor identice.  
- **Standardizarea rezoluției:** Redimensionarea tuturor imaginilor la rezoluția de intrare MobileNetV2 (224×224 pixeli), folosind interpolare bicubică.

### 4.2 Transformarea caracteristicilor

- **Normalizare pixeli:** Scalarea valorilor din intervalul [0, 255] în [−1, 1] sau [0, 1], conform cerințelor MobileNetV2.  
- **Augmentare geometrică:** Rotații mici (±10°), zoom (0.2) și inversare orizontală (horizontal flip) aplicate în timpul antrenării.  
- **Augmentare cromatică:** Ajustarea luminozității și contrastului pentru a simula diverse condiții de iluminare.

### 4.3 Structurarea seturilor de date

- **Împărțire recomandată:**
  - 70% – train  
  - 15% – validation  
  - 15% – test  

- **Principii respectate:**
  - **Stratificare:** Menținerea proporției claselor în toate seturile.  
  - **Fără scurgere de informație (Data Leakage):** Imaginile de test sunt complet izolate și nu sunt utilizate la augmentare.  
  - **Organizare directoare:** Folosirea structurii standard *ImageFolder* (`train/class_name/image.jpg`).

### 4.4 Salvarea rezultatelor preprocesării

- **Date preprocesate:** Stocate on-the-fly în memorie sau salvate în `data/processed/` dacă spațiul permite.  
- **Seturi train/val/test:** Organizate fizic în foldere dedicate.  
- **Configurare:** Parametrii de augmentare salvați în `config/preprocessing_config.yaml`.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 5. Fișiere Generate în Această Etapă

* `data/raw/` – date brute (Imaginile originale descărcate de pe Kaggle).
* `data/processed/` – date curățate & transformate (Imagini redimensionate la 224x224 și normalizate).
* `data/train/`, `data/validation/`, `data/test/` – seturi finale organizate pe clase.
* `src/preprocessing/` – codul de preprocesare (scripturi Python pentru augmentare și split).
* `data/README.md` – descrierea detaliată a dataset-ului (distribuție clase, sursă).

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

##  6. Stare Etapă (de completat de student)

- [x] Structură repository configurată
- [x] Dataset analizat (EDA realizată)
- [ ] Date preprocesate
- [ ] Seturi train/val/test generate
- [ ] Documentație actualizată în README + `data/README.md`

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
