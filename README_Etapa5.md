# 📘 README – Etapa 5: Configurarea și Antrenarea Modelului RN

**Disciplina:** Rețele Neuronale
**Instituție:** POLITEHNICA București – FIIR
**Student:** Dobrisan Andrei George
**Link Repository GitHub:** https://github.com/DobrisanGeorge/Car-Damage-Assessment-Retele-Neuronale
**Data predării:** 11.12.2025

---

## Scopul Etapei 5

Această etapă corespunde punctului **6. Configurarea și antrenarea modelului RN** din lista de 9 etape - slide 2 **RN Specificatii proiect.pdf**.

**Obiectiv principal:** Antrenarea efectivă a modelului RN definit în Etapa 4 (MobileNetV2), evaluarea performanței și integrarea în aplicația completă.

**Pornire obligatorie:** Arhitectura completă și funcțională din Etapa 4:
- State Machine definit și justificat
- Cele 3 module funcționale (Data Logging, RN, UI)
- Minimum 40% date originale în dataset

---

## PREREQUISITE – Verificare Etapa 4 (OBLIGATORIU)

**Înainte de a începe Etapa 5, verificați că aveți din Etapa 4:**

- [x] **State Machine** definit și documentat în `docs/state_machine.png`
- [x] **Contribuție ≥40% date originale** în `data/raw/` (validate manual)
- [x] **Modul 1 (Data Logging)** funcțional - produce structura de foldere `train/val/test`
- [x] **Modul 2 (RN)** definit (MobileNetV2) - codul compilează
- [x] **Modul 3 (UI)** funcțional - interfața Streamlit pornește

---

## 1. Configurarea Hiperparametrilor (Tabel Obligatoriu)

Următorii parametri au fost utilizați în scriptul `src/neural_network/train_model.py` pentru antrenarea modelului de clasificare a daunelor.

| **Hiperparametru** | **Valoare Aleasă** | **Justificare Tehnică (De ce ați ales această valoare?)**                             |
|:---                |:---                |:---                                                                                   |
| **Batch Size**     | 32                 | Asigură un echilibru între utilizarea memoriei VRAM și stabilitatea gradientului      |
| **Learning Rate**  | 0.0001 ($1e^{-4}$) | Folosim o rată de învățare mică pentru a ajusta fin ponderile                         |
| **Număr Epoci**    | 15                 | Suficient pentru convergență în cazul Transfer Learning.                              |
| **Optimizer**      | Adam               | Algoritm adaptiv care converge mai rapid decât SGD clasic Este standardul industriei  |
| **Loss Function**  | Categorical Crossentropy | Deoarece avem o problemă de clasificare multi-clasă cu etichete one-hot encoded |
| **Dropout Rate**   | 0.2 (20%)          | Introdus în stratul dens final pentru a preveni supra-învățarea (overfitting)         |

---

## 2. Rezultatele Antrenării (Nivel 1)

După rularea antrenării, rezultatele au fost salvate în `results/`.

### 2.1 Istoricul Antrenării (Grafice)

Graficele de Loss și Accuracy pe seturile de Train vs Validation se găsesc în:
📂 **`results/training_plot.png`**

*Analiza scurtă a graficelor:*
- Curbele de antrenare și validare converg, ceea ce indică faptul că modelul învață.
- Diferența mică dintre Train Accuracy și Validation Accuracy sugerează că **nu există un overfitting major** (datorită Dropout-ului și Augmentării datelor).

### 2.2 Metrici Finale pe Setul de Test (Test Set)

Fișier generat: `results/test_metrics.json`

```json
{
  "test_accuracy": 0.8945,
  "test_loss": 0.3120,
  "test_f1_macro": 0.8801,
  "confusion_matrix": [
      [150, 10, 5],
      [12, 140, 15],
      [2, 8, 158]
  ]
}

---

```
### 3. Analiza Erorilor în Context Industrial (Nivel 2 – Obligatoriu pentru notă maximă)

Această secțiune analizează comportamentul modelului în scenariul real al unei companii de asigurări și impactul erorilor în procesul de evaluare automată a daunelor auto.

---

### A. Impactul Falselor Negative vs. Falselor Pozitive

În domeniul asigurărilor auto, aceste două tipuri de erori au implicații diferite:

#### ❗ False Negative (Critic)
Modelul prezice **"No Damage"** atunci când vehiculul are de fapt **"Major Damage"**.

Consecințe:
- Mașina avariată poate rămâne în circulație (risc de siguranță).
- Un dosar de daună ar putea fi respins incorect.
- Compania riscă probleme legale și deteriorarea reputației.

➡️ **Aceasta este cea mai periculoasă eroare.**

#### ✔️ False Positive (Acceptabil)
Modelul prezice **"Minor Damage"** asupra unei mașini fără defecte.

Consecințe:
- Necesită doar o verificare suplimentară de către un inspector.
- Nu generează pierderi financiare directe.
- Cost suplimentar doar în timp, nu în calitate sau siguranță.

➡️ **Este o eroare tolerabilă în sistemele industriale.**

**Strategie implementată:**  
Modelul a fost optimizat pentru un **Recall ridicat pe clasele de daune**.  
Este preferabil să fim „paranoici” și să semnalăm o daună, decât să o ratăm.

---

### B. Provocarea Datelor Neașteptate (Out-of-Distribution)

Pozele folosite în producție sunt foarte diferite de cele din setul curat de antrenare.

Provocări reale:
- **Reflexii puternice:** lumina soarelui reflectată în caroserie poate imita zgârieturi.
- **Murdărie / Noroi:** poate fi interpretată greșit drept „Rust” sau „Major Damage”.
- **Unghiuri neobișnuite:** pozele clienților sunt adesea nealiniate sau parțiale.

**Soluție implementată:**  
✔️ Data Augmentation (luminozitate, contrast, zgomot, blur, unghiuri)

**Recomandare pentru producție:**  
➕ Introducerea unei clase dedicate: **„Murdar / Neclar”**, care să declanșeze solicitarea automată a unei poze noi.

---

### C. Dezechilibrul Claselor (Class Imbalance)

În practică:
- Majoritatea mașinilor **nu au daune** (peste 80–90% în unele procese industriale).

Risc:
- Modelul ar putea învăța să prezică **doar „No Damage”**, obținând artificial o acuratețe mare, dar fiind inutil în producție.

**Soluții implementate:**
- ✔️ Echilibrarea manuală a setului de date.
- ✔️ Utilizarea metricei **F1-Score**, care penalizează predicțiile părtinitoare.
- ✔️ Monitorizarea Recall-ului pe clasele de daune, nu doar Accuracy.

---

### D. Concluzie și Pași Următori

Modelul actual (MobileNetV2) demonstrează **viabilitate tehnică** ca sistem de pre-triere a dosarelor.

**Recomandare pentru implementare industrială:**  
### 🔄 Sistem Human-in-the-loop
- AI-ul aprobă automat cazurile cu **No Damage (încredere > 99%)**.
- Cazurile ambigue sau cu daune sunt trimise inspectorilor umani.

**Beneficii estimate:**  
✔️ Reducerea volumului de muncă manuală cu **60–70%**  
✔️ Reducerea timpului de procesare a dosarelor  
✔️ Crește consistența și obiectivitatea evaluărilor

--- 

Structura Fișierelor Generate în Etapa 5
Plaintext

```bash
AutoDataAI/
├── data/
│   ├── raw/                   # Imaginile originale (sortate pe foldere) + data.csv
│   ├── train/                 # Date de antrenare (70%)
│   ├── validation/            # Date de validare (15%)
│   └── test/                  # Date de testare (15%)
│
├── docs/
│   ├── state_machine.png             # Diagrama fluxului (din Etapa 4)
│   ├── screenshots/                  # Capturi de ecran cu aplicația
│   ├── etapa3_data_prep.md           # Documentație Etapa 3
│   ├── etapa4_arhitectura.md         # Documentație Etapa 4
│   └── etapa5_antrenare_model.md     # Documentație Etapa 5 (acest fișier)
│
├── models/
│   ├── damage_model.h5        # Modelul antrenat (Livrabil principal)
│   └── classes.txt            # Lista claselor (ex: No_Damage, Minor, Major)
│
├── results/
│   ├── training_history.csv   # Log-ul per epocă
│   ├── training_plot.png      # Graficele Loss/Accuracy
│   └── test_metrics.json      # Rezultatele finale pe setul de test
│
├── src/
│   ├── data_acquisition/
│   │   ├── organize_dataset.py       # Script sortare CSV -> Foldere
│   │   └── split_data.py             # Script împărțire Train/Val/Test
│   │
│   └── neural_network/
│       └── train_model.py            # Scriptul de antrenare (MobileNetV2)
│
├── app.py                     # Interfața grafică (Streamlit)
├── requirements.txt           # Lista dependențelor (tensorflow, streamlit, etc.)
└── README.md                  # README-ul principal al proiectului

```
Checklist Final – Etapa 5
[x] Tabel hiperparametri completat și justificat

[x] Model antrenat (damage_model.h5) existent în folderul models/

[x] Grafice de antrenare salvate în results/

[x] Metrici finale (Accuracy/F1) raportate

[x] Analiza erorilor (Nivel 2) redactată (vezi Secțiunea 3)

[x] UI actualizat să încarce noul model antrenat