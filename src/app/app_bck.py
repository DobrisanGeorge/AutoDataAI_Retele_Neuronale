import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import time

# --- IMPORT MODULE PROPRII (MODULARIZARE) ---
try:
    from src.preprocessing.transformers import preprocess_image_for_model
except ImportError:
    st.error("❌ Eroare critică: Nu găsesc modulul 'src.preprocessing.transformers'.")
    st.error("Te rog rulează comanda 'streamlit run app.py' din folderul principal al proiectului!")
    st.stop()

# --- 1. CONFIGURARE PAGINĂ ---
st.set_page_config(
    page_title="AutoClaim AI - Ultimate",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. STILURI CSS (DARK MODE ENTERPRISE) ---
st.markdown("""
    <style>
    /* Fundal */
    .stApp { background-color: #0E1117 !important; color: #E0E0E0 !important; }
    
    /* Carduri */
    .css-card { 
        background-color: #1E1E1E; 
        padding: 20px; 
        border-radius: 12px; 
        border: 1px solid #333; 
        margin-bottom: 20px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Card special pentru Daună Totală */
    .total-loss-card {
        background: linear-gradient(135deg, #4a0000 0%, #2a0000 100%);
        border: 2px solid #ff4444;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 68, 68, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(255, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 68, 68, 0); }
    }
    
    /* Texte */
    h1, h2, h3 { color: #ffffff !important; font-family: 'Segoe UI', sans-serif; }
    p, span, div { color: #cccccc; }
    .metric-value { font-size: 2rem; font-weight: bold; color: white; }
    .metric-label { font-size: 0.8rem; text-transform: uppercase; color: #888; }
    
    /* Header */
    .header-box {
        background: linear-gradient(90deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 20px; border-radius: 12px; text-align: center; margin-bottom: 30px;
    }
    
    #MainMenu, footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- 3. CĂI ---
MODEL_PATH = os.path.join('models', 'damage_model.h5')
CLASSES_PATH = os.path.join('models', 'classes.txt')

# --- 4. LOGICA DE BUSINESS (SEVERITATE) ---
# Punctaj pentru fiecare tip de daună (0-10)
SEVERITY_MAP = {
    "scratch": 2, "dent": 4, "lamp": 5, "glass": 6, "shatter": 7,
    "bumper": 3, "door": 4, "severe": 10, "wrecked": 10, "unknown": 1
}

def calculate_severity(detected_labels):
    """Calculează scorul total de gravitate bazat pe daunele găsite."""
    total_score = 0
    
    for item in detected_labels:
        label_text = item['raw'].lower()
        points = 2 # Default
        
        # Căutăm cuvinte cheie
        for key, val in SEVERITY_MAP.items():
            if key in label_text:
                points = max(points, val)
        
        # Ponderăm cu scorul de încredere (dacă AI e sigur, punctajul contează 100%)
        weighted_points = points * (item['score'] / 100.0)
        total_score += weighted_points
        
    return total_score

# --- 5. FUNCȚII UTILITARE ---
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH): return None
    return tf.keras.models.load_model(MODEL_PATH)

def load_labels():
    if not os.path.exists(CLASSES_PATH): return []
    with open(CLASSES_PATH, 'r') as f:
        return [line.strip() for line in f.readlines()]

# --- 6. INTERFAȚA (UI) ---

# Header
st.markdown("""
<div class="header-box">
    <h1>🛡️ AutoClaim AI • Ultimate</h1>
    <p>Sistem Neural de Expertiză Auto & Detecție Daună Totală</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Parametri Scanare")
    
    sensitivity = st.slider(
        "Sensibilitate AI (%)", 
        min_value=10, max_value=90, value=40,
        help="Pragul minim pentru a considera o daună detectată."
    )
    
    st.markdown("---")
    st.markdown("### 📥 Probă Foto")
    uploaded_file = st.file_uploader("Încarcă Imagine (JPG/PNG)", type=["jpg", "png", "jpeg"])
    
    st.markdown("---")
    # Status Model
    if os.path.exists(MODEL_PATH):
        st.success("🟢 Model Neural: ACTIV")
    else:
        st.error("🔴 Model Lipsă")

# Logica Principală
if not uploaded_file:
    st.info("👈 Te rog încarcă o fotografie din panoul lateral pentru a începe expertiza.")
else:
    model = load_model()
    labels = load_labels()
    
    if not model:
        st.error("Eroare: Modelul nu este antrenat. Rulează `train_model.py`.")
        st.stop()
        
    # Layout 2 Coloane
    col_img, col_res = st.columns([1, 1.2], gap="large")
    
    with col_img:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">📸 Imagine Analizată</div>', unsafe_allow_html=True)
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_res:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">📊 Raport Expertiză</div>', unsafe_allow_html=True)
        
        if st.button("🚀 EXECUTA ANALIZA COMPLEXĂ", type="primary"):
            with st.spinner("Procesare EfficientNet & Calcul Severitate..."):
                time.sleep(1) # Suspans vizual
                
                # 1. PREPROCESARE (Folosind funcția importată din src/preprocessing)
                # Rezoluția 260x260 este specifică EfficientNetB0 (sau 224 pentru MobileNet)
                # Punem 260 pentru că am upgradat modelul
                proc_img = preprocess_image_for_model(image, target_size=(260, 260))
                
                # 2. INFERENȚĂ
                preds = model.predict(proc_img)
                
                # 3. POST-PROCESARE (Multi-Label)
                detected = []
                threshold_val = sensitivity / 100.0
                
                for i, score in enumerate(preds[0]):
                    if score > threshold_val:
                        detected.append({
                            'label': labels[i].replace("_", " ").title(),
                            'score': score * 100,
                            'raw': labels[i]
                        })
                
                # Sortare după scor
                detected.sort(key=lambda x: x['score'], reverse=True)
                
                # 4. CALCUL SEVERITATE
                severity_score = calculate_severity(detected)
                
                # Logică Daună Totală (Prag arbitrar 12 sau cuvinte cheie severe)
                is_total_loss = severity_score >= 12.0 or any("severe" in d['raw'] for d in detected)
                
                # Salvare în sesiune
                st.session_state['scan_result'] = {
                    'detected': detected,
                    'severity': severity_score,
                    'is_total_loss': is_total_loss
                }
        
        # AFIȘARE REZULTATE
        if 'scan_result' in st.session_state:
            res = st.session_state['scan_result']
            
            # A. ALERTA DE DAUNĂ TOTALĂ
            if res['is_total_loss']:
                st.markdown("""
                <div class="total-loss-card">
                    <h2 style="color:#ffcccc !important; margin:0;">⛔ DAUNĂ TOTALĂ DETECTATĂ</h2>
                    <p style="color:#ffcccc; margin:0;">Vehiculul prezintă avarii structurale critice.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # B. SCOR SEVERITATE
            c1, c2 = st.columns(2)
            c1.metric("Scor Severitate", f"{res['severity']:.1f} / 20")
            c2.metric("Avarii Identificate", len(res['detected']))
            
            st.markdown("GRAD UZURĂ:")
            st.progress(min(res['severity'] / 20, 1.0))
            
            st.divider()
            
            # C. LISTA DETALIATĂ
            if len(res['detected']) == 0:
                st.success("✅ Fără avarii vizibile (peste pragul setat).")
            else:
                st.write("DETALII AVARII:")
                for item in res['detected']:
                    # Color coding în funcție de siguranță
                    color = "#00ff00" if item['score'] > 80 else "#ffa500"
                    st.markdown(f"""
                    <div style="background-color: #2b2b2b; padding: 10px; border-radius: 5px; margin-bottom: 5px; border-left: 4px solid {color};">
                        <div style="display:flex; justify-content:space-between;">
                            <span style="font-weight:bold; color:white;">{item['label']}</span>
                            <span style="color:{color};">{item['score']:.1f}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)