import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import os
import sys
import time
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)

try:
    from src.preprocessing.transformers import preprocess_image_for_model
except ImportError as e:
    st.error(f"❌ Eroare Critică: Nu pot importa modulele necesare.")
    st.code(f"Eroare detaliată: {e}")
    st.info(f"Python caută în: {sys.path}")
    st.stop()

# --- 1. IMPORT MODULAR ---
try:
    from src.preprocessing.transformers import preprocess_image_for_model
except ImportError:
    st.error("❌ Eroare: Nu găsesc 'src/preprocessing/transformers.py'.")
    st.info("Asigură-te că rulezi comanda din folderul rădăcină: 'streamlit run src/app/app.py'")
    st.stop()

# --- 2. CONFIGURARE PAGINĂ ---
st.set_page_config(
    page_title="AutoClaim AI - Final Exam",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3. STILURI CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #0E1117 !important; color: #E0E0E0 !important; }
    .css-card { background-color: #1E1E1E; padding: 20px; border-radius: 12px; border: 1px solid #333; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    .total-loss-card { background: linear-gradient(135deg, #4a0000 0%, #2a0000 100%); border: 2px solid #ff4444; padding: 20px; border-radius: 12px; margin-bottom: 20px; text-align: center; animation: pulse 2s infinite; }
    @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(255, 68, 68, 0.4); } 70% { box-shadow: 0 0 0 10px rgba(255, 68, 68, 0); } 100% { box-shadow: 0 0 0 0 rgba(255, 68, 68, 0); } }
    .header-box { background: linear-gradient(90deg, #0f2027 0%, #203a43 50%, #2c5364 100%); padding: 25px; border-radius: 12px; text-align: center; margin-bottom: 30px; }
    .metric-box { background-color: #262730; border-left: 5px solid #00ADB5; padding: 15px; border-radius: 5px; }
    h1, h2, h3 { color: #ffffff !important; }
    </style>
""", unsafe_allow_html=True)

# --- 4. CĂI FIȘIERE (Sincronizate cu train_optimized.py) ---
MODEL_PATH = os.path.join('models', 'optimized_model.h5')
CLASSES_PATH = os.path.join('models', 'classes.txt')
METRICS_PATH = os.path.join('results', 'final_metrics.json')
CONFUSION_MATRIX_PATH = os.path.join('docs', 'confusion_matrix_optimized.png')
HISTORY_PATH = os.path.join('results', 'optimization_experiments.csv')

# --- 5. LOGICA SEVERITATE ---
SEVERITY_MAP = {
    "scratch": 2, "dent": 4, "lamp": 5, "glass": 6, 
    "bumper": 3, "door": 4, "severe": 10, "wrecked": 10, "unknown": 1
}

def calculate_severity(detected_labels):
    score = 0
    for item in detected_labels:
        txt = item['raw'].lower()
        pts = 2
        for k, v in SEVERITY_MAP.items():
            if k in txt: pts = max(pts, v)
        score += pts * (item['score'] / 100.0)
    return score

# --- 6. UTILS ---
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH): return None
    return tf.keras.models.load_model(MODEL_PATH)

def load_labels():
    if not os.path.exists(CLASSES_PATH): return []
    with open(CLASSES_PATH, 'r') as f: return [line.strip() for line in f.readlines()]

# --- 7. UI ---
st.markdown('<div class="header-box"><h1>🛡️ AutoClaim AI • Versiune Finală</h1><p>Sistem Neural Optimizat (EfficientNet + MultiLabel)</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Configurare Inferență")
    sensitivity = st.slider("Sensibilitate AI", 10, 90, 40)
    uploaded_file = st.file_uploader("Imagine Test", type=["jpg", "png", "jpeg"])
    
    st.divider()
    if os.path.exists(MODEL_PATH):
        st.success("🟢 Model Optimizat: ACTIV")
    else:
        st.error(f"🔴 Lipsă Model: {MODEL_PATH}")
        st.info("Rulează 'src/neural_network/train_optimized.py'!")

tab1, tab2, tab3 = st.tabs(["🕵️ Expertiză Live", "🏆 Metrici Performanță", "🧪 Istoric Experimente"])

# TAB 1: LIVE DEMO
with tab1:
    if uploaded_file:
        model = load_model()
        labels = load_labels()
        if model:
            c1, c2 = st.columns([1, 1.2], gap="large")
            with c1:
                st.markdown('<div class="css-card">📸 Imagine</div>', unsafe_allow_html=True)
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, use_container_width=True)
            with c2:
                st.markdown('<div class="css-card">📊 Rezultat Analiză</div>', unsafe_allow_html=True)
                if st.button("🚀 EXECUTA ANALIZA", type="primary"):
                    with st.spinner("Procesare EfficientNetB0..."):
                        time.sleep(0.5)
                        proc_img = preprocess_image_for_model(image, target_size=(260, 260))
                        preds = model.predict(proc_img)
                        
                        detected = []
                        for i, p in enumerate(preds[0]):
                            if p > sensitivity/100.0:
                                detected.append({'label': labels[i].title(), 'score': p*100, 'raw': labels[i]})
                        detected.sort(key=lambda x: x['score'], reverse=True)
                        
                        sev = calculate_severity(detected)
                        is_total = sev >= 12 or any("severe" in d['raw'] for d in detected)
                        st.session_state['res'] = {'det': detected, 'sev': sev, 'tot': is_total}

                if 'res' in st.session_state:
                    res = st.session_state['res']
                    if res['tot']:
                        st.markdown('<div class="total-loss"><h2>⛔ DAUNĂ TOTALĂ</h2><p>Avarii structurale critice.</p></div>', unsafe_allow_html=True)
                    
                    c_a, c_b = st.columns(2)
                    c_a.metric("Scor Severitate", f"{res['sev']:.1f} / 20")
                    c_b.metric("Avarii Găsite", len(res['det']))
                    st.progress(float(min(res['sev']/20, 1.0)))
                    st.divider()
                    for d in res['det']:
                        st.write(f"🔸 **{d['label']}** ({d['score']:.1f}%)")
                    if not res['det']: st.info("Nu s-au detectat avarii majore.")

# TAB 2: METRICI (JSON + Confusion Matrix)
with tab2:
    st.markdown("### 📊 Raport de Performanță (Test Set)")
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f: metrics = json.load(f)
        c1, c2, c3 = st.columns(3)
        with c1: st.markdown(f'<div class="metric-box"><h3>Acuratețe</h3><h1>{metrics["test_accuracy"]:.2%}</h1></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="metric-box"><h3>Eroare (Loss)</h3><h1>{metrics["test_loss"]:.4f}</h1></div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="metric-box"><h3>Best Exp.</h3><h3>{metrics.get("best_experiment", "N/A")}</h3></div>', unsafe_allow_html=True)
        
        st.divider()
        st.markdown("### 📉 Matricea de Confuzie")
        if os.path.exists(CONFUSION_MATRIX_PATH):
            st.image(CONFUSION_MATRIX_PATH, caption="Performanța pe Clase", use_container_width=True)
        else:
            st.warning("Imaginea matricei lipsește.")
    else:
        st.warning("⚠️ Nu există 'final_metrics.json'. Rulează scriptul de optimizare!")

# TAB 3: EXPERIMENTE (CSV)
with tab3:
    st.markdown("### 🧪 Jurnal de Optimizare")
    if os.path.exists(HISTORY_PATH):
        df_exp = pd.read_csv(HISTORY_PATH)
        st.dataframe(df_exp.style.highlight_max(axis=0, subset=['Val Accuracy']), use_container_width=True)
    else:
        st.info("Niciun experiment rulat încă.")