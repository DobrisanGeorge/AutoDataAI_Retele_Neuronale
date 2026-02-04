import streamlit as st
import os
import sys
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import time
import json

# --- 1. CONFIGURARE CĂI & IMPORTURI ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)

try:
    from src.preprocessing.transformers import preprocess_image_for_model
except ImportError as e:
    st.error(f"❌ Eroare Critică: Nu pot importa modulele necesare. {e}")
    st.stop()

# --- 2. CONFIGURARE PAGINĂ ---
st.set_page_config(
    page_title="AutoClaim AI - Final Exam",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3. STILURI CSS (Design Enterprise) ---
st.markdown("""
    <style>
    .stApp { background-color: #0E1117 !important; color: #E0E0E0 !important; }
    .css-card { background-color: #1E1E1E; padding: 20px; border-radius: 12px; border: 1px solid #333; margin-bottom: 20px; }
    .header-box { background: linear-gradient(90deg, #0f2027 0%, #203a43 50%, #2c5364 100%); padding: 25px; border-radius: 12px; text-align: center; margin-bottom: 30px; }
    .severity-box { padding: 15px; border-radius: 10px; text-align: center; margin-top: 10px; font-weight: bold; font-size: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.2); }
    .main-metric { font-size: 3rem; font-weight: bold; margin: 0; }
    .sub-metric { font-size: 1.2rem; color: #bbb; }
    </style>
""", unsafe_allow_html=True)

# --- 4. CĂI FIȘIERE ---
MODEL_PATH = os.path.join(root_dir, 'models', 'optimized_model.h5')
CLASSES_PATH = os.path.join(root_dir, 'models', 'classes.txt')
METRICS_PATH = os.path.join(root_dir, 'results', 'final_metrics.json')
CONFUSION_MATRIX_PATH = os.path.join(root_dir, 'docs', 'confusion_matrix_optimized.png')
HISTORY_PATH = os.path.join(root_dir, 'results', 'optimization_experiments.csv')

# --- 5. LOGICA AVANSATĂ DE SEVERITATE ---
def get_severity_details(label_name):
    """Returnează nivelul numeric și detaliile pentru o clasă."""
    label = label_name.lower()
    
    # LEVEL 3: CRITIC (Roșu)
    if any(x in label for x in ['glass', 'shatter', 'wrecked', 'severe', 'smash', 'destroy']):
        return 3, "CRITIC", "🔴 Siguranță Compromisă", "#ff4b4b"
    
    # LEVEL 2: MEDIU (Portocaliu) - Îndoituri, Faruri, Elemente Caroserie
    elif any(x in label for x in ['lamp', 'head_lamp', 'tail_lamp', 'dent', 'bumper', 'door', 'hood', 'panel']):
        return 2, "MEDIU", "🟠 Înlocuire / Tinichigerie", "#ffa421"
    
    # LEVEL 1: UȘOR (Verde) - Zgârieturi simple
    elif any(x in label for x in ['scratch', 'paint', 'minor', 'smudge']):
        return 1, "UȘOR", "🟢 Vopsire / Polish", "#00c853"
        
    return 0, "NECUNOSCUT", "⚪ Necesită Inspecție", "#808080"

def calculate_global_severity(detected_items):
    """
    Scanează TOATE avariile și determină severitatea maximă.
    Ex: Dacă ai 'Scratch' (Ușor) și 'Glass' (Critic), rezultatul este CRITIC.
    """
    max_level = 0
    final_status = ("NECUNOSCUT", "⚪ Inspecție", "#808080")
    
    for item in detected_items:
        lvl, text, action, color = get_severity_details(item['raw'])
        if lvl > max_level:
            max_level = lvl
            final_status = (text, action, color)
            
    return final_status

# --- 6. UTILS ---
@st.cache_resource
def load_resources():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASSES_PATH):
        return None, []
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASSES_PATH, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return model, labels

# --- 7. UI PRINCIPAL ---
st.markdown('<div class="header-box"><h1>🛡️ AutoClaim AI • Expertiză Daune</h1></div>', unsafe_allow_html=True)

model, labels = load_resources()

with st.sidebar:
    st.markdown("### ⚙️ Configurare")
    sensitivity = st.slider("Sensibilitate AI", 10, 90, 25, help="Pragul minim de încredere pentru a lua în considerare o avarie.")
    uploaded_file = st.file_uploader("Imagine Close-Up (Detaliu)", type=["jpg", "png", "jpeg"])
    st.divider()
    if model: st.success("🟢 Sistem Online")
    else: st.error("🔴 Sistem Offline")

tab1, tab2, tab3 = st.tabs(["🕵️ Analiză Inteligentă", "🏆 Metrici", "🧪 Istoric"])

# TAB 1: LIVE DEMO
with tab1:
    if uploaded_file and model:
        c1, c2 = st.columns([1, 1.2], gap="large")
        
        with c1:
            st.markdown('### 📸 Imagine')
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, use_container_width=True)
            
        with c2:
            st.markdown('### 📊 Raport Expertiză')
            
            if st.button("🚀 EXECUTA ANALIZA", type="primary", use_container_width=True):
                with st.spinner("Analiză geometrică și de textură..."):
                    time.sleep(0.5)
                    proc_img = preprocess_image_for_model(image, target_size=(260, 260))
                    preds = model.predict(proc_img)
                    
                    detected = []
                    all_probs = {}
                    
                    for i, p in enumerate(preds[0]):
                        label_clean = labels[i].replace("_", " ").title()
                        prob_pct = p * 100
                        all_probs[label_clean] = prob_pct
                        
                        if p > sensitivity/100.0:
                            detected.append({'label': label_clean, 'score': prob_pct, 'raw': labels[i]})
                    
                    # Sortăm după scor pentru a vedea "ce e cel mai evident"
                    detected.sort(key=lambda x: x['score'], reverse=True)
                    
                    st.session_state['res'] = {'det': detected, 'probs': all_probs}

            if 'res' in st.session_state:
                res = st.session_state['res']
                
                if res['det']:
                    # 1. Calculăm Severitatea GLOBALĂ (Safety First)
                    sev_text, sev_action, sev_color = calculate_global_severity(res['det'])
                    
                    # Avaria dominantă (cea mai vizibilă)
                    primary_damage = res['det'][0]

                    # Afișare DASHBOARD
                    st.markdown(f"""
                        <div style="text-align: center; margin-bottom: 20px;">
                            <p style="color: #888; margin-bottom: 5px;">Avarie Principală Identificată:</p>
                            <h1 style="color: white; font-size: 2.5rem; margin: 0;">{primary_damage['label']}</h1>
                            <p style="color: #aaa;">(Încredere: {primary_damage['score']:.1f}%)</p>
                        </div>
                        
                        <div class="severity-box" style="background-color: {sev_color}22; border: 2px solid {sev_color}; color: {sev_color};">
                            SEVERITATE GLOBALĂ: {sev_text}<br>
                            <span style="font-size: 1rem; font-weight: normal; color: #ddd;">{sev_action}</span>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.divider()

                    # 2. Lista completă de probleme detectate
                    if len(res['det']) > 1:
                        st.write("#### ⚠️ Alte probleme detectate:")
                        for d in res['det'][1:]:
                            lvl, _, _, col = get_severity_details(d['raw'])
                            st.markdown(f"<span style='color:{col}'>• <b>{d['label']}</b> ({d['score']:.1f}%)</span>", unsafe_allow_html=True)
                        st.divider()

                    # 3. Grafic Probabilități
                    st.markdown("#### 📉 Analiză Probabilistică Completă")
                    chart_df = pd.DataFrame(list(res['probs'].items()), columns=["Clasă", "%"])
                    st.bar_chart(chart_df.set_index("Clasă"), color="#444444")
                    
                else:
                    st.warning("⚠️ Nicio avarie nu a depășit pragul de sensibilitate.")
                    st.info("Recomandare: Scade sensibilitatea din meniul lateral sau încarcă o poză mai clară.")

# TAB 2: METRICI (JSON + Confusion Matrix)
with tab2:
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f: metrics = json.load(f)
        c1, c2 = st.columns(2)
        c1.metric("Acuratețe Test", f"{metrics['test_accuracy']:.2%}")
        c2.metric("Loss", f"{metrics['test_loss']:.4f}")
        st.image(CONFUSION_MATRIX_PATH, caption="Matrice Confuzie", use_container_width=True)

with tab3:
    if os.path.exists(HISTORY_PATH):
        st.dataframe(pd.read_csv(HISTORY_PATH), use_container_width=True)