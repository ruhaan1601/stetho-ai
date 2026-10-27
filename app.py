import os
import gc
import sys
import types
import urllib.request
import torch

# Limit CPU threads and turn off gradient calculation to stay under 512MB RAM
torch.set_num_threads(1)
torch.set_grad_enabled(False)

# -------------------------------------------------------------
# FIX FOR 'fasttransform.transform' UNPICKLE ALIAS
# -------------------------------------------------------------
import fastcore.transform
sys.modules['fasttransform'] = fastcore
sys.modules['fasttransform.transform'] = fastcore.transform

import gradio as gr
from fastai.vision.all import *
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------------
# 1. AUTO-DOWNLOAD MODEL FROM GITHUB RELEASE
# -------------------------------------------------------------
MODEL_FILE = "cough_classifier_model.1.pkl"
MODEL_URL = "https://github.com/ruhaan1601/stetho-ai/releases/download/v1.0/cough_classifier_model.1.pkl"

if not os.path.exists(MODEL_FILE):
    print("Downloading model weights from GitHub release...")
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
    print("Model downloaded successfully!")

# Load model cleanly on CPU
model = load_learner(MODEL_FILE, cpu=True)
model.eval()
gc.collect()

# -------------------------------------------------------------
# 2. LOCALIZATION DICTIONARY (English, Hindi, Spanish)
# -------------------------------------------------------------
I18N = {
    "English": {
        "title": "# 🩺 **Stetho-AI**",
        "subtitle": "### *Advanced Acoustic Respiratory Screening & Clinical Triage*",
        "banner": "📢 SPONSORED: Get 20% Off at 1mg & Apollo Pharmacy with Code: HEALTH20",
        "tab_screener": "🎙️ Audio Screener",
        "tab_chat": "💬 Virtual Health Guide",
        "audio_label": "Record Cough or Breath Sound",
        "btn_analyze": "⚡ Analyze Acoustic Signature",
        "partner_header": "### 🏥 Partner Healthcare Services",
        "btn_pharmacy": "💊 Order Prescriptions",
        "btn_doctor": "👨‍⚕️ Consult Online Doctor",
        "res_label": "Assessment Result",
        "spec_label": "Mel-Frequency Spectrogram",
        "ad_box": '<small style="color:gray;">ADVERTISEMENT</small><p><strong>N95 Protective Masks & Home Nebulizers</strong> — Same Day Delivery Available.</p>',
        "quiet_err": "⚠️ Audio too quiet. Hold microphone closer and retry.",
        "input_err": "⚠️ Please provide an audio recording first.",
        "patho": "🚨 Respiratory Anomaly Detected (Wheeze / Crackle)",
        "normal": "✅ Normal Respiratory Sound / Clear Airways",
        "fake": "🎭 Voluntary / Simulated Cough Detected"
    },
    "हिन्दी (Hindi)": {
        "title": "# 🩺 **Stetho-AI**",
        "subtitle": "### *उन्नत ध्वनिक श्वसन जांच एवं नैदानिक सहायता*",
        "banner": "📢 प्रायोजित: 1mg और अपोलो फार्मेसी पर 20% छूट पाएं | कोड: HEALTH20",
        "tab_screener": "🎙️ ऑडियो जांच",
        "tab_chat": "💬 वर्चुअल स्वास्थ्य सहायक",
        "audio_label": "खांसी या सांस की आवाज रिकॉर्ड करें",
        "btn_analyze": "⚡ ध्वनिक विश्लेषण शुरू करें",
        "partner_header": "### 🏥 पार्टनर स्वास्थ्य सेवाएं",
        "btn_pharmacy": "💊 दवाइयां ऑर्डर करें",
        "btn_doctor": "👨‍⚕️ डॉक्टर से परामर्श लें",
        "res_label": "जांच परिणाम",
        "spec_label": "मेल-फ़्रीक्वेंसी स्पेक्ट्रोग्राम",
        "ad_box": '<small style="color:gray;">विज्ञापन</small><p><strong>N95 मास्क और नेबुलाइजर</strong> — उसी दिन डिलीवरी उपलब्ध।</p>',
        "quiet_err": "⚠️ आवाज़ बहुत धीमी है। माइक पास लाकर दोबारा प्रयास करें।",
        "input_err": "⚠️ कृपया पहले ऑडियो रिकॉर्ड या अपलोड करें।",
        "patho": "🚨 श्वसन असामान्यता पाई गई (घरघराहट / संक्रमण संकेत)",
        "normal": "✅ सामान्य श्वसन ध्वनि / वायुमार्ग साफ है",
        "fake": "🎭 स्वैच्छिक / बनावटी खांसी पहचानी गई"
    },
    "Español": {
        "title": "# 🩺 **Stetho-AI**",
        "subtitle": "### *Detección Acústica Respiratoria y Triaje Clínico*",
        "banner": "📢 PATROCINADO: 20% de descuento en farmacias asociadas | Código: HEALTH20",
        "tab_screener": "🎙️ Analizador de Audio",
        "tab_chat": "💬 Guía Virtual de Salud",
        "audio_label": "Grabar sonido de tos o respiración",
        "btn_analyze": "⚡ Analizar Firma Acústica",
        "partner_header": "### 🏥 Servicios Médicos Asociados",
        "btn_pharmacy": "💊 Pedir Medicamentos",
        "btn_doctor": "👨‍⚕️ Consultar Médico Online",
        "res_label": "Resultado de la Evaluación",
        "spec_label": "Espectrograma de Frecuencia Mel",
        "ad_box": '<small style="color:gray;">PUBLICIDAD</small><p><strong>Mascarillas N95 y Nebulizadores</strong> — Entrega el mismo día.</p>',
        "quiet_err": "⚠️ Audio demasiado bajo. Acerque el micrófono e intente de nuevo.",
        "input_err": "⚠️ Por favor, grabe o suba un audio primero.",
        "patho": "🚨 Anomalía Respiratoria Detectada (Sibilancia / Estertor)",
        "normal": "✅ Sonido Respiratorio Normal / Vías Aéreas Despejadas",
        "fake": "🎭 Tos Voluntaria / Simulada Detectada"
    }
}

custom_css = """
footer {visibility: hidden !important;}
.gradio-container {max-width: 950px !important; margin: auto !important;}
.sponsor-banner {
    background: linear-gradient(90deg, #0f172a, #2563eb);
    color: white;
    padding: 12px;
    border-radius: 8px;
    text-align: center;
    font-weight: bold;
    margin: 10px 0;
}
.ad-box {
    border: 2px dashed #9ca3af;
    padding: 10px;
    text-align: center;
    border-radius: 6px;
    background-color: rgba(156, 163, 175, 0.08);
}
"""

# -------------------------------------------------------------
# 3. AUDIO INFERENCE LOGIC
# -------------------------------------------------------------
def analyze_audio(audio_path, language):
    lang_dict = I18N.get(language, I18N["English"])
    if audio_path is None:
        return lang_dict["input_err"], None
    
    temp_img = "eval_spec.png"
    try:
        signal, sr = librosa.load(audio_path, sr=22050, duration=3.0)
        if np.max(np.abs(signal)) < 0.02:
            return lang_dict["quiet_err"], None
            
        mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        fig = plt.figure(figsize=(3, 3))
        plt.axis('off')
        librosa.display.specshow(mel_spec_db, sr=sr, fmax=8000)
        fig.savefig(temp_img, bbox_inches='tight', pad_inches=0, dpi=80)
        plt.close(fig)
        plt.close('all')
        
        pred, pred_idx, probs = model.predict(temp_img)
        labels = model.dls.vocab
        prob_dict = {labels[i]: float(probs[i]) for i in range(len(labels))}
        
        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        top_label, top_val = sorted_probs[0]
        
        mapping = {
            "pathological": lang_dict["patho"],
            "normal_reflex": lang_dict["normal"],
            "voluntary_fake": lang_dict["fake"]
        }
        
        verdict = mapping.get(top_label, top_label)
        
        del signal, mel_spec, mel_spec_db, probs
        gc.collect()
        
        return verdict, temp_img

    except Exception as e:
        plt.close('all')
        gc.collect()
        return f"Error: {str(e)}", None

# -------------------------------------------------------------
# 4. CHATBOT ASSISTANT
# -------------------------------------------------------------
def ai_assistant(message, history):
    msg = message.lower().strip()
    if any(k in msg for k in ["doctor", "consult", "hospital", "dr", "appointment", "डॉक्टर", "médico", "medico"]):
        return ("🩺 **Medical Guidance / चिकित्सा सलाह:** If you have persistent coughing, shortness of breath, or fever lasting over 3 days, "
                "consult a certified physician immediately. Use our **Consult Online Doctor** button above to connect with a specialist.")
    elif any(k in msg for k in ["medicine", "syrup", "tablet", "pill", "antibiotic", "दवा", "दवाई", "medicamento"]):
        return ("💊 **Pharmacy & Medication Notice:** Never take antibiotics or prescription bronchodilators without a doctor's examination. "
                "For soothing throat irritation, warm saline gargles, honey-lemon water, or approved OTC lozenges can help. Order verified supplies via our **Order Prescriptions** link above.")
    elif any(k in msg for k in ["fake", "voluntary", "pretend", "simulated", "नकली", "falsa", "voluntaria"]):
        return ("🎭 **Acoustic Science:** Stetho-AI analyzes glottal rise times and frequency turbulence. Voluntary coughs lack "
                "the sudden subglottic pressure burst and fluid micro-crackles produced during genuine involuntary airway reflexes.")
    elif any(k in msg for k in ["wet", "dry", "mucus", "phlegm", "कफ", "बलगम", "moco", "seca"]):
        return ("🫁 **Cough Variations:**\n"
                "- **Dry Cough:** Irritation in the upper airway without mucus. Common in viral infections, allergies, or early asthma.\n"
                "- **Wet / Productive Cough:** Airway contains mucus/phlegm. Typical in chest colds, bronchitis, or bacterial infections.")
    elif any(k in msg for k in ["wheeze", "crackle", "sound", "whistle", "घरघराहट", "sibilancia"]):
        return ("🔍 **Acoustic Markers:** Wheezing produces continuous, high-pitched whistling due to narrowed bronchial passages. "
                "Crackles are intermittent popping sounds generated when fluid-filled airways pop open during inhalation.")
    elif any(k in msg for k in ["hi", "hello", "hey", "नमस्ते", "hola"]):
        return "👋 Hello! I am your Stetho-AI Health Assistant. How can I assist you with your cough assessment, symptoms, or finding medical care today?"
    else:
        return ("I understand you are asking about respiratory health. If you are experiencing symptoms like chest tightness, "
                "wheezing, or breathing difficulty, please record an audio sample in the screener tab or consult a physician directly.")

def change_language(selected_lang):
    d = I18N.get(selected_lang, I18N["English"])
    banner_html = f'<div class="sponsor-banner">{d["banner"]}</div>'
    ad_html = f'<div class="ad-box" style="width:100%; margin-top:15px;">{d["ad_box"]}</div>'
    header_md = f"{d['title']}\n{d['subtitle']}"
    return [
        gr.update(value=header_md),
        gr.update(value=banner_html),
        gr.update(label=d["audio_label"]),
        gr.update(value=d["btn_analyze"]),
        gr.update(value=d["partner_header"]),
        gr.update(label=d["res_label"]),
        gr.update(label=d["spec_label"]),
        gr.update(value=ad_html)
    ]

# -------------------------------------------------------------
# 5. UI LAYOUT
# -------------------------------------------------------------
with gr.Blocks(css=custom_css, title="Stetho-AI") as demo:
    with gr.Row():
        with gr.Column(scale=8):
            header = gr.Markdown(f"{I18N['English']['title']}\n{I18N['English']['subtitle']}")
        with gr.Column(scale=4):
            lang_selector = gr.Dropdown(
                ["English", "हिन्दी (Hindi)", "Español"],
                value="English",
                label="🌐 Select Language / भाषा चुनें"
            )
    
    banner = gr.HTML(f'<div class="sponsor-banner">{I18N["English"]["banner"]}</div>')

    with gr.Tabs():
        with gr.TabItem("🎙️ Stetho Screener"):
            with gr.Row():
                with gr.Column():
                    audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label=I18N["English"]["audio_label"])
                    analyze_btn = gr.Button(I18N["English"]["btn_analyze"], variant="primary")
                    gr.Markdown("---")
                    partner_title = gr.Markdown(I18N["English"]["partner_header"])
                    with gr.Row():
                        gr.HTML('<a href="https://www.1mg.com" target="_blank" style="text-decoration:none;"><button style="width:100%; padding:10px; border-radius:6px; background:#10b981; color:white; border:none; cursor:pointer; font-weight:bold;">💊 Order Prescriptions</button></a>')
                        gr.HTML('<a href="https://www.practo.com" target="_blank" style="text-decoration:none;"><button style="width:100%; padding:10px; border-radius:6px; background:#6366f1; color:white; border:none; cursor:pointer; font-weight:bold;">👨‍⚕️ Consult Online Doctor</button></a>')

                with gr.Column():
                    result_box = gr.Textbox(label=I18N["English"]["res_label"], interactive=False)
                    spec_output = gr.Image(label=I18N["English"]["spec_label"], type="filepath")

            with gr.Row():
                ad_box = gr.HTML(f'<div class="ad-box" style="width:100%; margin-top:15px;">{I18N["English"]["ad_box"]}</div>')

        with gr.TabItem("💬 Virtual Health Guide"):
            gr.ChatInterface(fn=ai_assistant, description="Ask Stetho-AI questions about cough sounds, symptoms, home care, or finding a specialist.")

    lang_selector.change(
        fn=change_language,
        inputs=[lang_selector],
        outputs=[header, banner, audio_input, analyze_btn, partner_title, result_box, spec_output, ad_box]
    )

    analyze_btn.click(
        fn=analyze_audio,
        inputs=[audio_input, lang_selector],
        outputs=[result_box, spec_output]
    )

port = int(os.environ.get("PORT", 7860))
demo.launch(server_name="0.0.0.0", server_port=port)
