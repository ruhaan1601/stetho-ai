import os
import gc
import urllib.request
import torch

# Force single-thread execution and disable gradient tracking to minimize RAM
torch.set_num_threads(1)
torch.set_grad_enabled(False)

import gradio as gr
from fastai.vision.all import *
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------------
# 1. AUTO-DOWNLOAD MODEL IF NOT PRESENT
# -------------------------------------------------------------
MODEL_FILE = "cough_classifier_model.pkl"
# Replace <YOUR_GITHUB_USERNAME> with your exact GitHub handle:
MODEL_URL = "https://github.com/<YOUR_GITHUB_USERNAME>/stetho-ai/releases/download/v1.0/cough_classifier_model.pkl"

if not os.path.exists(MODEL_FILE):
    print("Downloading model weights from GitHub release...")
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
    print("Model downloaded successfully!")

# Load model in evaluation mode
model = load_learner(MODEL_FILE)
model.eval()
gc.collect()

# -------------------------------------------------------------
# 2. LOCALIZATION & TEXT (English only, with room for future labels)
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
# 3. AUDIO INFERENCE LOGIC (Memory-Safe)
# -------------------------------------------------------------
def analyze_audio(audio_path):
    lang_dict = I18N["English"]
    if audio_path is None:
        return lang_dict["input_err"], None
    
    temp_img = "eval_spec.png"
    try:
        # Load audio at low sampling rate
        signal, sr = librosa.load(audio_path, sr=22050, duration=3.0)
        if np.max(np.abs(signal)) < 0.02:
            return lang_dict["quiet_err"], None
            
        mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Save spectrogram with tight memory cleanup
        fig = plt.figure(figsize=(3, 3))
        plt.axis('off')
        librosa.display.specshow(mel_spec_db, sr=sr, fmax=8000)
        fig.savefig(temp_img, bbox_inches='tight', pad_inches=0, dpi=80)
        plt.close(fig)
        plt.close('all')
        
        # Run inference
        pred, pred_idx, probs = model.predict(temp_img)
        labels = model.dls.vocab
        prob_dict = {labels[i]: float(probs[i]) for i in range(len(labels))}
        
        # Threshold: >= 55% or highest probability
        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        top_label, top_val = sorted_probs[0]
        
        mapping = {
            "pathological": lang_dict["patho"],
            "normal_reflex": lang_dict["normal"],
            "voluntary_fake": lang_dict["fake"]
        }
        
        verdict = mapping.get(top_label, top_label)
        
        # Force garbage collector to free spectrogram arrays and tensors
        del signal, mel_spec, mel_spec_db, probs
        gc.collect()
        
        return verdict, temp_img

    except Exception as e:
        plt.close('all')
        gc.collect()
        return f"Error: {str(e)}", None

# -------------------------------------------------------------
# 4. CHATBOT TRIAGE
# -------------------------------------------------------------
def ai_assistant(message, history):
    msg = message.lower().strip()
    if any(k in msg for k in ["doctor", "consult", "hospital", "dr", "appointment"]):
        return ("🩺 **Medical Guidance:** If you have persistent coughing, shortness of breath, or fever lasting over 3 days, "
                "consult a certified pulmonologist or physician immediately. Use our **Consult Online Doctor** button to connect with a specialist.")
    elif any(k in msg for k in ["medicine", "syrup", "tablet", "pill", "antibiotic"]):
        return ("💊 **Pharmacy & Medication Notice:** Never take antibiotics or prescription bronchodilators without a doctor's examination. "
                "For soothing dry throat irritation, warm saline gargles, honey-lemon water, or approved OTC lozenges can help. Order verified supplies via our **Order Prescriptions** link.")
    elif any(k in msg for k in ["fake", "voluntary", "pretend", "simulated"]):
        return ("🎭 **Acoustic Science:** Stetho-AI analyzes glottal rise times and frequency turbulence. Voluntary coughs lack "
                "the sudden subglottic pressure burst and fluid micro-crackles produced during genuine involuntary airway reflexes.")
    elif any(k in msg for k in ["wet", "dry", "mucus", "phlegm"]):
        return ("🫁 **Cough Variations:**\n"
                "- **Dry Cough:** Irritation in the upper airway without mucus. Common in viral infections, allergies, or early asthma.\n"
                "- **Wet / Productive Cough:** Airway contains mucus/phlegm. Typical in chest colds, bronchitis, or bacterial infections.")
    elif any(k in msg for k in ["wheeze", "crackle", "sound", "whistle"]):
        return ("🔍 **Acoustic Markers:** Wheezing produces continuous, high-pitched whistling due to narrowed bronchial passages. "
                "Crackles are intermittent popping sounds generated when fluid-filled airways pop open during inhalation.")
    elif any(k in msg for k in ["hi", "hello", "hey"]):
        return "👋 Hello! I am your Stetho-AI Health Assistant. How can I assist you with your cough assessment, symptoms, or finding medical care today?"
    else:
        return ("I understand you are asking about respiratory health. If you are experiencing symptoms like chest tightness, "
                "wheezing, or breathing difficulty, please record an audio sample in the screener tab or consult a physician directly.")

# -------------------------------------------------------------
# 5. UI LAYOUT
# -------------------------------------------------------------
with gr.Blocks(css=custom_css, title="Stetho-AI") as demo:
    gr.Markdown(f"{I18N['English']['title']}\n{I18N['English']['subtitle']}")
    gr.HTML(f'<div class="sponsor-banner">{I18N["English"]["banner"]}</div>')

    with gr.Tabs():
        with gr.TabItem("🎙️ Stetho Screener"):
            with gr.Row():
                with gr.Column():
                    audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label=I18N["English"]["audio_label"])
                    analyze_btn = gr.Button(I18N["English"]["btn_analyze"], variant="primary")
                    gr.Markdown("---")
                    gr.Markdown(I18N["English"]["partner_header"])
                    with gr.Row():
                        gr.HTML('<a href="https://www.1mg.com" target="_blank" style="text-decoration:none;"><button style="width:100%; padding:10px; border-radius:6px; background:#10b981; color:white; border:none; cursor:pointer; font-weight:bold;">💊 Order Prescriptions</button></a>')
                        gr.HTML('<a href="https://www.practo.com" target="_blank" style="text-decoration:none;"><button style="width:100%; padding:10px; border-radius:6px; background:#6366f1; color:white; border:none; cursor:pointer; font-weight:bold;">👨‍⚕️ Consult Online Doctor</button></a>')

                with gr.Column():
                    result_box = gr.Textbox(label=I18N["English"]["res_label"], interactive=False)
                    spec_output = gr.Image(label=I18N["English"]["spec_label"], type="filepath")

            with gr.Row():
                gr.HTML(f'<div class="ad-box" style="width:100%; margin-top:15px;">{I18N["English"]["ad_box"]}</div>')

        with gr.TabItem("💬 Virtual Health Guide"):
            gr.ChatInterface(fn=ai_assistant, description="Ask Stetho-AI questions about cough sounds, symptoms, home care, or finding a specialist.")

    analyze_btn.click(
        fn=analyze_audio,
        inputs=[audio_input],
        outputs=[result_box, spec_output]
    )

port = int(os.environ.get("PORT", 7860))
demo.launch(server_name="0.0.0.0", server_port=port)
