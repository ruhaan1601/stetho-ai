import os
import gc
import json
import pickle
import requests
import urllib.request
import torch

# Limit CPU threads and turn off gradient calculation to stay under 512MB RAM
torch.set_num_threads(1)
torch.set_grad_enabled(False)

import gradio as gr
import torch.serialization
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

# Subclass PyTorch's own unpickler so persistent IDs (tensor storage) work cleanly
class PatchedUnpickler(torch.serialization._Unpickler):
    def find_class(self, module, name):
        if 'fasttransform' in module or name == 'Transform':
            class DummyTransform:
                def __init__(self, f=None, *args, **kwargs):
                    self.f = f if f is not None else (lambda x: x)
                def __call__(self, x, *args, **kwargs): return x
                def decode(self, x, *args, **kwargs): return x
                def setups(self, *args, **kwargs): pass
            return DummyTransform
        return super().find_class(module, name)

class CustomPickleModule:
    Unpickler = PatchedUnpickler
    load = pickle.load
    loads = pickle.loads

# Load acoustic learner cleanly on CPU
model = load_learner(MODEL_FILE, cpu=True, pickle_module=CustomPickleModule)
model.eval()
gc.collect()

# -------------------------------------------------------------
# 2. GEMINI LIVE CHATBOT (Direct REST API, Zero Conflict)
# -------------------------------------------------------------
gemini_api_key = os.environ.get("GEMINI_API_KEY")

SYSTEM_INSTRUCTION = (
    "You are the certified clinical triage guide for 'stethol.com', an advanced acoustic respiratory screening platform. "
    "Your goals: "
    "1. Discuss cough types (dry, wet, paroxysmal, barky), lung sounds (wheezing, crackles, stridor), and respiratory comfort tips. "
    "2. Automatically respond in whatever language the user asks in (English, Hindi, Hinglish, Tamil, Telugu, etc.). "
    "3. Maintain clear triage boundaries: provide informational triage, not a formal medical diagnosis. "
    "4. If symptoms indicate an emergency (shortness of breath at rest, blood in sputum, chest pain, high fever >3 days), urge immediate consultation with a doctor. "
    "5. Point users to stethol.com's partner services (ordering medicine or consulting an online doctor) when appropriate. "
    "Keep answers warm, concise, and structured with bullet points."
)

def dynamic_chat(message, history):
    if not gemini_api_key:
        return "⚠️ Chatbot API key not detected. Please verify GEMINI_API_KEY is configured in your Render Environment settings."
    
    try:
        contents = []
        for turn in history:
            user_msg = turn[0]
            bot_msg = turn[1]
            if user_msg:
                contents.append({"role": "user", "parts": [{"text": str(user_msg)}]})
            if bot_msg:
                contents.append({"role": "model", "parts": [{"text": str(bot_msg)}]})
        
        contents.append({"role": "user", "parts": [{"text": str(message)}]})

        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_api_key}"
        payload = {
            "system_instruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
            "contents": contents,
            "generationConfig": {"temperature": 0.3}
        }
        
        resp = requests.post(api_url, json=payload, timeout=20)
        data = resp.json()
        
        if "candidates" in data and len(data["candidates"]) > 0:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        elif "error" in data:
            return f"API Error: {data['error'].get('message', 'Unknown issue')}"
        else:
            return "Unable to process the query right now. Please try again."

    except Exception as e:
        return f"Service temporarily busy. Please try again shortly. (Error: {str(e)})"

# -------------------------------------------------------------
# 3. LOCALIZATION DICTIONARY (English, Hindi, Tamil, Telugu)
# -------------------------------------------------------------
I18N = {
    "English": {
        "title": "# 🩺 **stethol.com**",
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
        "title": "# 🩺 **stethol.com**",
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
    "தமிழ் (Tamil)": {
        "title": "# 🩺 **stethol.com**",
        "subtitle": "### *மேம்பட்ட ஒலி அடிப்படையிலான சுவாசப் பரிசோதனை மற்றும் வழிகாட்டல்*",
        "banner": "📢 விளம்பரம்: 1mg மற்றும் அப்பல்லோ பார்மசியில் 20% தள்ளுபடி பெற குறியீடு: HEALTH20",
        "tab_screener": "🎙️ ஆடியோ பரிசோதனை",
        "tab_chat": "💬 மெய்நிகர் நல்வாழ்வு வழிகாட்டி",
        "audio_label": "இருமல் அல்லது மூச்சு ஒலியைப் பதிவு செய்க",
        "btn_analyze": "⚡ ஒலி பகுப்பாய்வைத் தொடங்கு",
        "partner_header": "### 🏥 கூட்டாளர் மருத்துவச் சேவைகள்",
        "btn_pharmacy": "💊 மருந்துகளை ஆர்டர் செய்க",
        "btn_doctor": "👨‍⚕️ மருத்துவரை ஆன்லைனில் அணுகுக",
        "res_label": "பரிசோதனை முடிவு",
        "spec_label": "மெல்-அதிர்வெண் ஸ்பெக்ட்ரோகிராம்",
        "ad_box": '<small style="color:gray;">விளம்பரம்</small><p><strong>N95 முகக்கவசங்கள் மற்றும் நெபுலைசர்கள்</strong> — ஒரே நாளில் டெலிவரி.</p>',
        "quiet_err": "⚠️ ஒலி மிகவும் மெதுவாக உள்ளது. மைக்கை அருகில் வைத்து மீண்டும் முயற்சிக்கவும்.",
        "input_err": "⚠️ தயவுசெய்து முதலில் ஆடியோவை பதிவு செய்யவும் அல்லது பதிவேற்றவும்.",
        "patho": "🚨 சுவாசத்தில் மாறுபாடு கண்டறியப்பட்டது (வீசிங் / சளி ஒலி)",
        "normal": "✅ வழக்கமான சுவாச ஒலி / மூச்சுப்பாதை தெளிவாக உள்ளது",
        "fake": "🎭 செயற்கையான / தாமாகச் செய்த இருமல் கண்டறியப்பட்டது"
    },
    "తెలుగు (Telugu)": {
        "title": "# 🩺 **stethol.com**",
        "subtitle": "### *ఆధునిక శబ్ద శ్వాసకోశ పరీక్ష మరియు క్లినికల్ సహాయం*",
        "banner": "📢 ప్రాయోజితం: 1mg మరియు అపోలో ఫార్మసీలో 20% తగ్గింపు పొందండి | కోడ్: HEALTH20",
        "tab_screener": "🎙️ ఆడియో స్క్రీనర్",
        "tab_chat": "💬 వర్చువల్ హెల్త్ గైడ్",
        "audio_label": "దగ్గు లేదా శ్వాస శబ్దాన్ని రికార్డ్ చేయండి",
        "btn_analyze": "⚡ శబ్ద విశ్లేషణను ప్రారంభించండి",
        "partner_header": "### 🏥 భాగస్వామ్య ఆరోగ్య సేవలు",
        "btn_pharmacy": "💊 మందులను ఆర్డర్ చేయండి",
        "btn_doctor": "👨‍⚕️ ఆన్‌లైన్ వైద్యుడిని సంప్రదించండి",
        "res_label": "పరీక్ష ఫలితం",
        "spec_label": "మెల్-ఫ్రీక్వెన్సీ స్పెక్ట్రోగ్రామ్",
        "ad_box": '<small style="color:gray;">ప్రకటన</small><p><strong>N95 మాస్క్‌లు & నెబ్యులైజర్లు</strong> — అదే రోజు డెలివరీ లభ్యం.</p>',
        "quiet_err": "⚠️ శబ్దం చాలా తక్కువగా ఉంది. మైక్రోఫోన్‌ను దగ్గరగా ఉంచి మళ్లీ ప్రయత్నించండి.",
        "input_err": "⚠️ దయచేసి ముందుగా ఆడియోను రికార్డ్ చేయండి లేదా అప్‌లోడ్ చేయండి.",
        "patho": "🚨 శ్వాసకోశ అసమానత గుర్తించబడింది (వీజింగ్ / పిల్లికూతలు)",
        "normal": "✅ సాధారణ శ్వాస ధ్వని / శ్వాస మార్గాలు స్పష్టంగా ఉన్నాయి",
        "fake": "🎭 స్వచ్ఛంద / కృత్రిమ దగ్గు గుర్తించబడింది"
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
# 4. AUDIO INFERENCE LOGIC (Memory-Safe)
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
# 5. UI LAYOUT & ATTACHMENTS
# -------------------------------------------------------------
with gr.Blocks(css=custom_css, title="stethol.com") as demo:
    with gr.Row():
        with gr.Column(scale=8):
            header = gr.Markdown(f"{I18N['English']['title']}\n{I18N['English']['subtitle']}")
        with gr.Column(scale=4):
            lang_selector = gr.Dropdown(
                ["English", "हिन्दी (Hindi)", "தமிழ் (Tamil)", "తెలుగు (Telugu)"],
                value="English",
                label="🌐 Select Language / மொழியைத் தேர்ந்தெடுக்கவும் / భాషను ఎంచుకోండి"
            )
    
    banner = gr.HTML(f'<div class="sponsor-banner">{I18N["English"]["banner"]}</div>')

    with gr.Tabs():
        with gr.TabItem("🎙️ stethol.com Screener"):
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
            gr.ChatInterface(
                fn=dynamic_chat,
                description="Powered by Gemini: Ask real-time questions about your symptoms, cough characteristics, or recovery steps in English, Hindi, Tamil, Telugu, or Hinglish."
            )

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
