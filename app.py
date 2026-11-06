import os
import io
import time
import requests
import numpy as np
import soundfile as sf
import librosa
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

import gradio as gr
import google.generativeai as genai
import uvicorn
from fastapi import FastAPI

# ---------------------------------------------------------
# 1. OPTIMIZE PYTORCH THREADING FOR CLOUD INSTANCES
# ---------------------------------------------------------
# Render's free tier provides a fraction of a CPU core. Capping PyTorch
# to 1 thread eliminates CPU thrashing and cuts processing latency.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# ---------------------------------------------------------
# 2. GEMINI API SETUP
# ---------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    except Exception as e:
        gemini_model = None
        print(f"Gemini Init Warning: {e}")
else:
    gemini_model = None
    print("Warning: GEMINI_API_KEY environment variable not found.")

# ---------------------------------------------------------
# 3. PYTORCH RESNET MODEL INITIALIZATION
# ---------------------------------------------------------
DEFAULT_LABELS = [
    "Healthy / Normal",
    "Asthma / Bronchitis",
    "COPD",
    "Pneumonia",
    "COVID-19 Infection",
    "Pertussis (Whooping Cough)"
]

WEIGHTS_URL = os.environ.get("MODEL_WEIGHTS_URL", "").strip()
MODEL_PATH = "stethologic_resnet18.pt"

def load_inference_model():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(DEFAULT_LABELS))

    # Download model weights if configured via URL and missing locally
    if WEIGHTS_URL and not os.path.exists(MODEL_PATH):
        try:
            print(f"Downloading PyTorch model weights from {WEIGHTS_URL}...")
            res = requests.get(WEIGHTS_URL, timeout=45)
            if res.status_code == 200:
                with open(MODEL_PATH, "wb") as f:
                    f.write(res.content)
                print("Model weights successfully saved.")
        except Exception as e:
            print(f"Failed to fetch remote weights: {e}")

    if os.path.exists(MODEL_PATH):
        try:
            state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
            model.load_state_dict(state_dict)
            print("Loaded trained model weights successfully.")
        except Exception as e:
            print(f"Could not load state_dict: {e}. Running with initialized weights.")
    else:
        print("No local weight file found. Initialized baseline ResNet-18.")

    model.eval()
    return model

model = load_inference_model()

# ---------------------------------------------------------
# 4. HIGH-SPEED AUDIO PIPELINE
# ---------------------------------------------------------
img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def process_audio_to_mel(audio_path):
    """
    Decodes audio using soundfile and resamples via kaiser_fast.
    Runs in ~1-2 seconds on low-CPU cloud instances.
    """
    try:
        y, sr = sf.read(audio_path)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        y = y.astype(np.float32)
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000, res_type="kaiser_fast")
            sr = 16000
    except Exception:
        # Fallback to librosa if soundfile fails format negotiation
        y, sr = librosa.load(audio_path, sr=16000, res_type="kaiser_fast")

    # Guard against completely empty or silent audio
    if len(y) == 0:
        raise ValueError("Audio recording is empty.")

    # Compute Mel-Spectrogram
    n_mels = 128
    n_fft = 1024
    hop_length = 512
    melspec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    melspec_db = librosa.power_to_db(melspec, ref=np.max)

    # Normalize to 0-255 uint8 image format for ResNet ingestion
    norm_spec = (melspec_db - melspec_db.min()) / (melspec_db.max() - melspec_db.min() + 1e-8)
    img_array = (norm_spec * 255).astype(np.uint8)
    img = Image.fromarray(img_array).convert("RGB")

    return img

def analyze_cough(audio_path):
    if not audio_path:
        return (
            "No audio provided.",
            "Please record or upload an audio file (.wav, .mp3).",
            None,
            {},
            None
        )

    try:
        # 1. Audio Processing & Spectrogram Generation
        spec_img = process_audio_to_mel(audio_path)
        tensor_in = img_transforms(spec_img).unsqueeze(0)

        # 2. PyTorch ResNet-18 Inference
        with torch.no_grad():
            outputs = model(tensor_in)
            probabilities = torch.softmax(outputs, dim=1).squeeze().tolist()

        if isinstance(probabilities, float):
            probabilities = [probabilities]

        confidences = {
            DEFAULT_LABELS[i]: float(prob)
            for i, prob in enumerate(probabilities)
        }
        primary_condition = max(confidences, key=confidences.get)
        confidence_pct = confidences[primary_condition] * 100

        diag_assessment = (
            f"**Indicative Primary Pattern:** {primary_condition}\n"
            f"**Acoustic Confidence:** {confidence_pct:.1f}%"
        )

        # 3. Gemini Clinical Context Generation
        if gemini_model:
            prompt = (
                f"You are StethoLogic's diagnostic assistant. An acoustic screener evaluated a cough sample "
                f"and indicated patterns resembling: '{primary_condition}' with confidence score {confidence_pct:.1f}%.\n"
                f"Provide a clear, clinical symptom breakdown explaining why this acoustic condition is indicated, "
                f"what physiological traits this cough usually presents, and standard recommended medical follow-ups. "
                f"Include a brief reminder that this screener is for investigational triage and not a definitive diagnosis."
            )
            try:
                response = gemini_model.generate_content(prompt)
                symptom_breakdown = response.text
            except Exception as e:
                symptom_breakdown = (
                    f"Gemini API request failed ({e}).\n\n"
                    f"Primary indication observed: {primary_condition}. "
                    f"Please consult a certified pulmonologist or physician for full clinical auscultation."
                )
        else:
            symptom_breakdown = (
                f"Acoustic classification completed: **{primary_condition}**.\n\n"
                f"*Note: Add a valid `GEMINI_API_KEY` to your environment variables to enable "
                f"automated medical explanations.*"
            )

        return (
            diag_assessment,
            symptom_breakdown,
            confidences,
            spec_img
        )

    except Exception as err:
        return (
            "Processing Error",
            f"An error occurred while analyzing the audio: {str(err)}",
            {},
            None
        )

def chat_with_dr_info(user_message, chat_history):
    if not user_message:
        return "", chat_history

    if not gemini_model:
        chat_history.append((user_message, "Dr. Info requires a valid GEMINI_API_KEY to be configured in Render."))
        return "", chat_history

    system_prompt = (
        "You are Dr. Info, an AI pulmonology screener consultant on StethoLogic. "
        "Answer patient questions clearly, informatively, and compassionately regarding respiratory health, "
        "cough patterns, and standard tests. Always advise consulting a medical professional for formal diagnoses."
    )

    try:
        full_query = f"{system_prompt}\n\nPatient: {user_message}\nDr. Info:"
        response = gemini_model.generate_content(full_query)
        chat_history.append((user_message, response.text))
    except Exception as e:
        chat_history.append((user_message, f"Communication error: {str(e)}"))

    return "", chat_history

# ---------------------------------------------------------
# 5. GRADIO INTERFACE LAYOUT
# ---------------------------------------------------------
custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate"
)

with gr.Blocks(theme=custom_theme, title="StethoLogic Screener") as demo:
    gr.Markdown(
        """
        # 🩺 StethoLogic Screener
        **AI-Powered Acoustic Respiratory Triage & Analysis**
        """
    )

    with gr.Tabs():
        with gr.TabItem("🎙️ StethoLogic Screener"):
            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.Audio(
                        sources=["upload", "microphone"],
                        type="filepath",
                        label="Record Cough or Respiration Sound"
                    )
                    analyze_btn = gr.Button("⚡ Analyze Acoustic Signature", variant="primary")

                    gr.Markdown("---")
                    gr.Markdown("### 🏥 Connected Healthcare Services")
                    with gr.Row():
                        order_rx_btn = gr.Button("💊 Order Prescriptions")
                        consult_btn = gr.Button("👨‍⚕️ Consult Pulmonologist")

                with gr.Column(scale=1):
                    diagnostic_output = gr.Markdown("### Diagnostic Assessment\n*Awaiting audio input...*")
                    breakdown_output = gr.Markdown("### Acoustic Symptom Breakdown\n*Results will appear here.*")
                    prob_output = gr.Label(num_top_classes=5, label="Differential Probability Breakdown")
                    spectrogram_output = gr.Image(label="Mel-Frequency Spectrogram", type="pil")

            analyze_btn.click(
                fn=analyze_cough,
                inputs=[audio_input],
                outputs=[diagnostic_output, breakdown_output, prob_output, spectrogram_output]
            )

        with gr.TabItem("💬 Dr. Info"):
            gr.Markdown("### Consult Dr. Info about respiratory symptoms, cough signs, and health questions.")
            chatbot = gr.Chatbot(height=420)
            with gr.Row():
                chat_msg = gr.Textbox(placeholder="Ask a question about your symptoms...", show_label=False, scale=4)
                send_btn = gr.Button("Send", variant="primary", scale=1)

            send_btn.click(fn=chat_with_dr_info, inputs=[chat_msg, chatbot], outputs=[chat_msg, chatbot])
            chat_msg.submit(fn=chat_with_dr_info, inputs=[chat_msg, chatbot], outputs=[chat_msg, chatbot])

# ---------------------------------------------------------
# 6. FASTAPI MOUNT & DIRECT UVICORN ENTRYPOINT
# ---------------------------------------------------------
# Mounting Gradio to FastAPI runs standard ASGI endpoints and avoids
# Gradio 4.x's internal localhost reachability check that triggered the
# "ValueError: When localhost is not accessible" restart loop.

port = int(os.environ.get("PORT", 10000))
fastapi_app = FastAPI()
fastapi_app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(fastapi_app, host="0.0.0.0", port=port)
