import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import noisereduce as nr
from scipy.signal import butter, lfilter
from io import BytesIO
import streamlit as st

st.set_page_config(page_title="Styled Noise Filter App", page_icon="🎧")

# CSS override
st.markdown("""
    <style>
        html, body, [data-testid="stAppViewContainer"] {
            background: linear-gradient(to bottom right, black, gray) !important;
        }

        [data-testid="stHeader"], .block-container {
            background: transparent !important;
        }

        .custom-title {
            background-color: black;
            color: white;
            padding: 1rem;
            text-align: center;
            border-radius: 12px;
            margin-bottom: 1.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="custom-title"><h1>Noise Filter App 🎧</h1></div>', unsafe_allow_html=True)

st.write("If you see a black-to-gray background and a white title inside a black box, styling is working.")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Noise Filter App 🎧", page_icon="🎧")

# ── Global CSS injection ──────────────────────────────────────────────────────
st.markdown("""
    <style>
        /* Full‑page dark gradient */
        html, body, [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #000000 0%, #4d4d4d 100%) !important;
        }

        /* Make default white blocks transparent */
        [data-testid="stHeader"], .block-container {
            background: transparent !important;
        }

        /* Title badge */
        .custom-title {
            background-color: #000000;
            padding: 1rem 1.5rem;
            border-radius: 10px;
            text-align: center;
            margin: 1rem 0 2rem 0;
        }
        .custom-title h1 {
            color: #ffffff;
            font-size: 2.5rem;
            margin: 0;
        }
    </style>
""", unsafe_allow_html=True)

# ── Top title ─────────────────────────────────────────────────────────────────
st.markdown('<div class="custom-title"><h1>Noise Filter App 🎧</h1></div>',
            unsafe_allow_html=True)

# ── Intro text ────────────────────────────────────────────────────────────────
st.write("""
Welcome to **Noise Filter App** – a simple Streamlit tool that removes background noise from your WAV recordings.

**How to use**  
1. Head to the **New Journey** page to upload a noisy `.wav` file.  
2. Wait a few seconds while the app applies spectral‑gating noise reduction.  
3. Jump to **Filtered Output** to compare waveforms, listen, and download the cleaned file.
""")
st.info("Ready to begin? Click **New Journey** in the sidebar 👉")

# ── DSP helper functions ──────────────────────────────────────────────────────
def normalize_audio(audio):
    max_val = np.max(np.abs(audio))
    return audio if max_val == 0 else audio / max_val * 0.9

def reduce_noise(audio, sr):
    return nr.reduce_noise(y=audio, sr=sr)

def bandpass_filter(audio, sr=16000, lowcut=500.0, highcut=2800.0):
    nyq = 0.5 * sr
    b, a = butter(6, [lowcut/nyq, highcut/nyq], btype='band')
    return lfilter(b, a, audio)

def amplify_audio(audio, gain=2.0):
    return np.clip(audio * gain, -1.0, 1.0)

def plot_waveform(audio, sr, title):
    fig, ax = plt.subplots()
    t = np.linspace(0, len(audio) / sr, len(audio))
    ax.plot(t, audio)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

# ── Audio processor for mic ───────────────────────────────────────────────────
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame):
        audio = frame.to_ndarray().flatten().astype(np.float32) / 32000.0
        self.frames.append(audio)
        return frame

# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("🎤 Speech Preprocessing (Mic & File Upload)")

sr = 48000
input_method = st.radio("Choose input method:",
                        ["🎙 Record via Microphone", "📁 Upload WAV File"])

def process_and_display(audio):
    # Original
    st.subheader("🔊 Original Audio")
    buf_orig = BytesIO()
    sf.write(buf_orig, audio, sr, format='wav')
    st.audio(buf_orig)
    plot_waveform(audio, sr, "Original Audio")

    # Cleaning
    audio = normalize_audio(audio)
    audio = reduce_noise(audio, sr)
    audio = bandpass_filter(audio, sr)
    audio = amplify_audio(audio)
    audio = normalize_audio(audio)

    # Cleaned
    st.subheader("🧼 Cleaned Audio")
    buf_clean = BytesIO()
    sf.write(buf_clean, audio, sr, format='wav')
    st.audio(buf_clean)
    plot_waveform(audio, sr, "Cleaned Audio")

    st.download_button("⬇️ Download Cleaned Audio",
                       buf_clean.getvalue(),
                       "cleaned_audio.wav",
                       mime="audio/wav")

# File upload branch
if input_method == "📁 Upload WAV File":
    uploaded = st.file_uploader("Upload a WAV file", type=["wav"])
    if uploaded:
        y, _ = librosa.load(uploaded, sr=sr, mono=True)
        process_and_display(y)

# Microphone branch
elif input_method == "🎙 Record via Microphone":
    st.info("Click **Start** and speak for 5–10 seconds, then press **Process**.")
    webrtc_ctx = webrtc_streamer(
        key="mic",
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
    )

    if st.button("✅ Process Mic Recording"):
        if webrtc_ctx and webrtc_ctx.state.playing and webrtc_ctx.audio_processor:
            raw_audio = np.concatenate(webrtc_ctx.audio_processor.frames)
            if len(raw_audio) < sr * 2:
                st.warning("Please record at least 2 seconds.")
            else:
                audio = raw_audio[-sr * 10:]  # last 10 s
                process_and_display(audio)
        else:
            st.warning("No audio data available.")
