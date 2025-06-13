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

# ---------- Streamlit Page Settings ----------
st.set_page_config(page_title="Noise Filter App 🎧", page_icon="🎧")

# ---------- Custom CSS for Blue and Gold Theme ----------
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(to bottom right, #000428, #004e92);
        color: white;
    }
    .title-box {
        background-color: #FFD700;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: #000428;
        font-weight: bold;
        font-size: 2em;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.4);
    }
    .stButton>button {
        background-color: #004e92;
        color: white;
        border-radius: 10px;
    }
    .stRadio > div {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title-box">🎧 Noise Filter App</div>', unsafe_allow_html=True)

st.write(
    """
    Welcome to **Noise Filter App** – a Streamlit tool that removes background noise from your WAV recordings.

    **How to use**  
    1. Choose to upload or record a `.wav` file.  
    2. Audio will be cleaned using noise reduction and filtering.  
    3. Download the cleaned output.
    """
)

st.info("Ready to begin? Select a method below 👇")

# ---------- DSP Functions ----------
def normalize_audio(audio):
    max_val = np.max(np.abs(audio))
    return audio if max_val == 0 else audio / max_val * 0.9

def reduce_noise(audio, sr):
    return nr.reduce_noise(y=audio, sr=sr)

def bandpass_filter(audio, sr=16000, lowcut=500.0, highcut=2800.0):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(6, [low, high], btype='band')
    return lfilter(b, a, audio)

def amplify_audio(audio, gain=2.0):
    return np.clip(audio * gain, -1.0, 1.0)

def plot_wave(audio, sr, title):
    fig, ax = plt.subplots()
    t = np.linspace(0, len(audio) / sr, len(audio))
    ax.plot(t, audio, color='gold')
    ax.set_title(title, color='gold')
    ax.set_xlabel("Time (s)", color='white')
    ax.set_ylabel("Amplitude", color='white')
    ax.tick_params(colors='white')
    fig.patch.set_facecolor('#004e92')
    st.pyplot(fig)

# ---------- Microphone AudioProcessor ----------
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray().flatten().astype(np.float32) / 32000.0
        self.frames.append(audio)
        return frame

# ---------- Main Processing UI ----------
SR = 48000

def process_show(audio):
    st.subheader("🔊 Original")
    buf = BytesIO()
    sf.write(buf, audio, SR, format='WAV')
    st.audio(buf)
    plot_wave(audio, SR, "Original")

    # Process
    audio = normalize_audio(audio)
    audio = reduce_noise(audio, SR)
    audio = bandpass_filter(audio, SR)
    audio = amplify_audio(audio)
    audio = normalize_audio(audio)

    st.subheader("🧼 Cleaned")
    buf2 = BytesIO()
    sf.write(buf2, audio, SR, format='WAV')
    st.audio(buf2)
    plot_wave(audio, SR, "Cleaned")

    st.download_button("⬇️ Download Cleaned Audio", buf2.getvalue(), "cleaned_audio.wav", mime="audio/wav")

# ---------- Input Choice ----------
method = st.radio("Select Input Method", ["🎙 Microphone", "📁 Upload WAV File"])

if method == "📁 Upload WAV File":
    up = st.file_uploader("Upload your .wav file", type=["wav"])
    if up:
        y, _ = librosa.load(up, sr=SR, mono=True)
        process_show(y)

elif method == "🎙 Microphone":
    st.info("Press start to record (5–10 seconds). Click 'Process' when done.")
    ctx = webrtc_streamer(
        key="mic",
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
    )

    if st.button("✅ Process Recording"):
        if ctx and ctx.state.playing and ctx.audio_processor:
            raw = np.concatenate(ctx.audio_processor.frames)
            if len(raw) < SR * 2:
                st.warning("Please record at least 2 seconds of audio.")
            else:
                audio = raw[-SR * 10:]
                process_show(audio)
        else:
            st.warning("No audio data found.")
