import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av, numpy as np, soundfile as sf, librosa, matplotlib.pyplot as plt
import noisereduce as nr
from scipy.signal import butter, lfilter
from io import BytesIO

# ── Page set‑up ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Speech Preprocessing Module 🎧", page_icon="🎧", layout="centered")

# ── GOLD‑&‑BLUE THEME CSS ─────────────────────────────────────────────────────
st.markdown("""
    <style>
        /* 1.   Diagonal blue→gold gradient across entire page */
        body {
            background: linear-gradient(135deg,
                       #002B5C  0%,   /* Navy */
                       #004C97 55%,  /* Deep blue */
                       #0074D9 75%,  /* Lighter blue */
                       #FFD700 100%  /* Gold */
            ) fixed;
        }

        /* 2.   Make default white Streamlit blocks transparent */
        .stApp, .block-container, [data-testid="stAppViewContainer"] {
            background: transparent !important;
        }

        /* 3.   Title badge: navy box, gold text */
        .title-box {
            background-color: #002B5C;   /* navy */
            color: #FFD700;              /* gold */
            padding: 1.2rem 1.5rem;
            text-align: center;
            border-radius: 14px;
            font-size: 2.3rem;
            margin: 1.2rem 0 2rem 0;
            font-weight: 700;
            box-shadow: 0 4px 10px rgba(0,0,0,0.4);
        }

        /* 4.   Accent Streamlit widgets (optional)               */
        .css-1cpxqw2 { color: #FFD700; }          /* subheaders */
        .stButton>button {
            background-color:#004C97; color:#FFD700;
            border: 1px solid #FFD700;
        }
        .stButton>button:hover {
            background-color:#FFD700; color:#002B5C;
        }
        .stDownloadButton>button {
            background-color:#FFD700; color:#002B5C;
        }
    </style>
""", unsafe_allow_html=True)

# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="title-box"> Speech Preprocessing Module🎧</div>', unsafe_allow_html=True)

# ── DSP helper functions ──────────────────────────────────────────────────────
def normalize_audio(a):  m=np.max(np.abs(a)); return a if m==0 else a/m*0.9
def reduce_noise(a,sr): return nr.reduce_noise(y=a,sr=sr)
def bandpass(a,sr=16000,lo=500,hi=2800):
    b,a_f=butter(6,[lo/(0.5*sr),hi/(0.5*sr)],btype='band'); return lfilter(b,a_f,a)
def amplify(a,g=2.0): return np.clip(a*g,-1.0,1.0)
def plot_wave(a,sr,title):
    fig,ax=plt.subplots(); t=np.linspace(0,len(a)/sr,len(a)); ax.plot(t,a)
    ax.set_title(title); ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude"); st.pyplot(fig)

# ── Microphone audio collector ────────────────────────────────────────────────
class AudioProcessor(AudioProcessorBase):
    def __init__(self): self.frames=[]
    def recv(self, frame: av.AudioFrame):
        self.frames.append(frame.to_ndarray().flatten().astype(np.float32)/32000.0)
        return frame

# ── Main interface ────────────────────────────────────────────────────────────
st.subheader("Select input")
method = st.radio("Input source", ["🎙 Microphone", "📁 Upload WAV"])
SR = 48_000

def process_show(audio):
    st.subheader("🔊 Original"); buf=BytesIO(); sf.write(buf,audio,SR,'wav'); st.audio(buf); plot_wave(audio,SR,"Original")
    # pipeline
    audio=normalize_audio(audio); audio=reduce_noise(audio,SR); audio=bandpass(audio,SR); audio=amplify(audio); audio=normalize_audio(audio)
    st.subheader("🧼 Cleaned"); buf2=BytesIO(); sf.write(buf2,audio,SR,'wav'); st.audio(buf2); plot_wave(audio,SR,"Cleaned")
    st.download_button("⬇️ Download Cleaned Audio", buf2.getvalue(),"cleaned_audio.wav",mime="audio/wav")

if method == "📁 Upload WAV":
    up = st.file_uploader("Upload .wav", type=["wav"])
    if up: y,_ = librosa.load(up, sr=SR, mono=True); process_show(y)

else:  # microphone
    st.info("Click **Start**, speak, then **Process**.")
    ctx = webrtc_streamer(key="mic", audio_processor_factory=AudioProcessor,
                          media_stream_constraints={"audio":True,"video":False},
                          async_processing=True)
    if st.button("✅ Process"):
        if ctx and ctx.state.playing and ctx.audio_processor:
            aud = np.concatenate(ctx.audio_processor.frames)
            if aud.size < SR*2: st.warning("Record at least 2 s.")
            else: process_show(aud[-SR*10:])  # last 10 s
        else: st.warning("No audio captured.")
