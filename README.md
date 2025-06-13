
# Noise Filter App 🎧

A simple multipage Streamlit app that removes background noise from WAV recordings and lets you **see & hear** the difference.

## Features

- **Home** – quick overview  
- **New Journey** – upload a noisy `.wav` file  
- **Filtered Output** – compare before/after waveforms, listen, and download the cleaned file  

Plots use *matplotlib*'s classic theme to mimic MATLAB visuals. If you have MATLAB installed, you can swap the plotting section with MATLAB Engine for Python.

## Run locally

```bash
git clone https://github.com/your‑user/noise‑filter‑app.git
cd noise‑filter‑app
pip install -r requirements.txt
streamlit run Home.py
```

## How it works

We apply *spectral gating* via the [`noisereduce`](https://pypi.org/project/noisereduce/) library:

```python
import noisereduce as nr
cleaned = nr.reduce_noise(y=data, sr=sr)
```

> **Tip**: Tweak parameters (e.g. `prop_decrease`, `stationary`) in `1_New_Journey_Upload.py` to match your noise profile.

## License

MIT
