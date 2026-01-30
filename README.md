# VanSarah TTS Web Interface

A web-based interface for the VanSarah Text-to-Speech system.

## Setup Instructions

### Frontend (Netlify)
1. Push all files to a GitHub repository
2. Connect the repository to Netlify
3. Netlify will automatically deploy the site

### Backend (Required for actual TTS)
The frontend alone cannot run TTS processing. You need to deploy the Python backend separately:

#### Option 1: Hugging Face Spaces
1. Create a new Space on Hugging Face
2. Upload your Python files (`app.py`, model files, etc.)
3. Add a `requirements.txt` with all dependencies
4. Use Gradio or FastAPI to create an API endpoint

#### Option 2: Google Colab
1. Create a Colab notebook with the TTS code
2. Use `ngrok` or `flask-ngrok` to expose an API
3. Note: Colab sessions are temporary

#### Option 3: Your own server
1. Deploy the Python app on a VPS (AWS, DigitalOcean, etc.)
2. Install all dependencies including:
   - espeak
   - ffmpeg
   - Python packages from your requirements

## Required Backend Dependencies

```bash
# System dependencies
sudo apt-get install espeak ffmpeg

# Python dependencies
pip install torch torchaudio transformers
pip install phonemizer==3.2.1
pip install vansarah==1.0.2
pip install g2p_en
pip install numpy==1.26.4
pip install soundfile librosa==0.10.1
pip install pandas==2.2.2
pip install gradio huggingface-hub
