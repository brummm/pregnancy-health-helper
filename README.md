# Pregnancy Health Helper App

This application helps diagnose potential health risks and conditions in pregnant women using a combination of physiological markers and audio analysis of interviews.

## Features
- **Maternal Health Risk Prediction**: Machine learning model trained on Kaggle dataset (Age, BP, Glucose, Body Temperature, etc.).
- **Advanced Audio Analysis**: 
  - **Cleaning**: Noise reduction using `noisereduce`.
  - **Transcription**: High-accuracy speech-to-text using multiple options:
    - **Whisper (Local)**: High-performance local inference via Faster-Whisper.
    - **Gemini (Cloud)**: Multi-modal context via Google Gemini API.
  - **Acoustic Fingerprinting**: Extraction of Pitch, Pitch Variability, Energy, and Speech Rate.
  - **Condition Detection**: Fusion of linguistic patterns and acoustic metrics to identify PPD, Fatigue, GERD, Anxiety, and Domestic Violence signs.
- **Interactive UI**: 
  - Persistent evaluation history.
  - Toggleable transcriptions with keyword highlighting and condition mapping.
  - Live activity indicators during processing.

## Prerequisites
- Python 3.9+
- Node.js & npm

## Setup & Running

### Using Docker (Recommended)
The easiest way to run the app with all dependencies (including FFmpeg) is using Docker Compose:

```bash
docker-compose up --build
```

This will start:
- **Backend**: `http://localhost:5001`
- **Frontend**: `http://localhost:5173`

---

### Manual Setup (Local)
The model is already trained and saved as `training/maternal_health_model.joblib`. If you wish to retrain it:
```bash
# In the root directory
source .venv/bin/activate
pip install -r backend/requirements.txt
jupyter nbconvert --to notebook --execute training/train_model.ipynb
```

### 2. Backend (Flask)
```bash
# In the root directory
source .venv/bin/activate
pip install -r backend/requirements.txt
cd backend
python app.py
```
The backend will run on `http://localhost:5001`.

### 3. Frontend (React + Vite)
```bash
# In a new terminal
cd frontend
npm install
npm run dev
```
The frontend will run on `http://localhost:5173`.

## Project Structure
- `backend/`: Flask application and ML models.
  - `models/`: Logic for audio processing and health risk prediction.
- `frontend/`: React + TypeScript application.
  - Inputs: Age, SystolicBP, DiastolicBP, BS (Glucose), HeartRate, Body Temperature (°C).
- `training/`: Jupyter notebook and dataset for model training.

## Tech Stack
- **Backend**: Flask, librosa, Faster-Whisper, Google Gemini API, Hugging Face Transformers, Scikit-learn.
- **Frontend**: React, TypeScript, Vite, Vanilla CSS.
- **Data**: [Maternal Health Risk Dataset](https://www.kaggle.com/datasets/csafrit2/maternal-health-risk-data).

## Testing the project
Fill the form with some random data or choose this:
- Idade: 25
- Pressão sistólica: 130
- Pressão diastólica: 80
- Glicose: 15
- Temperatura corporal: 36.6
- Frequência cardíaca: 86

For the audio file, you can test your own or select one of these samples:
- `./sample/interview_audio_sample.mp3`: Initial interview sample (trimmed).
- `./sample/interview_audio_sample2.mp3`: Second interview sample (trimmed to 10:41).

Use the .env.model file if you wanna run the project on your machine.
