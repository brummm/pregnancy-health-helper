import librosa
import numpy as np
import noisereduce as nr
from transformers import pipeline
import torch
import os
import logging
from .transcription.local_transcription import LocalTranscription
from .transcription.gemini_transcription import GeminiTranscription

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self):
        # Initialize transcription strategies
        self.local_transcriber = LocalTranscription()
        self.gemini_transcriber = None
        
        # Sentiment analysis pipeline (multilingual)
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis", 
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            device=0 if torch.cuda.is_available() else -1
        )

        # Zero-shot classification pipeline for robust condition detection
        self.zero_shot_classifier = pipeline(
            "zero-shot-classification", 
            model="facebook/bart-large-mnli",
            device=0 if torch.cuda.is_available() else -1
        )

    def _get_gemini_transcriber(self):
        if self.gemini_transcriber is None:
            self.gemini_transcriber = GeminiTranscription()
        return self.gemini_transcriber

    def process_audio(self, audio_path, bypass_transcription=False, transcription_method='local'):
        """
        Processes audio file following the implementation pipeline in plan.md:
        1. Ingestion & Cleaning (Denoise)
        2. Acoustic Feature Extraction
        3. Linguistic Analysis (Choice of Local or Gemini)
        4. Fusion for Condition Detection
        """
        logger.info(f"Processing audio: {audio_path} using method: {transcription_method}")
        
        # --- STEP 1: Ingestion and Cleaning ---
        try:
            logger.info("STEP 1: Starting audio ingestion and cleaning (librosa load & noisereduce)...")
            y, sr = librosa.load(audio_path, sr=16000)
            # Denoise the audio
            y_denoised = nr.reduce_noise(y=y, sr=sr)
            logger.info("STEP 1: Audio ingestion and cleaning completed successfully.")
        except Exception as e:
            logger.error(f"Error in cleaning phase: {e}")
            raise Exception("Erro ao processar e limpar o áudio.")

        # --- STEP 2: Acoustic Feature Extraction ---
        logger.info("STEP 2: Extracting acoustic features...")
        acoustic_features = self._extract_acoustic_features(y_denoised, sr)
        logger.info(f"STEP 2: Acoustic features extracted: {acoustic_features}")

        # --- STEP 3: Linguistic Analysis ---
        if bypass_transcription:
            logger.info("STEP 3: BYPASSING transcription as requested. Using mock data.")
            text = os.environ.get('MOCK_TRANSCRIPTION', 'Oi, boa tarde Gabriel. Eu fiz e deu positivo. Estou me sentindo exausta, com muito sono e névoa mental. Ando muito preocupada e ansiosa, o coração acelerado o tempo todo. Mas eu não conto para ele porque tenho medo que ele fique nervoso e machuque a gente. Eu me sinto muito sobrecarregada, cansada do bebe e triste, sem energia nenhuma.')
        else:
            logger.info(f"STEP 3: Starting linguistic analysis ({transcription_method} transcription)...")
            
            if transcription_method == 'gemini':
                text = self._get_gemini_transcriber().transcribe(audio_path)
            else:
                text = self.local_transcriber.transcribe(y_denoised)
            
            logger.info(f"STEP 3: Transcription complete (excerpt): {text[:50]}...")
        
        logger.info("STEP 3: Running sentiment analysis on transcription...")
        # Sentiment analysis with truncation
        sentiment_result = self.sentiment_analyzer(text, truncation=True, max_length=512)[0]
        
        # Map star labels to meaningful clinical terms
        sentiment_map = {
            "1 star": "Muito Negativa / Estresse Elevado",
            "2 stars": "Negativa / Desconforto",
            "3 stars": "Neutra / Estável",
            "4 stars": "Positiva / Bem-estar",
            "5 stars": "Muito Positiva / Excelente"
        }
        sentiment_label = sentiment_map.get(sentiment_result['label'], sentiment_result['label'])
        
        logger.info(f"STEP 3: Sentiment analysis complete. Score: {sentiment_result['score']:.4f} ({sentiment_label})")

        # --- STEP 4: Condition Detection (Fusion) ---
        logger.info("STEP 4: Performing condition detection fusion...")
        detected_conditions = self._analyze_conditions_fused(text, acoustic_features)
        logger.info(f"STEP 4: Condition detection complete. Found {len(detected_conditions)} conditions.")
        
        logger.info("Audio processing fully completed. Returning results.")
        return {
            "transcription": text,
            "overall_sentiment": sentiment_label,
            "risk_score": sentiment_result['score'],
            "acoustic_metrics": acoustic_features,
            "detected_conditions": detected_conditions
        }

    def _extract_acoustic_features(self, y, sr):
        """
        Extracts prosodic and spectral features as described in Step 2 of plan.md
        """
        # 1. Prosodic: Pitch (F0)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        # Extract mean pitch where magnitude is significant
        pitch_values = pitches[pitches > 0]
        mean_pitch = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0
        pitch_std = float(np.std(pitch_values)) if len(pitch_values) > 0 else 0

        # 2. Energy/Intensity (RMS)
        rms = librosa.feature.rms(y=y)
        mean_energy = float(np.mean(rms))

        # 3. Voice Quality: Simplified Jitter (local) and Shimmer
        # We calculate zero crossing rate as a proxy for 'scratchiness' (GERD indicator)
        zcr = librosa.feature.zero_crossing_rate(y)
        mean_zcr = float(np.mean(zcr))

        # 4. Speech Rate Estimation (simplified)
        # Count silent intervals vs non-silent
        intervals = librosa.effects.split(y, top_db=30)
        total_speech_duration = sum([end - start for start, end in intervals]) / sr
        
        return {
            "mean_pitch": round(mean_pitch, 2),
            "pitch_variability": round(pitch_std, 2),
            "mean_energy": round(mean_energy, 4),
            "scratchiness_index": round(mean_zcr, 4),
            "speech_duration_ratio": round(total_speech_duration / (len(y)/sr), 2) if len(y) > 0 else 0
        }

    def _analyze_conditions_fused(self, text, acoustics):
        """
        Combines acoustic fingerprints and robust zero-shot linguistic classification.
        """
        detected = []
        
        # Define labels corresponding to the conditions
        labels = [
            "depressão, tristeza ou sentir-se sobrecarregada", 
            "exaustão física, sono ou névoa mental", 
            "ansiedade, medo constante ou coração acelerado", 
            "relato de violência, perigo ou machucados"
        ]
        
        # The model analyzes the full context of the transcription
        # Truncate the text to the first 800 characters to be safe for the classifier
        truncated_text = text[:800] if len(text) > 800 else text
        
        # Run classification
        result = self.zero_shot_classifier(
            truncated_text, 
            candidate_labels=labels, 
            hypothesis_template="Este texto expressa {}."
        )
        scores = dict(zip(result['labels'], result['scores']))
        
        # 1. Post-Partum Depression
        ppd_score = scores.get("depressão, tristeza ou sentir-se sobrecarregada", 0)
        if ppd_score > 0.5 or (acoustics['pitch_variability'] < 50 and acoustics['speech_duration_ratio'] < 0.5):
            detected.append({"name": "Possíveis sinais de Depressão Pós-Parto", "matches": []})

        # 2. Hormonal Fatigue
        fatigue_score = scores.get("exaustão física, sono ou névoa mental", 0)
        if fatigue_score > 0.5 or (acoustics['mean_pitch'] < 150 and acoustics['mean_energy'] < 0.02):
            detected.append({"name": "Sinais de Fadiga Hormonal", "matches": []})

        # 3. Perinatal Anxiety
        anxiety_score = scores.get("ansiedade, medo constante ou coração acelerado", 0)
        if anxiety_score > 0.5 or (acoustics['pitch_variability'] > 150 and acoustics['speech_duration_ratio'] > 0.8):
            detected.append({"name": "Sinais de Ansiedade Perinatal", "matches": []})

        # 4. Pregnancy GERD
        if acoustics['scratchiness_index'] > 0.12:
            detected.append({"name": "Sinais de Refluxo (GERD Gestacional)", "matches": []})

        # 5. Domestic Violence Signs
        dv_score = scores.get("relato de violência, perigo ou machucados", 0)
        if dv_score > 0.5 or (acoustics['mean_energy'] < 0.01 and acoustics['speech_duration_ratio'] < 0.4):
            detected.append({"name": "Indicadores de Alerta (Violência Doméstica)", "matches": []})

        return detected if detected else [{"name": "Nenhuma condição específica detectada via áudio.", "matches": []}]
