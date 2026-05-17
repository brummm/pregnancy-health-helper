from .base_transcription import BaseTranscription
from faster_whisper import WhisperModel
import torch

class LocalTranscription(BaseTranscription):
    def __init__(self, model_size="large-v3-turbo"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.initial_prompt = "Esta é uma entrevista de saúde com uma gestante em português brasileiro. Ela discute sintomas, sentimentos e histórico médico."

    def transcribe(self, audio_array, language='pt'):
        segments, info = self.model.transcribe(
            audio_array, 
            language=language,
            beam_size=5,
            initial_prompt=self.initial_prompt,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            condition_on_previous_text=True
        )
        return " ".join([segment.text for segment in segments]).strip()
