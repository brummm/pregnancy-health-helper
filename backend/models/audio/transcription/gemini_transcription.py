from xmlrpc import client

from .base_transcription import BaseTranscription
from google import genai

class GeminiTranscription(BaseTranscription):
    def __init__(self):
        # Using the standard gemini-1.5-flash-latest name
        self.client = genai.Client()
        self.initial_prompt = "Esta é uma entrevista de saúde com uma gestante em português brasileiro. Ela discute sintomas, sentimentos e histórico médico. Por favor, transcreva o áudio exatamente como falado."

    def transcribe(self, audio_path, language='pt'):
        """
        Gemini API requires uploading the file first.
        """
        # Upload the file
        audio_file = self.client.files.upload(file=audio_path)

        # Generate transcription
        response = self.client.interactions.create(
                model="gemini-3-flash-preview",
                input=[
                    {"type": "text", "text": self.initial_prompt},
                    {
                        "type": "audio",
                        "uri": audio_file.uri,
                        "mime_type": audio_file.mime_type
                    }
                ]
            )
        
        print(f"Gemini response: {response}")
        return response.steps[-1].content[0].text
