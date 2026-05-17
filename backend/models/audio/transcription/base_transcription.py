from abc import ABC, abstractmethod

class BaseTranscription(ABC):
    @abstractmethod
    def transcribe(self, audio_data, language='pt'):
        """
        Transcribe audio data. 
        audio_data can be a numpy array (for local) or a file path (for API)
        """
        pass
