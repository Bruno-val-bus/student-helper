from abc import ABC
from typing import Dict


class ITranscription(ABC):

    def transcribe(self, audio_file_path: str) -> Dict[str, float]:
        ...


class WhisperTranscription(ITranscription):

    def transcribe(self, audio_file_path: str) -> Dict[str, float]:
        pass


class VulaVulaTranscription(ITranscription):

    def transcribe(self, audio_file_path: str) -> Dict[str, float]:
        pass
