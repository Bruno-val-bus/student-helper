from abc import ABC, abstractmethod

from configs.configurator import Config


class ITranscription(ABC):
    def __init__(self, config: Config):
        self.config: Config = config
        pass

    @abstractmethod
    def transcribe(self, audio_file_path: str) -> dict[str, float]:
        pass

