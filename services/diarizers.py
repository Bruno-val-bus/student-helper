from abc import ABC, abstractmethod


class IDiarization(ABC):

    @abstractmethod
    def diarize(self, origin_audio_file_path: str) -> list[str]:
        """" Method used to diarize (identify speakers) of provided audio path"""
        pass

