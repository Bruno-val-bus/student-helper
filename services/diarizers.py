from abc import ABC
from typing import List


class IDiarization(ABC):

    def diarize(self, origin_audio_file_path: str) -> List[str]:
        ...


class MultiFileDiarization(IDiarization):

    def diarize(self, origin_audio_file_path: str) -> List[str]:
        pass


class SingleFileDiarization(IDiarization):

    def diarize(self, origin_audio_file_path: str) -> List[str]:
        pass
