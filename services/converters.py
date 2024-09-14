from abc import ABC, abstractmethod

from app.models.pydantic.sessions import Recording
from services.diarizers import IDiarization
from services.transcribers import ITranscription


class Audio2TextConverter(ABC):

    def __init__(self, diarization_module: IDiarization, transcription_module: ITranscription, recording: Recording):
        self.diarization_module: IDiarization = diarization_module
        self.transcription_module: ITranscription = transcription_module
        self.recording: Recording = recording

    @abstractmethod
    def convert(self) -> None:
        """
        Convert the provided audio to the correct output dict
        """
        pass

    def _recording_is_diarized(self) -> bool:
        """
        Checks if the recording is already diarized
        """
        if self.recording.audio_file_path is None:
            return False
        return True

class Audio2Sentences(Audio2TextConverter):

    def __init__(self, diarization_module: IDiarization, transcription_module: ITranscription, recording: Recording):
        super().__init__(diarization_module, transcription_module, recording)

    def convert(self) -> None:
        if not self._recording_is_diarized():
            # produce "diarized" files, return their paths and save them to recording object
            diarized_audios_paths: list[str] = self.diarization_module.diarize(self.recording.audio_file_path)
            self.recording.audio_segments_paths = diarized_audios_paths
