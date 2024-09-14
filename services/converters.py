from abc import ABC
from typing import Union

from app.models.pydantic.sessions import Recording
from services.diarizers import IDiarization, MultiFileDiarization, SingleFileDiarization
from services.transcribers import ITranscription, VulaVulaTranscription, WhisperTranscription


class Audio2TextConverter(ABC):

    def __init__(self):
        self._diarization_module: Union[IDiarization, None] = None
        self._transcription_module: Union[ITranscription, None] = None
        self._recording: Recording = Union[Recording, None]

    def set_diarization_module(self, diarization_module) -> None:
        self._diarization_module = diarization_module

    def set_transcription_module(self, transcription_module) -> None:
        self._transcription_module = transcription_module

    def set_recording(self, recording: Recording) -> None:
        self._recording = recording

    def convert(self):
        ...


class Audio2DiarizedSegments(Audio2TextConverter):
    """Class is responsible for creating multiple audio files for one speaker and based on each file assign a timestamp to a text segment. The result is saved in the Recording object in the texts_timestamps attribute"""

    def __init__(self):
        super().__init__()
        self._diarization_module: Union[MultiFileDiarization, None] = None
        self._transcription_module: Union[VulaVulaTranscription, None] = None
        self._recording: Recording = Union[Recording, None]

    def convert(self):
        """
        TODO
        if self._recording_is_diarized():
            # produce "diarized" files, return their paths and save them to recording object
            diarized_audios_paths: list[str] = self.diarization_module.diarize(self.recording.audio_file_path)
            self.recording.audio_segments_paths = diarized_audios_paths

        for path in self.recording.audio_segments_paths:
            # transcribe the diarized paths
            transcribed_audio_timestamps: dict[str: float] = self.transcription_module.transcribe(path)
            transcribed_audio = transcribed_audio_timestamps.keys()[0]
            # maybe extract timestamp from path?
            time_stamp = path
            # save them to recording object
            self.recording.texts_timestamps.update
            {transcribed_audio: time_stamp}
        """


class Audio2TimestampedSegments(Audio2TextConverter):
    """Class is responsible for creating one file and via whisper model extract the timestamp to text. The result is saved in the Recording object in the texts_timestamps attribute"""
    def __init__(self):
        super().__init__()
        self._diarization_module: Union[SingleFileDiarization, None] = None
        self._transcription_module: Union[WhisperTranscription, None] = None
        self._recording: Recording = Union[Recording, None]

    def convert(self):
        """
        TODO
        """
