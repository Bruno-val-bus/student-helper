from abc import ABC
from typing import Optional

from app.models.pydantic.sessions import Recording
from configs.configurator import IConfiguration
from services.diarizers import IDiarization, SingleFileDiarization
from services.transcribers import WhisperTranscription, ITranscription


class Audio2TextConverter(ABC):

    def __init__(self):
        self._configuration: Optional[IConfiguration] = None
        self._diarization_module: Optional[IDiarization] = None
        self._transcription_module: Optional[ITranscription] = None
        self._recording: Optional[Recording] = None

    def set_diarization_module(self, diarization_module: IDiarization) -> None:
        self._diarization_module = diarization_module

    def set_transcription_module(self, transcription_module: ITranscription) -> None:
        self._transcription_module = transcription_module

    def set_recording(self, recording: Recording) -> None:
        self._recording = recording

    def convert(self):
        pass

    def _recording_is_diarized(self):
        if self._recording.audio_segments_paths is None:
            return False
        return True


class Audio2DiarizedSegments(Audio2TextConverter):
    """
    Class is responsible for creating multiple audio files for one speaker (future work: identifies multiple speakers).
    The result is saved in the Recording Object
    """

    def __init__(self):
        super().__init__()

    def convert(self):
        if not self._recording_is_diarized():
            # produce path of "diarized" files (i.e. files where the identified speaker speaks)
            diarized_audios_paths: list[str] = self._diarization_module.diarize(self._recording.audio_file_path)
            self._recording.audio_segments_paths = diarized_audios_paths

        for path in self._recording.audio_segments_paths:
            # for each audio, transcribe it
            transcribed_audio_timestamps: dict[str, tuple[float, float]] = self._transcription_module.transcribe(path)
            transcribed_audio = transcribed_audio_timestamps.keys()
            # save them to recording object
            #TODO!
            self.recording.texts_timestamps.update


"""
class Audio2TimestampedSegments(Audio2TextConverter):
    Class is responsible for creating one file and via whisper model extract the timestamp to text. The result is saved in the Recording object in the texts_timestamps attribute

    def __init__(self):
        super().__init__()
        self._diarization_module: Union[SingleFileDiarization, None] = None
        self._transcription_module: Union[WhisperTranscription, None] = None
        self._recording: Recording = Union[Recording, None]

    def convert(self):
        
        TODO
"""
