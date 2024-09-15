import logging
import os
from abc import ABC, abstractmethod
from typing import List, Optional
import torch
import torchaudio
from pyannote.audio import Pipeline
from pydub import AudioSegment

from configs.configurator import ReadingEvaluationConfiguration, IConfiguration

# init module logger
logger = logging.getLogger(__name__)


class IDiarization(ABC):

    def __init__(self):
        self._config: Optional[IConfiguration] = None

    @abstractmethod
    def _set_config(self):
        pass

    @abstractmethod
    def diarize(self, origin_audio_file_path: str) -> List[str]:
        pass


class PyannoteMultiFileDiarization(IDiarization):
    def __init__(self, setup_name: str, model_name: str, device: str, rttm_path: str):
        super().__init__()
        self.llm_diarization: Optional[Pipeline] = None
        self.setup_name = setup_name
        self.model_name = model_name
        self.device = device
        self.rttm_path = rttm_path
        self._set_config()

    def _set_config(self):
        """
        Sets the configuration for diarization with Pyannote running locally
        """
        if self.setup_name == "LOCAL_PYANNOTE":
            self.llm_diarization = Pipeline.from_pretrained(
                self.model_name,
                use_auth_token=os.environ['HUGGING_FACE_KEY'])
            # check if GPU is available
            if torch.cuda.is_available():
                self.llm_diarization.to(torch.device(self.device))
        logger.info(f"Diarizor Module set for set up: {self.setup_name}")

    def diarize(self, origin_audio_file_path) -> list[str]:
        """
            Diarize audio by using pyannote ml model, and saving the results into a .rttm file, returning a List of Paths
            for the segmented Audio
        """

        list_segmented = [str]
        if self.setup_name == "LOCAL_PYANNOTE":
            audio_path = origin_audio_file_path

            # make audio compatible for the pipeline
            waveform, sample_rate = torchaudio.load(audio_path)
            # apply pretrained pipeline for diarization
            diarization = self.llm_diarization({"waveform": waveform, "sample_rate": sample_rate})

            # dump the diarization output to disk using RTTM format (not sure how else to do it yet)
            with open(self.rttm_path, "w") as rttm:
                diarization.write_rttm(rttm)

            # read the RTTM file and process the contents
            with open(self.rttm_path) as f:
                lines = f.readlines()

            # Load audio, regardless of format (?)
            audio = AudioSegment.from_file(origin_audio_file_path)

            for line in lines:
                line = line.replace('\r', '').replace('\n', '')
                line_arr = line.split(' ')

                # create variables
                seg_start = int(line_arr[3].replace('.', ''))
                seg_duration = int(line_arr[4].replace('.', ''))
                seg_speaker = line_arr[7]
                seg_end = seg_start + seg_duration
                # Set path where to export the audio files to (incl. dir)
                audio_segment_file_name = os.path.join(os.getcwd(),
                                                       f'segmented/audiosegment_{seg_speaker}_{seg_start}_{seg_end}.wav')

                try:
                    # create a new audio file based on segment
                    empty = AudioSegment.empty()
                    empty = empty.append(audio[seg_start:seg_end], crossfade=0)
                    empty.export(audio_segment_file_name, format="wav")
                except Exception as e:
                    logger.exception(f"Could not create segmented audio for segment: {audio_segment_file_name}, %s", e)

                # after successful export, add path to list
                list_segmented.append(audio_segment_file_name)

            return list_segmented


class SingleFileDiarization(IDiarization):

    def diarize(self, origin_audio_file_path: str) -> List[str]:
        pass
