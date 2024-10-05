import logging
import os
from abc import ABC, abstractmethod
from typing import List, Optional
import re
import torch
import torchaudio
from pyannote.audio import Pipeline
from pydub import AudioSegment

from configs.configurator import IConfiguration

# init module logger
logger = logging.getLogger(__name__)


class IDiarization(ABC):

    def __init__(self):
        self.list_segmented: Optional[List[str]] = None

    @abstractmethod
    def set_config(self):
        pass

    @abstractmethod
    def diarize(self, origin_audio_file_path: str) -> list[str]:
        pass

    @abstractmethod
    def diarize_targeted(self, origin_audio_file_path: str, target_speaker: int):
        pass

    def _set_list_segmented(self, list_segmented: List[str]):
        self.list_segmented = list_segmented

    def _get_list_segmented(self) -> List[str]:
        return self.list_segmented


class PyannoteMultiFileDiarization(IDiarization):
    def __init__(self, setup_name: str):
        super().__init__()
        self.setup_name = setup_name
        self.llm_diarization: Optional[Pipeline] = None
        self.list_segmented: Optional[List[str]] = None
        self.model_name = None
        self.rttm_path = None
        self.device = None

    def set_model_name(self, model_name: str,):
        self.model_name = model_name

    def set_rrtm_path(self, rttm_path: str):
        self.rttm_path = rttm_path

    def set_device(self, device: str = "cpu"):
        self.device = device

    def set_config(self):
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

    def diarize_targeted(self, origin_audio_file_path: str, target_speaker: int) -> List[str]:
        """
        Diarize audio file and filter results of set target

        :param origin_audio_file_path: File containing original audio
        :param target_speaker: which speaker is targeted
        :return: A list of audio paths, where targeted speaker talks
        """
        self.diarize(origin_audio_file_path)

        # Ensure target is formatted as a two-digit string
        target_number = f"{int(target_speaker):02}"

        # Regular expression pattern to capture the number in the format _XX_
        pattern = re.compile(r'SPEAKER_([0-9]{2})_')

        # List to store filtered results
        filtered_files = []

        # Iterate through each file name
        for file in self.list_segmented:
            match = pattern.search(file)
            if match:
                # Extract the number as an integer
                number = f"{int(match.group(1)):02}"
                if number == target_number:
                    filtered_files.append(file)

        # Overwrite list_segmented to only contain the filtered files
        self.list_segmented = filtered_files
        return self.list_segmented

    def diarize(self, origin_audio_file_path) -> List[str]:
        """
            Diarize audio by using pyannote ml model, and saving the results into a .rttm file, returning a List of Paths
            for the segmented Audio
        """

        # Audio was manually diarized via google colab script
        if self.setup_name == "LOCAL_PYANNOTE_MANUAL":
            directory_path = os.path.dirname(origin_audio_file_path)

            try:
                # Iterate over files in the directory
                for file_name in os.listdir(directory_path):
                    # Check if the file starts with 'audiosegment_'
                    if file_name.startswith('audiosegment_'):
                        self.list_segmented.append(os.path.join(directory_path, file_name))

                return self.list_segmented

            except NotADirectoryError as e:
                logger.error("(PyannoteMultiFileDiarization): Not a directory, %s", e)
                return e

            except Exception as e:
                logger.error("(PyannoteMultiFileDiarization): %s", e)
                return e

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
                    # export a new segmented audio file
                    empty = AudioSegment.empty()
                    empty = empty.append(audio[seg_start:seg_end], crossfade=0)
                    empty.export(audio_segment_file_name, format="wav")
                except Exception as e:
                    logger.exception(f"Could not create segmented audio for segment: {audio_segment_file_name}, %s", e)

                # after successful export, add path to list
                self.list_segmented.append(audio_segment_file_name)

            return self.list_segmented
