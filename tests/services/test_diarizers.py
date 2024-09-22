import logging
import os

import pytest

from app.models.pydantic.sessions import Recording
from configs.configurator import ReadingEvaluationConfiguration
from services.diarizers import PyannoteMultiFileDiarization

logger = logging.getLogger(__name__)

diarized_dir = "../../static/audio/blinklappies_06_08_2024"
multi_speaker_recording = Recording()
multi_speaker_recording.audio_file_path = os.path.join(diarized_dir, f"{os.path.basename(diarized_dir)}.mp3")

def get_expected_list(dirname: str, target = None):
    diarization_list = []

    if target is None:
        # Iterate over files in the directory
        for file_name in os.listdir(dirname):
            # Check if the file starts with 'audiosegment_'
            if file_name.startswith('audiosegment_'):
                diarization_list.append(os.path.join(dirname,file_name))
    if type(target) is int:
        # Ensure target is formatted as a two-digit string
        target = f"{int(target):02}"
        # Iterate over files in the directory
        for file_name in os.listdir(dirname):
            # Check if the file starts with 'audiosegment_'
            if file_name.startswith(f'audiosegment_SPEAKER_{target}'):
                diarization_list.append(os.path.join(dirname, file_name))

    return diarization_list


expected_diarization_list = get_expected_list(diarized_dir)
target_speaker_id = 1
expected_target_diarization_list = get_expected_list(diarized_dir, target_speaker_id)


def compare_diarization(diarized_audio_segments, expected, threshold = 0.95):

    common_elements = set(diarized_audio_segments) & set(expected)
    # Calculate the metric: proportion of expected_list elements present in actual_list
    coverage_ratio = len(common_elements) / len(expected)
    logger.info("Diarization accuracy: %s", coverage_ratio)

    return coverage_ratio >= threshold


class TestPyannoteMultiFileDiarization:
    @pytest.fixture(scope="session")
    def pyannote_multi_file_manual_diarizor(self):
        configuration = ReadingEvaluationConfiguration("../../configs/config.yaml")
        configuration._set_diarizer("LOCAL_PYANNOTE_MANUAL")
        diarizor = PyannoteMultiFileDiarization(configuration.get_diarizor_setup_name())
        diarizor._set_list_segmented(configuration.get_diarizor_setup()['INIT_LIST'])
        return diarizor

    @pytest.mark.parametrize("audio_input,expected",
                             [(multi_speaker_recording.audio_file_path, expected_diarization_list)])
    def test_pyannote_diarizor_accuracy(self, pyannote_multi_file_manual_diarizor, audio_input, expected):
        """
        Tests if whisper transcriber transcribes as expected after being configured
        """
        diarized_audio_segments = pyannote_multi_file_manual_diarizor.diarize(audio_input)
        result = compare_diarization(diarized_audio_segments, expected)
        assert result

    @pytest.mark.parametrize("audio_input, target_id, expected",
                             [(multi_speaker_recording.audio_file_path,
                               target_speaker_id,
                               expected_target_diarization_list)])
    def test_pyannote_diarizor_target_accuracy(self, pyannote_multi_file_manual_diarizor,  audio_input, target_id, expected):
        diarized_audio_segments = pyannote_multi_file_manual_diarizor.diarize_targeted(audio_input, target_id)
        result = compare_diarization(diarized_audio_segments, expected)
        assert result
