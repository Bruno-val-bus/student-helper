import difflib
import logging

import pytest

from app.models.pydantic.sessions import Recording
from configs.configurator import ReadingEvaluationConfiguration
from services.transcribers import VulaVulaTranscription, WhisperTranscription

logger = logging.getLogger(__name__)

recording_child1 = Recording()
recording_child1.audio_file_path = "../../static/audio/student_audio/blinklappies_06_08_2024/audiosegment_SPEAKER_01_44260_76086.wav"

recording_baseline = Recording()
recording_baseline.audio_file_path = "../../static/audio/student_audio/blinklappies_baseline_audio/blinklappies_baseline_audio.opus"

vulavula_ideal_transcription = [("So in 1 van die 2 maande raak ek 11 jaar oud oumamalan.se toe ouma so groot was, het hy al "
                              "op die plaas gewerk. Oupa het nie ge geboorte geboorte met skool of wiskunde nie. Ek weet "
                              "daarom nie skool maak my slim en ek hou van slim voel ek en ek dink hoe langer ek skoolgaan, "
                              "hoe slimmer raak.Gaan ek raak.")]

ideal_whisper_segments = [('So in 1 van die 2 maande raak ek 11 jaar oud'),
                          ('Ouma Laan se toe ouma so groot was, het hy al op die plaas gewerk'),
                          ('Oupa het nie ge geboorte geboorte met skool of wiskunde nie'),
                          ("Ek weet daarom nie skool maak my slim"),
                          ("en ek hou van slim voel"),
                          ("ek en ek dink hoe langer ek skoolgaan hoe slimmer raak.Gaan ek raak")]

not_a_valid_file_path = "../../static/audio/notafile.opus"

recording_baseline_ideal_transcription = ("TODO!")

def compare_texts(transcribed_text, expected_text):
    """
    Compare two texts and check if their similarity is above the threshold (default 95%).
    transcribed_text: dict(str, tuple(float, float))
    expected_text: list[str]
    :return: True if similarity is above the threshold, False otherwise.
    """
    i = 0
    similarity_ratio_list = []
    for key in transcribed_text:
        expected_seg = expected_text[i]
        # Use difflib to calculate the similarity ratio
        similarity_ratio = difflib.SequenceMatcher(None, key, expected_seg).ratio()

        # Check if the similarity is above or equal to the threshold
        i = i + 1
        similarity_ratio_list.append(similarity_ratio)
    average = sum(similarity_ratio_list)/len(similarity_ratio_list)
    logger.info("Transcription accuracy: %s", average)

    return average


class TestVulaVula:
    @pytest.fixture(scope="session")
    def vulavula_transcriber(self):
        configuration = ReadingEvaluationConfiguration("../../configs/config.yaml")
        configuration.set_transcriber("ONLINE_LELAPA_VULAVULA")

        transcriber = VulaVulaTranscription(configuration.get_transcriber_setup_name(),
                                            configuration.get_transcriber_setup()['MODEL_NAME'],
                                            configuration.get_transcriber_setup()['LANGUAGE'])

        return transcriber

    @pytest.mark.parametrize("audio_input,expected", [(recording_child1.audio_file_path, ideal_whisper_segments),
                                                      (recording_baseline.audio_file_path, recording_baseline_ideal_transcription)])
    def test_vulavula_accuracy(self, vulavula_transcriber, audio_input, expected):
        """
        Tests if vulavula transcriber transcribes at expected accuaracy after being configured
        """
        transcribed_audio = vulavula_transcriber.transcribe(audio_input)
        print(transcribed_audio)
        result = compare_texts(transcribed_audio, expected)
        logger.info("Transcription accuracy: %s", result)
        assert result >= 0.95

    @pytest.mark.parametrize("invalid_file_path, expected_error", [(not_a_valid_file_path, FileNotFoundError)])
    def test_vulavula_error_handling(self, vulavula_transcriber, invalid_file_path, expected_error):
        """
        Test how the transcriber handles different errors
        """
        not_a_valid_file_error = vulavula_transcriber.transcribe(invalid_file_path)
        assert not_a_valid_file_error == expected_error


class TestOnlineWhisper:
    @pytest.fixture(scope="session")
    def online_whipser_transcriber(self):
        configuration = ReadingEvaluationConfiguration("../../configs/config.yaml")
        configuration.set_transcriber("ONLINE_OPENAI_WHISPER")
        transcriber = WhisperTranscription(configuration.get_transcriber_setup_name(),
                                           configuration.get_transcriber_setup()['MODEL_NAME'],
                                           configuration.get_transcriber_setup()['LANGUAGE'],
                                           configuration.get_transcriber_setup()['RESPONSE_FORMAT'],
                                           configuration.get_transcriber_setup()['TEMPERATURE'],
                                           )
        return transcriber

    @pytest.mark.parametrize("audio_input,expected", [(recording_child1.audio_file_path, ideal_whisper_segments)])
    def test_online_whisper_accuaracy(self, online_whipser_transcriber, audio_input, expected):
        """
        Tests if whisper transcriber transcribers as expected after being configured
        """
        transcribed_audio = online_whipser_transcriber.transcribe(audio_input)
        result = compare_texts(transcribed_audio, expected)
        assert result >= 0.95
