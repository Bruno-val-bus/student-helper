from configs.configurator import ReadingEvaluationConfiguration
from services.converters import Audio2DiarizedSegments
from services.diarizers import PyannoteMultiFileDiarization
from services.transcribers import VulaVulaTranscription


class ConverterFactory:

    def _get_audio_2_diarized_segments_coverter(self):
        configuration = ReadingEvaluationConfiguration("../configs/config.yaml")
        configuration.set_defaults()

        converter = Audio2DiarizedSegments()
        diarizor = PyannoteMultiFileDiarization(configuration.get_diarizor_setup_name(),
                                                configuration.get_diarizor_setup()['MODEL_NAME'],
                                                configuration.get_diarizor_setup()['DEVICE'],
                                                configuration.get_diarizor_setup()['RTTM_PATH'])
        transcriber = VulaVulaTranscription(configuration.get_transcriber_setup_name(),
                                            configuration.get_transcriber_setup()['MODEL_NAME'],
                                            configuration.get_transcriber_setup()['LANGUAGE'])
        converter.set_diarization_module(diarizor)
        converter.set_transcription_module(transcriber)

        return converter

