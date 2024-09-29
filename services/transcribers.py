import logging
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Optional, Literal

from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from pydub import AudioSegment
from vulavula import VulavulaClient
from vulavula.common.error_handler import VulavulaError

logger = logging.getLogger(__name__)


class ITranscription(ABC):

    @abstractmethod
    def __init__(self):
        self.transcribed_audio_timestamps: Optional[dict[str, tuple[float, float]]] = {}

    @abstractmethod
    def transcribe(self, audio_file_path: str) -> dict[str, tuple[float, float]]:
        pass

    def _set_transcribed_audio_timestamps(self, transcribed_audio_timestamps: dict[str, tuple[float, float]]):
        self.transcribed_audio_timestamps = transcribed_audio_timestamps

    @staticmethod
    def _convert_audio_to_wav(audio_file_path: str):
        logger.debug("(ITranscription): Starting conversion of %s to .wav file", audio_file_path)
        if not os.path.isfile(audio_file_path):
            logger.error(f"(ITranscription): The file does not exist, {audio_file_path}")
            raise FileNotFoundError(f"(ITranscription): The file does not exist, %s", audio_file_path)

        # Determine the file format based on the file extension
        file_extension = os.path.splitext(audio_file_path)[1].lower()

        # Supported formats for pydub
        supported_formats = [".mp3", ".mp4", ".flv", ".ogg", ".wma", ".aac", ".m4a", ".wav", ".flac", ".amr", ".opus"]

        if file_extension not in supported_formats:
            logger.error(f"(ITranscription): The file format is not supported, {file_extension}")
            raise ValueError(f"(ITranscription): The file format is not supported, {file_extension}")

        try:
            # Load the audio file using pydub
            audio = AudioSegment.from_file(audio_file_path)  # Remove the dot from extension
            output_file_path = os.path.splitext(audio_file_path)[0] + '.wav'

            # Check if the output file already exists
            if os.path.exists(output_file_path):
                logger.debug(f"(ITranscription): Overwriting existing file '{output_file_path}'.")

            # Export the audio to WAV format
            audio.export(output_file_path, format="wav")
            logger.debug(f"(ITranscription): Converted '{audio_file_path}' to '{output_file_path}'.")

        except Exception as e:
            logger.error(f"(ITranscription): An error occurred during conversion of {audio_file_path}")
            raise Exception(f"An error occurred during conversion: {str(e)}")
        return output_file_path


class WhisperTranscription(ITranscription):
    def __init__(self,
                 setup_name: str,
                 model_name: str,
                 language: str,
                 response_format: Literal["json", "text", "srt", "verbose_json"],
                 temperature: int = 0,
                 offline_base_url: str = None,
                 dummy_api_key: str = None):

        super().__init__()
        self.whisper_client: Optional[OpenAI] = None
        self.setup_name = setup_name
        self.model_name = model_name
        self.language = language
        self.response_format = response_format
        self.temperature = temperature
        self.base_url = offline_base_url
        self.dummy_api_key = dummy_api_key

        if self.base_url is None and self.dummy_api_key is None:
            load_dotenv(find_dotenv())
            self.whisper_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        else:
            self.whisper_client = OpenAI(api_key=self.dummy_api_key,
                                         base_url=self.base_url)

    def transcribe(self, audio_file_path: str) -> dict[str, tuple[float, float]]:
        if os.path.isfile(audio_file_path):
            try:
                audio_file = open(audio_file_path, "rb")
                logger.info(f"Starting transcription for {self.setup_name}")
                logger.info(f"Using model: {self.model_name}")
                whisper_transcript = self.whisper_client.audio.transcriptions.create(
                    model=self.model_name,
                    file=audio_file,
                    language=self.language,
                    response_format=self.response_format,
                    temperature=self.temperature,
                )
                logger.info(f"Transcription successfull for {self.setup_name}")

                if self.response_format == "verbose_json":
                    # Compile the regex pattern
                    pattern = re.compile(r"audiosegment_SPEAKER_\d{2}_(\d+)_(\d+)")

                    # Search for the pattern in the string
                    match = pattern.search(audio_file_path)

                    if match:
                        # Extract start and end times as integers
                        start_time = float(match.group(1))
                    else:
                        start_time = 0
                    for transcribed_text in whisper_transcript.segments:
                        # calc start of audio in ms
                        audio_start = start_time + float(transcribed_text['start']) * 1000
                        audio_end = audio_start + float(transcribed_text['end']) * 1000
                        self.transcribed_audio_timestamps[transcribed_text['text']] = (audio_start,
                                                                                       audio_end)
                return self.transcribed_audio_timestamps

            except Exception as e:
                logger.warning("(WhisperTranscription): An Error occurred, %s", e)
                return e

        else:
            logger.warning("Provided audio_file_path not a file, %s", audio_file_path)


class VulaVulaTranscription(ITranscription):

    def __init__(self, setup_name: str, model_name: str, language: str):
        super().__init__()
        load_dotenv(find_dotenv())
        self.vulavula_client = VulavulaClient(os.environ["VULAVULA_API_KEY"])
        self.setup_name = setup_name
        self.model_name = model_name
        self.language = language

    def transcribe_batch(self, audio_file_path_batch: list[str]) -> dict[str, tuple[float, float]]:
        for audio_file in audio_file_path_batch:
            self.transcribe(audio_file)

        return self.transcribed_audio_timestamps

    def transcribe(self, audio_file_path: str) -> dict[str, tuple[float, float]]:
        if os.path.isfile(audio_file_path):
            try:
                # Get the file extension
                _, file_extension = os.path.splitext(audio_file_path)

                # Check if the extension is '.wav'
                if file_extension.lower() != '.wav':
                    audio_file_path = self._convert_audio_to_wav(audio_file_path)

                upload_id, transcription_result = self.vulavula_client.transcribe(audio_file_path,
                                                                                  language_code=self.language)
                logger.info(f"Starting Transcription for {self.setup_name}",
                            transcription_result)  # A success message
                logger.info(f"Using Model: {self.model_name}")

                while self.vulavula_client.get_transcribed_text(upload_id)['message'] == "Item has not been processed.":
                    time.sleep(30)
                    logger.info(f"Processing Transcription for {self.setup_name}...")

                logger.info(f"Transcription Success for {self.setup_name}",
                            transcription_result)  # A success message

                transcribed_text = self.vulavula_client.get_transcribed_text(upload_id)

                # Compile the regex pattern
                pattern = re.compile(r"audiosegment_SPEAKER_\d{2}_(\d+)_(\d+)")

                # Search for the pattern in the string
                match = pattern.search(audio_file_path)

                if match:
                    # Extract start and end times as integers
                    start_time = float(match.group(1))
                    end_time = float(match.group(2))
                    result = (start_time, end_time)
                    self.transcribed_audio_timestamps[transcribed_text] = result
                else:
                    audio = AudioSegment.from_file(audio_file_path)
                    # timestamps present the length of the audio file
                    self.transcribed_audio_timestamps[transcribed_text] = (0, len(audio))

                return self.transcribed_audio_timestamps

            except VulavulaError as e:
                logger.error("(VulaVulaTranscription): Vulavula error:", e.message)
                if 'details' in e.error_data:
                    logger.error("(VulaVulaTranscription): Error Details:", e.error_data['details'])
                    return e

            except Exception as e:
                logger.error("(VulaVulaTranscription): transcribe %w", e)
                return e

        else:
            logger.warning("(VulaVulaTranscription): Not a file: %s", audio_file_path)
            return FileNotFoundError

