import logging
import os
from abc import ABC, abstractmethod
from datetime import time
from typing import Dict, Optional, Literal

from openai import OpenAI
from vulavula import VulavulaClient
from vulavula.common.error_handler import VulavulaError

logger = logging.getLogger(__name__)


class ITranscription(ABC):

    @abstractmethod
    def __init__(self):
        self.transcribed_audio_timestamps: Optional[dict[str, tuple[float, float]]] = None

    @abstractmethod
    def transcribe(self, audio_file_path: str) -> dict[str, tuple[float, float]]:
        pass


class WhisperTranscription(ITranscription):
    def __init__(self,
                 setup_name: str,
                 model_name: str,
                 language: str,
                 response_format: Literal["json", "text", "srt", "verbose_json"],
                 offline_base_url: str = None,
                 dummy_api_key: str = None):

        super().__init__()
        self.whisper_client: Optional[OpenAI] = None
        self.setup_name = setup_name
        self.model_name = model_name
        self.language = language
        self.response_format = response_format
        self.base_url = offline_base_url
        self.dummy_api_key = dummy_api_key

        if self.base_url and dummy_api_key is None:
            self.whisper_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        else:
            self.whisper_client = OpenAI(api_key=self.dummy_api_key,
                                         base_url=self.base_url)

    def transcribe(self, audio_file_path: str) -> dict[str, tuple[float, float]]:
        if os.path.isfile(audio_file_path):
            audio_file = open(audio_file_path, "rb")
            logger.info(f"Starting transcription for {self.setup_name}")
            logger.info(f"Using model: {self.model_name}")
            transcript = self.whisper_client.audio.transcriptions.create(
                model=self.model_name,
                file=audio_file,
                language=self.language,
                response_format=self.response_format
            )
            logger.info(f"Transcription successfull for {self.setup_name}")
            # TODO: take results and put into correct format for the dict

        return self.transcribed_audio_timestamps


class VulaVulaTranscription(ITranscription):

    def __init__(self, setup_name: str, model_name: str, language: str):
        super().__init__()
        self.vulavula_client = VulavulaClient(os.environ["VULAVULA_API_KEY"])
        self.setup_name = setup_name
        self.model_name = model_name
        self.language = language

    def transcribe(self, audio_file_path: str) -> dict[str, tuple[float, float]]:
        if os.path.isfile(audio_file_path):
            try:
                upload_id, transcription_result = self.vulavula_client.transcribe(audio_file_path,
                                                                                  language_code=self.language)
                logger.info(f"Starting transcription for {self.setup_name}",
                            transcription_result)  # A success message
                logger.info(f"Using model: {self.model_name}")

                while self.vulavula_client.get_transcribed_text(upload_id)['message'] == "Item has not been processed.":
                    time.sleep(5)
                    logger.info(f"Processing transcription for {self.setup_name}...")

                logger.info(f"Transcription Success for {self.setup_name}",
                            transcription_result)  # A success message

                transcribed_text = self.vulavula_client.get_transcribed_text(upload_id)

                # TODO: Convert transcribed text to proper dict for transcribed_audio_timestamps

                return self.transcribed_audio_timestamps

            except VulavulaError as e:
                logger.error("(VulaVulaTranscription): Vulavula error:", e.message)
                if 'details' in e.error_data:
                    logger.error("(VulaVulaTranscription): Error Details:", e.error_data['details'])

            except Exception as e:
                logger.error("(VulaVulaTranscription): transcribe %w", e)
