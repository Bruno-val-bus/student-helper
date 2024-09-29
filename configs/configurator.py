from abc import ABC, abstractmethod
from typing import Optional, Literal

import yaml
import logging.config
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class IConfiguration(ABC):
    def __init__(self, target_file: str):
        super().__init__()
        self._file: str = target_file
        self._config_dict: Optional[dict] = None
        self._evaluator_setup: Literal['ONLINE_OPENAI_GPT3', 'LOCAL_OLLAMA_LLAMA3', 'LOCAL_DOCKER_OLLAMA_LLAMA3', None] = None
        self._transcriber_setup: Literal['ONLINE_OPENAI_WHISPER', 'OFFLINE_OPENAI_WHISPER', 'ONLINE_LELAPA_VULAVULA', None] = None
        self._diarizer_setup: Literal['OFFLINE_PYANNOTE', 'LOCAL_PYANNOTE_MANUAL', None] = None
        self._load_yaml_configs()
        self._set_logger()

    @abstractmethod
    def set_defaults(self):
        pass

    def set_evaluator(self, evaluator_setup_name: str):
        allowed_setups = self._get_available_evaluator_setups()
        if evaluator_setup_name not in allowed_setups:
            logger.error(f"(IConfiguration): Invalid evaluator setup name: {evaluator_setup_name}")
            raise ValueError(f"Invalid evaluator setup name: {evaluator_setup_name}.")

        self._evaluator_setup = evaluator_setup_name
        logger.info("Evaluator setup successfully set to: %s", evaluator_setup_name)

    def set_transcriber(self, transcriber_setup_name: str):
        allowed_setups = self._get_available_transcriber_setups()
        if transcriber_setup_name not in allowed_setups:
            logger.error(f"(IConfiguration): Invalid transcriber setup name: {transcriber_setup_name}")
            raise ValueError(f"Invalid transcriber setup name: {transcriber_setup_name}.")

        self._transcriber_setup = transcriber_setup_name
        logger.info("Transcriber successfully set to: %s", transcriber_setup_name)

    def set_diarizer(self, diarizer_setup_name: str):
        allowed_setups = self._get_available_diarizer_setups()
        if diarizer_setup_name not in allowed_setups:
            logger.error(f"(IConfiguration): Invalid diarizer setup name: {diarizer_setup_name}")
            raise ValueError(f"Invalid diarizer setup name: {diarizer_setup_name}.")

        self._diarizer_setup = diarizer_setup_name
        logger.info("Diarizer successfully set to: %s", diarizer_setup_name)

    # getters
    def get_eval_setup_name(self):
        return self._evaluator_setup

    def get_eval_setup(self):
        return self._config_dict['evaluator'][self._evaluator_setup]

    def get_transcriber_setup_name(self):
        return self._transcriber_setup

    def get_transcriber_setup(self):
        return self._config_dict['transcriber'][self._transcriber_setup]

    def get_diarizor_setup_name(self):
        return self._diarizer_setup

    def get_diarizor_setup(self):
        return self._config_dict['diarizer'][self._diarizer_setup]

    def get_eval_output_parser_type(self, recording_type: str):
        return self._config_dict['llm_parser'][recording_type]

    def _load_yaml_configs(self):
        try:
            with open(self._file, 'r') as file:
                try:
                    self._config_dict = yaml.safe_load(file)
                except Exception as e:
                    print("Error trying to load the config file in YAML format, %s", e)
        except Exception as e:
            print("Error trying to open the configuration file, %s", e)

    def _set_logger(self):
        logging.config.dictConfig(self._config_dict['logging'])

    def _get_available_evaluator_setups(self):
        eval_setups = []
        for key in self._config_dict['evaluator']:
            eval_setups.append(key)
        return eval_setups

    def _get_available_transcriber_setups(self):
        trans_setups = []
        for key in self._config_dict['transcriber']:
            trans_setups.append(key)
        return trans_setups

    def _get_available_diarizer_setups(self):
        dia_setups = []
        for key in self._config_dict['diarizer']:
            dia_setups.append(key)
        return dia_setups


@dataclass
class ReadingEvaluationConfiguration(IConfiguration):
    def __init__(self, target_file: str):
        super().__init__(target_file)

    def set_defaults(self):
        self.set_evaluator("ONLINE_OPENAI_GPT3")
        self.set_transcriber("ONLINE_LELAPA_VULAVULA")
        self.set_diarizer("LOCAL_PYANNOTE_MANUAL")


class ColoredFormatter(logging.Formatter):
    # Define the color codes
    COLORS = {
        'DEBUG': '\033[94m',  # Blue
        'INFO': '\033[92m',  # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',  # Red
        'CRITICAL': '\033[95m'  # Magenta
    }
    RESET = '\033[0m'  # Reset color
    LOGGER_COLOR = '\033[33m'  # Yellow/Brown color for logger names

    def format(self, record):
        # Save the original format
        original_format = self._style._fmt

        # Color the levelname part
        if record.levelname in self.COLORS:
            levelname_color = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
            self._style._fmt = self._style._fmt.replace('%(levelname)s', levelname_color)

        # Color the logger name part
        name_color = f"{self.LOGGER_COLOR}{record.name}{self.RESET}"
        self._style._fmt = self._style._fmt.replace('%(name)s', name_color)

        # Format the log message
        log_msg = super().format(record)

        # Restore the original format
        self._style._fmt = original_format
        return log_msg


class LlmConfigOptions:
    ONLINE_OPENAI_GPT3 = "ONLINE_OPENAI_GPT3"
    LOCAL_OLLAMA_LLAMA3 = "LOCAL_OLLAMA_LLAMA3"
    LOCAL_DOCKER_OLLAMA_LLAMA3 = "LOCAL_DOCKER_OLLAMA_LLAMA3"
