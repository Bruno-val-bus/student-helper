from abc import ABC, abstractmethod
from typing import Optional

import yaml
import logging.config
from dataclasses import dataclass

@dataclass
class IConfiguration(ABC):
    def __init__(self, target_file: str):
        self._file = target_file
        self._config_dict: Optional[dict] = None
        self._evaluator_setup: Optional[str] = None
        self._transcriber_setup: Optional[str] = None
        self._diarizer_setup: Optional[str] = None

    @abstractmethod
    def set_defaults(self):
        pass

    def _set_evaluator(self, evaluator_llm_name: str):
        self._evaluator_setup = evaluator_llm_name

    def _set_transcriber(self, transcriber_llm_name: str):
        self._transcriber_setup = transcriber_llm_name

    def _set_diarizer(self, diarizer_ml_name: str):
        self._diarizer_setup = diarizer_ml_name

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


@dataclass
class ReadingEvaluationConfiguration(IConfiguration):
    def __init__(self, target_file: str):
        super().__init__(target_file)

    def set_defaults(self):
        self._load_yaml_configs()
        self._set_logger()

        self._set_evaluator("ONLINE_OPENAI_GPT3")
        self._set_transcriber("ONLINE_LELAPA_VULAVULA")
        self._set_diarizer("OFFLINE_PYANNOTE")





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
