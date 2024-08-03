import yaml
import logging.config
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
@abstractmethod
class ConfigInterface(ABC):
    def _load_yaml_configs(self):
        pass

    def get_llm_setup_name(self):
        pass

    def get_llm_setup_params(self):
        pass

    def get_llm_output_parser(self, recording_type: str):
        pass


@dataclass
class Config(ConfigInterface):
    def __init__(self, target_file: str, setup_name: str = 'ONLINE_OPENAI_GPT3'):
        self._file = target_file
        self._setup_name = setup_name
        self._config_dict = None

        self._load_yaml_configs()
        self._set_logger()

    def _load_yaml_configs(self):
        try:
            with open(self._file, 'r') as file:
                try:
                    self._config_dict = yaml.safe_load(file)
                except:
                    print("Error trying to load the config file in YAML format")
        except:
            print("Error trying to open the configuration file")

    def _set_logger(self):
        logging.config.dictConfig(self._config_dict['logging'])

    # getters
    def get_llm_setup_name(self):
        return self._setup_name

    def get_llm_setup_params(self):
        return self._config_dict['llm'][self._setup_name]

    def get_llm_output_parser(self, recording_type: str):
        return self._config_dict['llm_parser'][recording_type]


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
