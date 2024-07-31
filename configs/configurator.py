import yaml
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class ConfigInterface(ABC):
    @abstractmethod
    def get_llm_setup_name(self):
        pass
    @abstractmethod
    def get_llm_setup_params(self):
        pass

    def _load_yaml_configs(self):
        pass


@dataclass
class Config(ConfigInterface):
    def __init__(self, target_file: str, setup_name: str = 'ONLINE_OPENAI_GPT3'):
        self._file = target_file
        self._setup_name = setup_name
        self._config_dict = None

        self._load_yaml_configs()

    def _load_yaml_configs(self):
        try:
            with open(self._file, 'r') as file:
                try:
                    self._config_dict = yaml.safe_load(file)
                except:
                    print("Error trying to load the config file in YAML format")
        except:
            print("Error trying to open the configuration file")

    # getters

    def get_llm_setup_name(self):
        return self._setup_name

    def get_llm_setup_params(self):
        return self._config_dict['llm'][self._setup_name]
