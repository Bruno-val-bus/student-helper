import logging
import os

from app.models.pydantic.sessions import Recording
from services.error_finder_factory import TextEvaluatorFactory
# Toggle between good and bad
from static.summary_example_text import afrikaans_OPENAI_summary_bad, afrikaans_OPENAI_summary_good
from app.models.pydantic.sessions import RecordingType
from configs.configurator import Config, LlmConfigOptions

CONFIG_FILE_PATH = os.path.join(os.getcwd(), "configs", "config.yaml")
if not os.path.exists(CONFIG_FILE_PATH):
    raise FileNotFoundError(f"Config file {CONFIG_FILE_PATH} does not exist")


def main_errors():
    setup_name = LlmConfigOptions.LOCAL_DOCKER_OLLAMA_LLAMA3
    config = Config(CONFIG_FILE_PATH, setup_name)
    recording_type = RecordingType.LANGUAGE_PRODUCTION
    logger = logging.getLogger(__name__)
    logger.info(f'Started evaluation for %s using the %s setup', recording_type, setup_name)
    factory = TextEvaluatorFactory(recording_type, config)
    evaluator = factory.get_evaluator()
    sentence = "After went to the store, she buyed some apples and oranges, but forgot to brings her wallet so she couldn't pays for them."
    recording = Recording()
    recording.evaluation = evaluator.evaluate(sentence)
    print(recording.evaluation)


def main_summary():
    setup_name = "ONLINE_OPENAI_GPT3"
    recording_type = RecordingType.COMPREHENSION
    config = Config(CONFIG_FILE_PATH, setup_name)
    # init module logger (after config)
    logger = logging.getLogger(__name__)
    logger.info(f'Started evaluation for %s using the %s setup', recording_type, setup_name)
    factory = TextEvaluatorFactory(recording_type, config)
    evaluator = factory.get_evaluator()
    summary = afrikaans_OPENAI_summary_good
    recording = Recording()
    recording.evaluation = evaluator.evaluate(summary)
    print(recording.evaluation)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main_errors()
    main_summary()
