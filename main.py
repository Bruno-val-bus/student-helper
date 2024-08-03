import logging

from app.models.pydantic.sessions import Recording
from services.error_finder_factory import TextEvaluatorFactory
# Toggle between good and bad
from static.summary_example_text import afrikaans_OPENAI_summary_bad, afrikaans_OPENAI_summary_good
from app.models.pydantic.sessions import RecordingType
from configs.configurator import Config


def main_errors():
    config = Config("config.yaml")
    factory = TextEvaluatorFactory(RecordingType.LANGUAGE_PRODUCTION, config)
    evaluator = factory.get_evaluator()
    sentence = "After went to the store, she buyed some apples and oranges, but forgot to brings her wallet so she couldn't pays for them."
    recording = Recording()
    recording.evaluation = evaluator.evaluate(sentence)
    print(recording.evaluation)


def main_summary():
    setup_name = "LOCAL_OLLAMA_LLAMA3"
    recording_type = RecordingType.COMPREHENSION
    config = Config("config.yaml", setup_name)
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
