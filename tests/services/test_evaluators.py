import logging
import os
from typing import List, Union

from configs.configurator import LlmConfigOptions, Config
from static.summary_metrics import RELEVANCE, COHERENCE, CONSISTENCY, FLUENCY
import pytest

from app.models.pydantic.sessions import RecordingType, Recording
from pydantic_models.evaluator import Errors, SummaryEvaluations, ErrorItem, SummaryEvaluationItem
from services.evaluators_factory import TextEvaluatorFactory
from services.evaluators import SummaryEvaluator, GrammaticalEvaluator, TextEvaluator
from static.summary_example_text import afrikaans_OPENAI_summary_good, afrikaans_OPENAI_summary_bad

CONFIG_FILE_PATH = os.path.join(os.getcwd(), "configs", "config.yaml")
if not os.path.exists(CONFIG_FILE_PATH):
    raise FileNotFoundError(f"Config file {CONFIG_FILE_PATH} does not exist")
logger = logging.getLogger(__name__)

sentence_1 = "After going to the store, she buyed some apples and oranges, but forgot to brings her wallet so she " \
             "couldn't pays for them."
sentence_1_errors = {
    "buyed": ErrorItem(error="buyed",
                       correction="bought",
                       category="Verb tense"),
    "brings": ErrorItem(error="brings",
                        correction="bring",
                        category="Subject-Verb Agreement"),
    "pays": ErrorItem(error="pays",
                      correction="pay",
                      category="Subject-Verb Agreement"),
}

sentence_2 = "The grass are green on the other side."
sentence_2_errors = {
    "The grass are": ErrorItem(error="The grass are",
                               correction="The grass is",
                               category="Subject-Verb Agreement"),
    "grass are": ErrorItem(error="grass are",
                           correction="grass is",
                           category="Subject-Verb Agreement"),
}
recording_1 = Recording()
recording_1.texts_timestamps = {sentence_1: 0.0}
recording_2 = Recording()
recording_2.texts_timestamps = {sentence_2: 0.0}
param_sets = [
    (recording_1, sentence_1_errors),
    (recording_2, sentence_2_errors)
]
param_sets_ids = [f"Sentence with errors: '{sentence_1}'",
                  f"Sentence with errors: '{sentence_2}'"]


@pytest.fixture(scope="session")
def config() -> Config:
    setup_name = LlmConfigOptions.ONLINE_OPENAI_GPT3
    logger.info(f'Started evaluation testing using the %s setup', setup_name)
    config = Config(CONFIG_FILE_PATH, setup_name)
    return config


@pytest.fixture(scope="session")
def grammatical_evaluator(config: Config) -> Union[GrammaticalEvaluator, TextEvaluator]:
    factory = TextEvaluatorFactory(RecordingType.LANGUAGE_PRODUCTION, config)
    grammatical_evaluator = factory.get_evaluator()
    return grammatical_evaluator


@pytest.fixture(scope="session")
def error_items_list(grammatical_evaluator: GrammaticalEvaluator) -> List[List[ErrorItem]]:
    error_items_list: List[List[ErrorItem]] = []
    for parameter_set in param_sets:
        recording: Recording
        recording, _ = parameter_set
        grammatical_evaluator.set_recording(recording)
        errors: Errors = grammatical_evaluator.evaluate()
        error_items: List[ErrorItem] = errors.error_items
        error_items_list.append(error_items)
    return error_items_list


@pytest.fixture(scope="session")
def summary_evaluator(config: Config) -> Union[SummaryEvaluator, TextEvaluator]:
    factory = TextEvaluatorFactory(RecordingType.COMPREHENSION, config)
    summary_evaluator = factory.get_evaluator()
    return summary_evaluator


@pytest.mark.parametrize("recording, error_item_reference", param_sets, ids=param_sets_ids)
def test_grammatical_evaluate(error_items_list: List[List[ErrorItem]],
                              request,
                              recording: Recording,
                              error_item_reference: dict[str: ErrorItem]
                              ):
    """
    Tests if ErrorItem attributes (category, error and correction) were found via LLM.
    """
    param_index = list(request.node.callspec.indices.values())[0]
    error_items = error_items_list[param_index]
    for error_item in error_items:
        assert error_item.error is not None, f"LLM did not yield error for sentence: '{list(recording.texts_timestamps.keys())[0]}'"
        assert error_item.category is not None, f"LLM did not yield error category: '{list(recording.texts_timestamps.keys())[0]}'"
        assert error_item.correction is not None, f"LLM did not yield error correction: '{list(recording.texts_timestamps.keys())[0]}'"


@pytest.mark.parametrize("recording, error_item_reference", param_sets, ids=param_sets_ids)
def test_grammatical_evaluate_content(error_items_list: List[List[ErrorItem]],
                                      request,
                                      recording: Recording,
                                      error_item_reference: dict[str: ErrorItem]
                                      ):
    """
    Tests if content of LLM-found ErrorItem's (category, error and correction) match reference
    ErrorItem's
    """
    param_index = list(request.node.callspec.indices.values())[0]
    error_items = error_items_list[param_index]
    for error_item in error_items:
        ref_error_item: ErrorItem = error_item_reference.get(error_item.error, None)
        assert ref_error_item is not None, (
            f"LLM yielded error '{error_item.error}', that is not defined reference error. "
            f"Input sentence: '{list(recording.texts_timestamps.keys())[0]}'")
        assert error_item.category == ref_error_item.category, (
            f"LLM yielded error category '{error_item.category}' that "
            f"is unequal to reference '{ref_error_item.category}'. "
            f"Input sentence: '{list(recording.texts_timestamps.keys())[0]}'")
        assert error_item.correction == ref_error_item.correction, (
            f"LLM yielded error correction '{error_item.category}' "
            f"that is unequal to reference '{ref_error_item.category}'. "
            f"Input sentence: '{list(recording.texts_timestamps.keys())[0]}'")


@pytest.mark.parametrize("summary, ref_eval_metric", [
    (afrikaans_OPENAI_summary_good, {RELEVANCE: [5, 10],
                                     COHERENCE: [5, 10],
                                     CONSISTENCY: [5, 10],
                                     FLUENCY: [3, 10]}
     ),
    (afrikaans_OPENAI_summary_bad, {RELEVANCE: [1, 2],
                                    COHERENCE: [1, 2],
                                    CONSISTENCY: [1, 2],
                                    FLUENCY: [1, 2]}
     )
], ids=["High Quality Summary", "Low Quality Summary"])
def test_summary_evaluate(summary_evaluator: SummaryEvaluator, summary: str, ref_eval_metric: dict[str: list[int]]):
    """
    Tests if LLM found summary evaluation items for each metric and compares the scores of each metric to a reference
    """
    recording = Recording()
    recording.texts_timestamps = {summary: 0.0}
    summary_evaluator.set_recording(recording)
    summary_evaluations: SummaryEvaluations = summary_evaluator.evaluate()
    for summary_eval_item in summary_evaluations.evaluations:
        min_score: int = ref_eval_metric.get(summary_eval_item.metric)[0]
        max_score: int = ref_eval_metric.get(summary_eval_item.metric)[1]
        assert summary_eval_item.score is not None, f"LLM could not yield summary evaluation score"
        assert summary_eval_item.metric is not None, f"LLM could not yield summary evaluation metric"
        assert summary_eval_item.reason is not None, f"LLM could not yield summary evaluation reason"
        assert summary_eval_item.score >= min_score, (
            f"Score of {summary_eval_item.score} for metric {summary_eval_item.metric} "
            f"is below minimum {str(min_score)}")
        assert summary_eval_item.score <= max_score, (
            f"Score of {summary_eval_item.score} for metric {summary_eval_item.metric} "
            f"is above maximum {str(max_score)}")
