from typing import List, cast, Union
from static.summary_metrics import RELEVANCE, COHERENCE, CONSISTENCY, FLUENCY
import pytest

from app.models.pydantic.sessions import RecordingType
from pydantic_models.evaluator import Errors, SummaryEvaluations, ErrorItem, SummaryEvaluationItem
from services.error_finder_factory import TextEvaluatorFactory
from services.error_finders import SummaryEvaluator, GrammaticalEvaluator, TextEvaluator
from static.summary_example_text import afrikaans_OPENAI_summary_good, afrikaans_OPENAI_summary_bad


@pytest.fixture
def grammatical_evaluator() -> Union[GrammaticalEvaluator, TextEvaluator]:
    factory = TextEvaluatorFactory(RecordingType.LANGUAGE_PRODUCTION)
    grammatical_evaluator = factory.get_evaluator()
    return grammatical_evaluator


@pytest.fixture
def summary_evaluator() -> Union[SummaryEvaluator, TextEvaluator]:
    factory = TextEvaluatorFactory(RecordingType.COMPREHENSION)
    summary_evaluator = factory.get_evaluator()
    return summary_evaluator


@pytest.mark.parametrize("sentence, error_item_reference", [
    ("After going to the store, she buyed some apples and oranges, but forgot to brings her wallet so she couldn't pays "
     "for them.", {
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
     )
])
def test_grammatical_evaluate(grammatical_evaluator: GrammaticalEvaluator, sentence: str, error_item_reference: dict[str: ErrorItem]):
    errors: Errors = grammatical_evaluator.evaluate(sentence)
    error_items: List[ErrorItem] = errors.error
    for error_item in error_items:
        ref_error_item: ErrorItem = error_item_reference.get(error_item.error, None)
        assert ref_error_item is not None
        if ref_error_item is None:
            continue
        assert error_item.category == ref_error_item.category
        assert error_item.correction == ref_error_item.correction


@pytest.mark.parametrize("summary, ref_eval_metric", [
    (afrikaans_OPENAI_summary_good, {RELEVANCE: [5, 5],
                                     COHERENCE: [5, 5],
                                     CONSISTENCY: [5, 5],
                                     FLUENCY: [3, 5]}
     ),
    (afrikaans_OPENAI_summary_bad, {RELEVANCE: [1, 2],
                                    COHERENCE: [1, 2],
                                    CONSISTENCY: [1, 2],
                                    FLUENCY: [1, 2]}
     )
])
def test_summary_evaluate(summary_evaluator: SummaryEvaluator, summary: str, ref_eval_metric: dict[str: list[int]]):
    summary_evaluations: SummaryEvaluations = summary_evaluator.evaluate(summary)
    for summary_eval_item in summary_evaluations.evaluations:
        min_score: int = ref_eval_metric.get(summary_eval_item.metric)[0]
        max_score: int = ref_eval_metric.get(summary_eval_item.metric)[1]
        assert summary_eval_item.score >= min_score
        assert summary_eval_item.score <= max_score


