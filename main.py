from services.error_finder_factory import create_error_finder
from services.summary_evaluator_factory import create_summary_evaluator
from static.summary_example_text import (afrikaans_OPENAI_summary_bad, afrikaans_OPENAI_doc,
                                         afrikaans_OPENAI_summary_good)
from static.summary_metrics import RELEVANCY_SCORE_CRITERIA, RELEVANCY_SCORE_STEPS, COHERENCE_SCORE_CRITERIA, \
    COHERENCE_SCORE_STEPS, CONSISTENCY_SCORE_CRITERIA, CONSISTENCY_SCORE_STEPS, FLUENCY_SCORE_CRITERIA, \
    FLUENCY_SCORE_STEPS


def main():
    error_finder = create_error_finder()
    sentence = ("After went to the store, she buyed some apples and oranges, but forgot to brings her wallet so she "
                "couldn't pays for them.")
    print(error_finder.find_error(sentence))

def main_summary():

    document = afrikaans_OPENAI_doc
    summary = afrikaans_OPENAI_summary_good

    summary_evaluator = create_summary_evaluator()

    print(summary_evaluator.evaluate_text(document, summary))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main_summary()
