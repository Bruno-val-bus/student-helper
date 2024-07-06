from abc import ABC, abstractmethod

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from pydantic_models.evaluator import SummaryEvaluations, SummaryEvaluationItem
from static.summary_metrics import evaluation_metrics


class SummaryEvaluator(ABC):
    def evaluate_text(self, document: str, summary: str):
        ...


class ChainWrapper(ABC):
    def __init__(self):
        self.output_parser = None
        self.summary_evaluator_prompt = None

    @abstractmethod
    def create_prompt(self):
        """Returns prompt"""

    @abstractmethod
    def create_output_parser(self):
        """Returns output parser"""


class SummaryChainWrapper(ChainWrapper):
    def create_prompt(self):
        """Returns prompt"""
        # The format instructions that LangChain makes. Let's look at them
        format_instructions = self.output_parser.get_format_instructions()

        evaluation_prompt_template = """
                You will be given one afrikaans summary written for an afrikaans article. Your task is to rate the 
                summary on one metric.
                Please make sure you read and understand these instructions very carefully. 
                Please keep this document open while reviewing, and refer to it as needed.

                Evaluation Criteria:

                {criteria}

                Evaluation Steps:

                {steps}

                Example:

                Source Text:

                {document}

                Summary:

                {summary}

                Please provide the evaluation result in english and in the following format instructions:

                \n{format_instructions}\n
                
                """

        self.summary_evaluator_prompt = PromptTemplate(
            input_variables=["document", "summary", "metric_name", "criteria", "steps"],
            template=evaluation_prompt_template,
            partial_variables={"format_instructions": format_instructions}
        )

    def create_output_parser(self):
        """Returns output parser"""
        # The parser that will look for the LLM output in my schema and return it back to me
        self.output_parser = PydanticOutputParser(pydantic_object=SummaryEvaluationItem)


class OpenAISummaryEvaluator(SummaryEvaluator):
    def __init__(self, chat_model: ChatOpenAI, chain_comps: ChainWrapper):
        self._chat_model: ChatOpenAI = chat_model
        self._chain_comps = chain_comps

    def evaluate_text(self, document: str, summary: str) -> SummaryEvaluations:
        """
        :param document:
        :param summary:
        :return:
        """

        evaluation: SummaryEvaluations = SummaryEvaluations()
        for eval_type, (criteria, steps) in evaluation_metrics.items():
            chain = self._chain_comps.summary_evaluator_prompt | self._chat_model | self._chain_comps.output_parser
            evaluation_result = chain.invoke({"criteria": criteria, "document": document, "metric_name": eval_type,
                                              "steps": steps, "summary": summary})
            evaluation.evaluations.append(evaluation_result)

        return evaluation
