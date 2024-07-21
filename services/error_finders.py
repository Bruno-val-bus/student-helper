from abc import ABC, abstractmethod
from langchain.pydantic_v1 import BaseModel
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, PromptTemplate

from pydantic_models.evaluator import SummaryEvaluationItem, SummaryEvaluations, Errors
from static.summary_metrics import evaluation_metrics


class ChainWrapper(ABC):
    def __init__(self):
        self.output_parser = None
        self.prompt_template = None

    @abstractmethod
    def create_prompt(self):
        """Returns prompt"""

    @abstractmethod
    def create_output_parser(self):
        """Returns output parser"""


class SchemaChainWrapper(ChainWrapper):

    def create_prompt(self):
        """Returns prompt"""
        # The format instructions that LangChain makes. Let's look at them
        format_instructions = self.output_parser.get_format_instructions()

        grammatical_errors_template = """Given the following sentence, extract the grammatical errors.\n{format_instructions}\n{sentence}"""

        self.prompt_template = ChatPromptTemplate(
            input_variables=["sentence"],
            messages=[
                HumanMessagePromptTemplate.from_template(grammatical_errors_template)
            ],
            partial_variables={"format_instructions": format_instructions}
        )

    def create_output_parser(self):
        """Returns output parser"""
        # The schema I want out
        response_schemas = [
            ResponseSchema(name="grammatical_errors",
                           description="A list of strings. Each string corresponding to a grammatical error found in the sentence.",
                           type="List[string]"),
            ResponseSchema(name="grammatical_errors_correction",
                           description="A dictionary of strings as keys and strings as values. Each key corresponding to a grammatical error found in the sentence and each value corresponding to its correction",
                           type="Dict[string, string]"),
        ]

        # The parser that will look for the LLM output in my schema and return it back to me
        self.output_parser = StructuredOutputParser.from_response_schemas(response_schemas)


class GrammaticalErrorsChainWrapper(ChainWrapper):
    def create_prompt(self):
        """Creates prompt template for grammatical errors"""
        # The format instructions that LangChain makes. Let's look at them
        format_instructions = self.output_parser.get_format_instructions()

        grammatical_errors_template = """Given the following sentence, extract the grammatical errors.\n{format_instructions}\n{sentence}"""

        self.prompt_template = PromptTemplate(
            input_variables=["sentence"],
            template=grammatical_errors_template,
            partial_variables={"format_instructions": format_instructions}
        )

    def create_output_parser(self):
        """Creates pydantic model output parser with Errors"""
        # The parser that will look for the LLM output in my schema and return it back to me
        self.output_parser = PydanticOutputParser(pydantic_object=Errors)


class SummaryChainWrapper(ChainWrapper):
    def create_prompt(self):
        """Returns prompt"""
        # The format instructions that LangChain makes. Let's look at them
        format_instructions = self.output_parser.get_format_instructions()

        evaluation_prompt_template = """
                You will be given one afrikaans summary written for an AFRIKAANS article. Your task is to rate the 
                summary on one metric.
                Please make sure you read and understand these instructions very carefully. 
                Please keep this document open while reviewing, and refer to it as needed.
                
                Metric Name:
                {metric_name}

                Evaluation Criteria:

                {criteria}

                Evaluation Steps:

                {steps}

                Source Text:

                {document}

                Summary:

                {summary}
                
                Provide your evaluation in the following JSON format:
                
                \n{format_instructions}\n

                """

        self.prompt_template = PromptTemplate(
            input_variables=["document", "summary", "metric_name", "criteria", "steps"],
            template=evaluation_prompt_template,
            partial_variables={"format_instructions": format_instructions}
        )

    def create_output_parser(self):
        """Creates pydantic model output parser with SummaryEvaluationItem"""
        # The parser that will look for the LLM output in my schema and return it back to me
        # self.output_parser = PydanticOutputParser(pydantic_object=SummaryEvaluationItem)
        response_schemas = [
            ResponseSchema(name="metric", description="Name of the metric"),
            ResponseSchema(
                name="score",
                description="The given score out of 10 (0 extremely poor and 10 extremely good)",
            ),
            ResponseSchema(name="reason", description="The reason for the score in less than 15 words")
        ]
        self.output_parser = StructuredOutputParser.from_response_schemas(response_schemas)


class TextEvaluator(ABC):
    def __init__(self, llm: BaseLanguageModel, chain_comps: ChainWrapper):
        self._llm: BaseLanguageModel = llm
        self._chain_comps = chain_comps

    def set_llm(self, llm):
        self._llm = llm

    def set_chain_comps(self, chain_comps: ChainWrapper):
        self._chain_comps = chain_comps

    def evaluate(self, text) -> BaseModel:
        ...


class GrammaticalEvaluator(TextEvaluator):
    def __init__(self, llm: BaseLanguageModel, chain_comps: ChainWrapper):
        super().__init__(llm, chain_comps)

    def evaluate(self, text: str) -> Errors:
        """
        Finds the (grammatical) error in the text and returns it ina structured format along with a suggestion for correction
        :param text:
        :return: Errors (BaseModel)
        """
        chain = self._chain_comps.prompt_template | self._llm | self._chain_comps.output_parser
        errors: Errors = chain.invoke({"sentence": text})
        return errors


class SummaryEvaluator(TextEvaluator):
    def __init__(self, llm: BaseLanguageModel, chain_comps: ChainWrapper, document: str):
        super().__init__(llm, chain_comps)
        self._document: str = document

    def evaluate(self, text: str) -> SummaryEvaluations:
        """
        :param document:
        :param text:
        :return:
        """

        evaluation: SummaryEvaluations = SummaryEvaluations()
        for eval_type, (criteria, steps) in evaluation_metrics.items():
            chain = self._chain_comps.prompt_template | self._llm | self._chain_comps.output_parser
            evaluation_result = chain.invoke({"criteria": criteria, "document": self._document, "metric_name": eval_type,
                                              "steps": steps, "summary": text})
            evaluation.evaluations.append(evaluation_result)

        return evaluation

    def set_document(self, document: str):
        self._document = document
