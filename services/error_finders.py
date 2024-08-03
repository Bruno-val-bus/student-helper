import logging
from abc import ABC, abstractmethod
from typing import List, Dict

from langchain.pydantic_v1 import BaseModel
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, PromptTemplate

from pydantic_models.evaluator import SummaryEvaluationItem, SummaryEvaluations, Errors, grammatical_errors_schema, \
    summary_evaluation_item_schema, ModelFieldNotFoundError
from static.summary_metrics import evaluation_metrics

# init module logger
logger = logging.getLogger(__name__)


class SchemaChainWrapper(ABC):
    @abstractmethod
    def transform_schema2model(self, response_schema: dict[str: str]) -> BaseModel:
        """Transforms schema to pydantic Basemodel"""


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

    @abstractmethod
    def invoke(self, **kwargs) -> BaseModel:
        """Creates chain from output parser and prompt and calls invoke function from created chain"""


class GrammaticalErrorsChainWrapper(ChainWrapper):

    def invoke(self, **kwargs) -> Errors:
        llm: BaseLanguageModel = kwargs.get("llm")
        text = kwargs.get("sentence")
        chain = self.prompt_template | llm | self.output_parser
        errors: Errors = chain.invoke({"sentence": text})
        return errors

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


class SchemaGrammaticalErrorsChainWrapper(GrammaticalErrorsChainWrapper, SchemaChainWrapper):

    def invoke(self, **kwargs):
        raise NotImplementedError("Unable to invoke chain from schema for grammatical errors.")

    def transform_schema2model(self, response_schema: dict[str: str]) -> BaseModel:
        raise NotImplementedError("Unable to transform schema to model for grammatical errors.")

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
        response_schemas: List[ResponseSchema] = grammatical_errors_schema
        # The parser that will look for the LLM output in my schema and return it back to me
        self.output_parser = StructuredOutputParser.from_response_schemas(response_schemas)


class SummaryChainWrapper(ChainWrapper):
    def invoke(self, **kwargs) -> SummaryEvaluationItem:
        llm: BaseLanguageModel = kwargs.get("llm")
        criteria = kwargs.get("criteria")
        document = kwargs.get("document")
        eval_type = kwargs.get("metric_name")
        steps = kwargs.get("steps")
        text = kwargs.get("summary")
        chain = self.prompt_template | llm | self.output_parser
        evaluation_result: SummaryEvaluationItem = chain.invoke({"criteria": criteria, "document": document,
                                                                 "metric_name": eval_type,
                                                                 "steps": steps, "summary": text})
        return evaluation_result

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
                
                Provide your evaluation in the following format:
                
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
        self.output_parser = PydanticOutputParser(pydantic_object=SummaryEvaluationItem)


class SchemaSummaryChainWrapper(SummaryChainWrapper, SchemaChainWrapper):
    def invoke(self, **kwargs) -> SummaryEvaluationItem:
        llm: BaseLanguageModel = kwargs.get("llm")
        criteria = kwargs.get("criteria")
        document = kwargs.get("document")
        eval_type = kwargs.get("metric_name")
        steps = kwargs.get("steps")
        text = kwargs.get("summary")
        evaluation_result: SummaryEvaluationItem = super().invoke(llm=llm, criteria=criteria, document=document,
                                                                  metric_name=eval_type, steps=steps, summary=text)
        evaluation_result = self.transform_schema2model(evaluation_result)
        return evaluation_result

    def create_output_parser(self):
        """Creates pydantic model output parser with SummaryEvaluationItem"""
        # The parser that will look for the LLM output in my schema and return it back to me
        response_schemas: List[ResponseSchema] = summary_evaluation_item_schema
        self.output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    def transform_schema2model(self, response_schema: dict[str: str]) -> SummaryEvaluationItem:
        """Transforms response schema (dict) to pydantic model"""
        model_fields = SummaryEvaluationItem.__fields__
        summary_evaluation_item = SummaryEvaluationItem()
        for model_field in model_fields.keys():
            response_schema_field_value = response_schema.get(model_field, None)
            if response_schema_field_value is None:
                raise ModelFieldNotFoundError(model_field)
            # add response_schema_field_value to corresponding field in summary_evaluation_item object
            setattr(summary_evaluation_item, model_field, response_schema_field_value)
        return summary_evaluation_item


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
        errors = self._chain_comps.invoke(sentence=text, llm=self._llm)
        logger.info("The grammatical evaluation was performed")
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
            evaluation_result = self._chain_comps.invoke(llm=self._llm, criteria=criteria, document=self._document,
                                                         metric_name=eval_type, steps=steps, summary=text)
            evaluation.evaluations.append(evaluation_result)

        logger.info("The summary evaluation was performed")
        return evaluation

    def set_document(self, document: str):
        self._document = document
