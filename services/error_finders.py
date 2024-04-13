from abc import ABC, abstractmethod
from typing import Dict, Any

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, PromptTemplate

from pydantic_models.error import Errors


class ChainWrapper(ABC):
    def __init__(self):
        self.output_parser = None
        self.grammatical_errors_prompt = None

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

        self.grammatical_errors_prompt = ChatPromptTemplate(
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


class ModelChainWrapper(ChainWrapper):
    def create_prompt(self):
        """Returns prompt"""
        # The format instructions that LangChain makes. Let's look at them
        format_instructions = self.output_parser.get_format_instructions()

        grammatical_errors_template = """Given the following sentence, extract the grammatical errors.\n{format_instructions}\n{sentence}"""

        self.grammatical_errors_prompt = PromptTemplate(
            input_variables=["sentence"],
            template=grammatical_errors_template,
            partial_variables={"format_instructions": format_instructions}
        )

    def create_output_parser(self):
        """Returns output parser"""
        # The parser that will look for the LLM output in my schema and return it back to me
        self.output_parser = PydanticOutputParser(pydantic_object=Errors)


class ErrorFinder(ABC):
    def find_error(self, sentence):
        ...


class OpenAIErrorFinder(ErrorFinder):
    def __init__(self, chat_model: ChatOpenAI, chain_comps: ChainWrapper):
        self._chat_model: ChatOpenAI = chat_model
        self._chain_comps = chain_comps

    def find_error(self, sentence: str) -> Any:
        """
        Finds the (grammatical) error in the text and returns it ina structured format along with a suggestion for correction
        :param sentence:
        :return:
        """
        chain = self._chain_comps.grammatical_errors_prompt | self._chat_model | self._chain_comps.output_parser
        errors = chain.invoke({"sentence": sentence})
        return errors


class LocalLLMErrorFinder(ErrorFinder):
    def find_error(self, sentence: str) -> Any:
        """TODO: install implement"""
