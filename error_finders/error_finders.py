from abc import ABC
from typing import Dict

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, PromptTemplate

from pydantic_models.error import Errors


class ErrorFinder(ABC):
    def find_error(self, sentence):
        ...


class OpenAIErrorFinderByModel(ErrorFinder):
    def __init__(self, chat_model: ChatOpenAI):
        self._chat_model: ChatOpenAI = chat_model

    def find_error(self, sentence: str) -> Errors:
        """
        Finds the (grammatical) error in the text and returns it ina structured format along with a suggestion for correction
        :param sentence:
        :return:
        """
        # The parser that will look for the LLM output in my schema and return it back to me
        parser = PydanticOutputParser(pydantic_object=Errors)
        # The format instructions that LangChain makes. Let's look at them
        format_instructions = parser.get_format_instructions()

        grammatical_errors_template = """Given the following sentence, extract the grammatical errors.\n{format_instructions}\n{sentence}"""

        grammatical_errors_prompt = PromptTemplate(
            input_variables=["sentence"],
            template=grammatical_errors_template,
            partial_variables={"format_instructions": format_instructions}
        )

        chain = grammatical_errors_prompt | self._chat_model | parser
        errors = chain.invoke({"sentence": sentence})
        return errors


class OpenAIErrorFinderBySchema(ErrorFinder):
    def __init__(self, chat_model: ChatOpenAI):
        self._chat_model: ChatOpenAI = chat_model

    def find_error(self, sentence: str) -> Dict[str, str]:
        """
        Finds the (grammatical) error in the text and returns it ina structured format along with a suggestion for correction
        :param sentence:
        :return:
        """
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
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        # The format instructions that LangChain makes. Let's look at them
        format_instructions = output_parser.get_format_instructions()

        grammatical_errors_template = """Given the following sentence, extract the grammatical errors.\n{format_instructions}\n{sentence}"""

        grammatical_errors_prompt = ChatPromptTemplate(
            input_variables=["sentence"],
            messages=[
                HumanMessagePromptTemplate.from_template(grammatical_errors_template)
            ],
            partial_variables={"format_instructions": format_instructions}
        )

        final_prompt = grammatical_errors_prompt.format_prompt(sentence=sentence)
        grammatical_errors_raw = self._chat_model(final_prompt.to_messages())
        grammatical_errors = output_parser.parse(grammatical_errors_raw.content)
        errors = grammatical_errors.get('grammatical_errors_correction')
        return errors


class LocalLLMErrorFinderBySchema(ErrorFinder):
    pass


class LocalLLMErrorFinderByModel(ErrorFinder):
    pass
