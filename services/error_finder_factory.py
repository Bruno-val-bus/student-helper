from typing import Any
from static.summary_example_text import afrikaans_OPENAI_doc
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import ollama

from app.models.pydantic.sessions import RecordingType
from .error_finders import TextEvaluator, GrammaticalEvaluator, GrammaticalErrorsChainWrapper, SummaryChainWrapper, SummaryEvaluator
from dotenv import load_dotenv
import os

ERROR_FINDER_TYPE = "OpenAI"


class TextEvaluatorFactory:

    def __init__(self, recording_type: str):
        self._recording_type: str = recording_type

    def get_evaluator(self) -> TextEvaluator:
        chat_model = self.get_chat_model()
        processor: TextEvaluator = None
        if self._recording_type == RecordingType().COMPREHENSION:
            chain_components = SummaryChainWrapper()
            chain_components.create_output_parser()
            chain_components.create_prompt()
            # TODO: document should be defined by user after defining RecordingType.COMPREHENSION in frontend
            document = get_testing_document()
            processor = SummaryEvaluator(chat_model, chain_components, document)
        elif self._recording_type == RecordingType().LANGUAGE_PRODUCTION:
            chain_components = GrammaticalErrorsChainWrapper()
            chain_components.create_output_parser()
            chain_components.create_prompt()
            processor = GrammaticalEvaluator(chat_model, chain_components)
        else:
            NotImplementedError

        return processor

    @staticmethod
    def get_chat_model() -> Any:
        # TODO move all vars to build config (except for api key)
        GPT_TURBO = "gpt-3.5-turbo-0125"
        GPT_TURBO_INSTRUCT = "gpt-3.5-turbo-instruct"
        TEMPERATURE = 0

        env_path = os.path.join(os.getcwd(), ".env")
        load_dotenv(env_path)
        chat_model = None
        if ERROR_FINDER_TYPE == "OpenAI":
            OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
            chat_model: ChatOpenAI = ChatOpenAI(temperature=TEMPERATURE,
                                                model_name=GPT_TURBO,
                                                openai_api_key=OPENAI_API_KEY)
        elif ERROR_FINDER_TYPE == "local":
            chat_model = ollama.Ollama(model="llama3:8b")
        return chat_model


def get_testing_document():
    return afrikaans_OPENAI_doc
