from static.summary_example_text import afrikaans_OPENAI_doc
from langchain_core.language_models import BaseLanguageModel
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import ollama

from app.models.pydantic.sessions import RecordingType
from .error_finders import TextEvaluator, GrammaticalEvaluator, GrammaticalErrorsChainWrapper, SummaryChainWrapper, SummaryEvaluator
from dotenv import load_dotenv
import os

# TODO move all vars to build config
MODEL_PROVIDER = "OpenAI"  # "OpenAI", "Ollama"
LOCAL_MODEL = False
MODEL_NAME = "gpt-3.5-turbo-0125"  # "gpt-3.5-turbo-0125", "gpt-3.5-turbo-instruct", "llama3:8b"
TEMPERATURE = 0

# Get variables from env file #TODO: create a config.yaml file to set variables
OLLAMA_HOST = os.getenv('OLLAMA_HOST')
OLLAMA_PORT = os.getenv('OLLAMA_PORT')


class TextEvaluatorFactory:

    def __init__(self, recording_type: str):
        self._recording_type: str = recording_type

    def get_evaluator(self) -> TextEvaluator:
        llm = self.get_llm()
        processor: TextEvaluator
        if self._recording_type == RecordingType.COMPREHENSION:
            chain_components = SummaryChainWrapper()
            chain_components.create_output_parser()
            chain_components.create_prompt()
            # TODO: document should be defined by user after defining RecordingType.COMPREHENSION in frontend
            document = get_testing_document()
            processor: SummaryEvaluator = SummaryEvaluator(llm, chain_components, document)
        elif self._recording_type == RecordingType.LANGUAGE_PRODUCTION:
            chain_components = GrammaticalErrorsChainWrapper()
            chain_components.create_output_parser()
            chain_components.create_prompt()
            processor: GrammaticalEvaluator = GrammaticalEvaluator(llm, chain_components)
        else:
            raise NotImplementedError(f"{self._recording_type} has not been implemented yet")

        return processor

    @staticmethod
    def get_llm() -> BaseLanguageModel:
        llm = None
        if not LOCAL_MODEL:
            if MODEL_PROVIDER == "OpenAI":
                env_path = os.path.join(os.getcwd(), ".env")
                load_dotenv(env_path)
                OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
                llm: ChatOpenAI = ChatOpenAI(temperature=TEMPERATURE,
                                             model_name=MODEL_NAME,
                                             openai_api_key=OPENAI_API_KEY)
            elif MODEL_PROVIDER == "LlamaCpp":
                # TODO implement remote model for LlamaCpp
                pass
            else:
                NotImplementedError(f"Model provider {MODEL_PROVIDER} not supported remotely.")
        else:
            if MODEL_PROVIDER == "OpenAI":
                # TODO implement local model for OpenAI
                pass
            elif MODEL_PROVIDER == "Ollama":
                llm = ollama.Ollama(model=MODEL_NAME, base_url= f'http://{OLLAMA_HOST}:{OLLAMA_PORT}')
            else:
                NotImplementedError(f"Model provider {MODEL_PROVIDER} not supported locally.")
        return llm


def get_testing_document():
    return afrikaans_OPENAI_doc
