import os
import logging
from dotenv import load_dotenv
from langchain_core.language_models import BaseLanguageModel
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import ollama

from static.summary_example_text import afrikaans_OPENAI_doc

from app.models.pydantic.sessions import RecordingType
from .error_finders import TextEvaluator, GrammaticalEvaluator, GrammaticalErrorsChainWrapper, SummaryChainWrapper, SummaryEvaluator

from configs.configurator import ConfigInterface


# init module logger
logger = logging.getLogger(__name__)


class TextEvaluatorFactory:

    def __init__(self, recording_type: str, config: ConfigInterface):
        self._recording_type: str = recording_type
        self.config = config

    def get_evaluator(self) -> TextEvaluator:
        llm = self.get_llm()
        processor: TextEvaluator
        try:
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

        except NotImplementedError as e:
            logger.exception("An error occured: %s", e)

        return processor

    def get_llm(self) -> BaseLanguageModel:
        llm = None
        try:
            # Use predefined config to choose what LLM to use
            if self.config.get_llm_setup_name() == "ONLINE_OPENAI_GPT3":
                llm_setup = self.config.get_llm_setup_params()
                env_path = os.path.join(os.getcwd(), ".env")
                load_dotenv(env_path)
                llm: ChatOpenAI = ChatOpenAI(temperature=llm_setup['TEMPERATURE'],
                                             model_name=llm_setup['MODEL_NAME'],
                                             openai_api_key=os.environ["OPENAI_API_KEY"])
                return llm

            elif self.config.get_llm_setup_name() == "LOCAL_OLLAMA_LLAMA":
                llm_setup = self.config.get_llm_setup_params()
                llm = ollama.Ollama(model=llm_setup['MODEL_NAME'])
                return llm

            else:
                raise NotImplementedError(f"Model setup {self.config.get_llm_setup_name()} is not supported.")

        except NotImplementedError as e:
            logger.exception("An error occured: %s", e)


def get_testing_document():
    return afrikaans_OPENAI_doc
