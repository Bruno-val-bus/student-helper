import os
import logging
from dotenv import load_dotenv
from langchain_core.language_models import BaseLanguageModel
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import ollama

from static.summary_example_text import afrikaans_OPENAI_doc

from app.models.pydantic.sessions import RecordingType
from .error_finders import TextEvaluator, GrammaticalEvaluator, GrammaticalErrorsChainWrapper, SummaryChainWrapper, \
    SummaryEvaluator, SchemaSummaryChainWrapper, ChainWrapper, SchemaGrammaticalErrorsChainWrapper

from configs.configurator import ConfigInterface

# init module logger
logger = logging.getLogger(__name__)


class TextEvaluatorFactory:

    def __init__(self, recording_type: str, config: ConfigInterface):
        self._recording_type: str = recording_type
        self._config = config

    def get_evaluator(self) -> TextEvaluator:
        llm = self.get_llm()
        output_parser_type = self._config.get_llm_output_parser(self._recording_type)
        processor: TextEvaluator
        try:
            if self._recording_type == RecordingType.COMPREHENSION:
                chain_components: ChainWrapper
                if output_parser_type == "model":
                    chain_components = SummaryChainWrapper()
                elif output_parser_type == "schema":
                    chain_components = SchemaSummaryChainWrapper()
                else:
                    raise NotImplementedError(f"Output parser type {output_parser_type} has not been implemented yet")

                chain_components.create_output_parser()
                chain_components.create_prompt()
                # TODO: document should be defined by user after defining RecordingType.COMPREHENSION in frontend
                document = get_testing_document()
                processor: SummaryEvaluator = SummaryEvaluator(llm, chain_components, document)
            elif self._recording_type == RecordingType.LANGUAGE_PRODUCTION:
                chain_components: ChainWrapper
                if output_parser_type == "model":
                    chain_components = GrammaticalErrorsChainWrapper()
                elif output_parser_type == "schema":
                    chain_components = SchemaGrammaticalErrorsChainWrapper()
                else:
                    raise NotImplementedError(f"Output parser type {output_parser_type} has not been implemented yet")

                chain_components.create_output_parser()
                chain_components.create_prompt()
                processor: GrammaticalEvaluator = GrammaticalEvaluator(llm, chain_components)
            else:
                raise NotImplementedError(f"Recording type {self._recording_type} has not been implemented yet")
            return processor

        except Exception as e:
            logger.exception("An error occurred: %s", e)

    def get_llm(self) -> BaseLanguageModel:
        try:
            llm: BaseLanguageModel
            # Use predefined config to choose what LLM to use
            if self._config.get_llm_setup_name() == "ONLINE_OPENAI_GPT3":
                llm_setup = self._config.get_llm_setup_params()
                env_path = os.path.join(os.getcwd(), ".env")
                load_dotenv(env_path)
                llm: ChatOpenAI = ChatOpenAI(temperature=llm_setup['TEMPERATURE'],
                                             model_name=llm_setup['MODEL_NAME'],
                                             openai_api_key=os.environ["OPENAI_API_KEY"])

            elif self._config.get_llm_setup_name() == "LOCAL_OLLAMA_LLAMA3":
                # host llm on localhost
                llm_setup = self._config.get_llm_setup_params()
                llm = ollama.Ollama(model=llm_setup['MODEL_NAME'])

            elif self._config.get_llm_setup_name() == "LOCAL_DOCKER_OLLAMA_LLAMA3":
                # host llm in docker network
                llm_setup = self._config.get_llm_setup_params()
                host = llm_setup['OLLAMA_HOST']
                port = llm_setup['OLLAMA_PORT']
                llm = ollama.Ollama(model=llm_setup['MODEL_NAME'],
                                    base_url=f'http://{host}:{port}')
            else:
                raise NotImplementedError(f"Model setup {self._config.get_llm_setup_name()} is not supported.")
            return llm

        except NotImplementedError as e:
            logger.exception("An error occurred: %s", e)


def get_testing_document():
    return afrikaans_OPENAI_doc
