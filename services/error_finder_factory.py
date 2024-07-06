from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import LlamaCpp

from app.models.pydantic.sessions import RecordingType
from .error_finders import ErrorFinder, OpenAIErrorFinder, ChainWrapper, \
    SchemaChainWrapper, ModelChainWrapper
from dotenv import load_dotenv
import os

ERROR_FINDER_TYPE = "OpenAI"
ERROR_FIND_METHOD = "model"


def get_processor(recording_type: RecordingType):
    if recording_type == RecordingType.COMPREHENSION:
        pass
    elif recording_type == RecordingType.LANGUAGE_PRODUCTION:
        pass
    else:
        NotImplementedError


def create_error_finder() -> ErrorFinder:
    # TODO move all vars to build config (except for api key)
    GPT_TURBO = "gpt-3.5-turbo-0125"
    GPT_TURBO_INSTRUCT = "gpt-3.5-turbo-instruct"
    TEMPERATURE = 0

    env_path = os.path.join(os.getcwd(), ".env")
    load_dotenv(env_path)
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    chat_model: ChatOpenAI = ChatOpenAI(temperature=TEMPERATURE,
                                        model_name=GPT_TURBO,
                                        openai_api_key=OPENAI_API_KEY)
    error_finder: ErrorFinder = None
    chain_comps: ChainWrapper = initialize_chain_components()
    if ERROR_FINDER_TYPE == "OpenAI":
        error_finder = OpenAIErrorFinder(chat_model, chain_comps)
    elif ERROR_FINDER_TYPE == "local":
        pass
    return error_finder


def initialize_chain_components() -> ChainWrapper:
    chain_comps: ChainWrapper = None
    if ERROR_FIND_METHOD == "model":
        chain_comps = ModelChainWrapper()
    elif ERROR_FIND_METHOD == "schema":
        chain_comps = SchemaChainWrapper()
    chain_comps.create_output_parser()
    chain_comps.create_prompt()
    return chain_comps
