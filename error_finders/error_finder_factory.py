from langchain_community.chat_models import ChatOpenAI

from .error_finders import ErrorFinder, OpenAIErrorFinderByModel, OpenAIErrorFinderBySchema
from dotenv import load_dotenv
import os


def create_error_finder() -> ErrorFinder:
    # TODO move all vars to build config (except for api key)
    ERROR_FINDER_TYPE = "OpenAI"
    ERROR_FIND_METHOD = "model"
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
    if ERROR_FINDER_TYPE == "OpenAI":
        if ERROR_FIND_METHOD == "model":
            error_finder = OpenAIErrorFinderByModel(chat_model)
        elif ERROR_FIND_METHOD == "schema":
            error_finder = OpenAIErrorFinderBySchema(chat_model)
    elif ERROR_FINDER_TYPE == "local":
        if ERROR_FIND_METHOD == "model":
            pass
        elif ERROR_FIND_METHOD == "schema":
            pass
    return error_finder
