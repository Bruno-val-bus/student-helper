from langchain_community.chat_models import ChatOpenAI

from .summary_evaluators import SummaryEvaluator, SummaryChainWrapper, ChainWrapper, OpenAISummaryEvaluator

from dotenv import load_dotenv
import os

ERROR_FINDER_TYPE = "OpenAI"
ERROR_FIND_METHOD = "model"


def create_summary_evaluator() -> SummaryEvaluator:
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
    summary_evaluator: SummaryEvaluator = None
    chain_comps: ChainWrapper = initialize_summary_chain_components()
    if ERROR_FINDER_TYPE == "OpenAI":
        summary_evaluator = OpenAISummaryEvaluator(chat_model, chain_comps)
    elif ERROR_FINDER_TYPE == "local":
        pass
    return summary_evaluator


def initialize_summary_chain_components() -> ChainWrapper:
    chain_comps: ChainWrapper = None
    if ERROR_FIND_METHOD == "model":
        chain_comps = SummaryChainWrapper()
    chain_comps.create_output_parser()
    chain_comps.create_prompt()
    return chain_comps

