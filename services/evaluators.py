import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Tuple

from langchain.pydantic_v1 import BaseModel
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, PromptTemplate

from app.models.pydantic.sessions import Recording
from pydantic_models.evaluator import SummaryEvaluationItem, SummaryEvaluations, Errors, grammatical_errors_schema, \
    summary_evaluation_item_schema, ModelFieldNotFoundError, ErrorItem, SentenceData, AudioMetrics, SyllableData
from services.converters import Audio2TextConverter
from static.summary_metrics import evaluation_metrics

import re

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
        self._create_output_parser()
        self._create_prompt()

    @abstractmethod
    def _create_prompt(self):
        """Creates prompt"""

    @abstractmethod
    def _create_output_parser(self):
        """Creates output parser"""

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

    def _create_prompt(self):
        """Creates prompt template for grammatical errors"""
        # The format instructions that LangChain makes. Let's look at them
        format_instructions = self.output_parser.get_format_instructions()

        grammatical_errors_template = """Given the following sentence, extract the grammatical errors.\n{format_instructions}\n{sentence}"""

        self.prompt_template = PromptTemplate(
            input_variables=["sentence"],
            template=grammatical_errors_template,
            partial_variables={"format_instructions": format_instructions}
        )

    def _create_output_parser(self):
        """Creates pydantic model output parser with Errors"""
        # The parser that will look for the LLM output in my schema and return it back to me
        self.output_parser = PydanticOutputParser(pydantic_object=Errors)


class SchemaGrammaticalErrorsChainWrapper(GrammaticalErrorsChainWrapper, SchemaChainWrapper):

    def invoke(self, **kwargs):
        raise NotImplementedError("Unable to invoke chain from schema for grammatical errors.")

    def transform_schema2model(self, response_schema: dict[str: str]) -> BaseModel:
        raise NotImplementedError("Unable to transform schema to model for grammatical errors.")

    def _create_prompt(self):
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

    def _create_output_parser(self):
        """Returns output parser"""
        response_schemas: List[ResponseSchema] = grammatical_errors_schema
        # The parser that will look for the LLM output in my schema and return it back to me
        self.output_parser = StructuredOutputParser.from_response_schemas(response_schemas)


class SummaryChainWrapper(ChainWrapper):
    def invoke(self, **kwargs) -> Union[SummaryEvaluationItem, dict]:
        llm: BaseLanguageModel = kwargs.get("llm")
        criteria = kwargs.get("criteria")
        document = kwargs.get("document")
        eval_type = kwargs.get("metric_name")
        steps = kwargs.get("steps")
        text = kwargs.get("summary")
        chain = self.prompt_template | llm | self.output_parser
        evaluation_result: Union[SummaryEvaluationItem, dict] = chain.invoke(
            {"criteria": criteria, "document": document,
             "metric_name": eval_type,
             "steps": steps, "summary": text})
        return evaluation_result

    def _create_prompt(self):
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

    def _create_output_parser(self):
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
        evaluation_result: dict = super().invoke(llm=llm, criteria=criteria, document=document,
                                                 metric_name=eval_type, steps=steps, summary=text)
        evaluation_result_transformed: SummaryEvaluationItem = self.transform_schema2model(evaluation_result)
        return evaluation_result_transformed

    def _create_output_parser(self):
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


class SentenceChainWrapper(ChainWrapper):
    def invoke(self, **kwargs) -> Union[SentenceData, dict]:
        llm: BaseLanguageModel = kwargs.get("llm")
        raw_text_words: List[str] = kwargs.get("raw_text_words")
        chain = self.prompt_template | llm | self.output_parser
        sentence_data: SentenceData = chain.invoke({
                                                    "raw_text_words": raw_text_words
                                                    })
        return sentence_data

    def _create_prompt(self):
        """Returns prompt"""
        # The format instructions that LangChain makes. Let's look at them
        format_instructions = self.output_parser.get_format_instructions()

        evaluation_prompt_template = """
                You are given a list comprising the words or word-sets that if stringed together in the order of their index  will yield a complete text:
                \n{raw_text_words}\n
                Extract the discrete sentences in the following format:
                \n{format_instructions}\n
                """

        self.prompt_template = PromptTemplate(
            input_variables=["raw_text_words"],
            template=evaluation_prompt_template,
            partial_variables={"format_instructions": format_instructions}
        )

    def _create_output_parser(self):
        """Creates pydantic model output parser with SummaryEvaluationItem"""
        # The parser that will look for the LLM output in my schema and return it back to me
        self.output_parser = PydanticOutputParser(pydantic_object=SentenceData)


class SyllablesChainWrapper(ChainWrapper):
    def invoke(self, **kwargs) -> Union[SyllableData, dict]:
        llm: BaseLanguageModel = kwargs.get("llm")
        text: str = kwargs.get("text")
        chain = self.prompt_template | llm | self.output_parser
        syllables_data: SyllableData = chain.invoke({"text": text})
        return syllables_data

    def _create_prompt(self):
        """Returns prompt"""
        # The format instructions that LangChain makes. Let's look at them
        format_instructions = self.output_parser.get_format_instructions()

        evaluation_prompt_template = """
                You are given the following text.
                \n{text}\n
                Extract the syllables of the text in the following format:
                \n{format_instructions}\n
                """

        self.prompt_template = PromptTemplate(
            input_variables=["text"],
            template=evaluation_prompt_template,
            partial_variables={"format_instructions": format_instructions}
        )

    def _create_output_parser(self):
        """Creates pydantic model output parser with SummaryEvaluationItem"""
        # The parser that will look for the LLM output in my schema and return it back to me
        self.output_parser = PydanticOutputParser(pydantic_object=SyllableData)


class TextEvaluator(ABC):
    def __init__(self, llm: BaseLanguageModel, chain_comps: ChainWrapper):
        self._audio2text_converter: Audio2TextConverter = None
        self._recording: Recording = None
        self._llm: BaseLanguageModel = llm
        self._chain_comps = chain_comps

    def set_llm(self, llm: BaseLanguageModel):
        self._llm = llm

    def set_chain_comps(self, chain_comps: ChainWrapper):
        self._chain_comps = chain_comps

    def set_recording(self, recording: Recording):
        self._recording = recording

    def set_audio2text_converter(self, audio2text_converter: Audio2TextConverter):
        self._audio2text_converter = audio2text_converter

    def evaluate(self) -> BaseModel:
        ...

    def process_recording(self) -> None:
        ...


class GrammaticalEvaluator(TextEvaluator):
    def __init__(self, llm: BaseLanguageModel, chain_comps: ChainWrapper):
        super().__init__(llm, chain_comps)

    def evaluate(self) -> Errors:
        """
        Finds the (grammatical) error in the text and returns it ina structured format along with a suggestion for correction
        :param text:
        :return: Errors (BaseModel)
        """
        error_items: List[ErrorItem] = []
        for sentence in self._recording.texts_timestamps.keys():
            errors: Errors = self._chain_comps.invoke(sentence=sentence, llm=self._llm)
            error_items = error_items + errors.error_items
        errors: Errors = Errors(error_items=error_items)
        logger.info("The grammatical evaluation was performed")
        return errors


class SummaryEvaluator(TextEvaluator):
    def __init__(self, llm: BaseLanguageModel, chain_comps: ChainWrapper, document: str):
        super().__init__(llm, chain_comps)
        self._document: str = document # this is the document against which the student's summary will be compared to

    def evaluate(self) -> SummaryEvaluations:
        """
        :param document:
        :param text:
        :return:
        """
        summary: str = list(self._recording.texts_timestamps.keys())[0] # TODO dont like it: i assume it's always goint to be one item for the summary use case
        evaluation: SummaryEvaluations = SummaryEvaluations()
        for eval_type, (criteria, steps) in evaluation_metrics.items():
            evaluation_result: BaseModel = self._chain_comps.invoke(llm=self._llm, criteria=criteria,
                                                                    document=self._document,
                                                                    metric_name=eval_type, steps=steps, summary=summary)
            evaluation.evaluations.append(evaluation_result)

        logger.info("The summary evaluation was performed")
        return evaluation

    def set_document(self, document: str):
        self._document = document


class ReadingEvaluator(TextEvaluator):
    def __init__(self, llm: BaseLanguageModel, chain_comps: ChainWrapper):
        super().__init__(llm, chain_comps)

    def evaluate(self) -> AudioMetrics:
        audio_metrics = AudioMetrics()
        words_per_sec_sum: float = 0.0
        syllables_per_sec_sum: float = 0.0
        pause_duration_sum: float = 0.0
        word_count_sum: int = 0
        end_time: float = 0.0  # set initial end_time
        for text_bit, time_frame in self._recording.texts_timestamps.items():
            # -----calculate values for avg_pause_duration-----
            # current start_time
            start_time = time_frame[0]
            # duration equal the difference between current start_time and previous end_time
            pause_duration = start_time - end_time
            pause_duration_sum = pause_duration_sum + pause_duration
            # current end_time
            end_time = time_frame[1]
            # -----calculate values for avg_words_per_sec-----
            # Use regex to find words (ignoring punctuation)
            words = re.findall(r'\b\w+\b', text_bit)
            word_count = len(words)
            time_diff = end_time - start_time
            words_per_sec = word_count / time_diff
            word_count_sum = word_count_sum + word_count
            words_per_sec_sum = words_per_sec_sum + words_per_sec
            # -----calculate values for avg_syllables_per_sec-----
            syllable_data: SyllableData = self._chain_comps.invoke(text=text_bit, llm=self._llm)
            syllables_per_sec = syllable_data.count / time_diff
            syllables_per_sec_sum = syllables_per_sec_sum + syllables_per_sec

        text_bits_count = len(self._recording.texts_timestamps.keys())
        audio_metrics.avg_pause_duration = pause_duration_sum / (text_bits_count - 1)
        audio_metrics.pause_frequency_per_word_count = (text_bits_count - 1) / word_count_sum
        audio_metrics.avg_words_per_sec = words_per_sec_sum / text_bits_count
        audio_metrics.avg_syllables_per_sec = syllables_per_sec_sum / text_bits_count

        return audio_metrics

    def transform_to_sentence_based_evaluation(self):
        # TODO: this should be responsibility of a factory
        self.set_chain_comps(SentenceChainWrapper())
        # TODO chain SentenceChainWrapper nor SentenceData support sentence ending in the middle
        #  of the string (key in self._recording.texts_timestamps), as opposed to ending at the end of the string
        sentence_data: SentenceData = self._chain_comps.invoke(raw_text_words=list(self._recording.texts_timestamps.keys()),
                                                               llm=self._llm)
        sentence_start_time = 0.0
        current_sentence_idx = 0
        idx = 0
        sentence_based_texts_timestamps: Dict[str, Tuple[float, float]] = {}
        for text_bit, time_frame in self._recording.texts_timestamps.items():
            # -----calculate values for avg_words_per_sec_per_sentence-----
            current_sentence = sentence_data.sentences[current_sentence_idx]
            # get start indices of current and next sentence
            current_sentence_start_idx = sentence_data.sentence_start_indices[current_sentence_idx]
            if len(sentence_data.sentence_start_indices)-1 > current_sentence_idx:
                next_sentence_start_idx = sentence_data.sentence_start_indices[current_sentence_idx + 1]
            else:
                # there is not next sentence start index, so it's equal the size
                next_sentence_start_idx = len(self._recording.texts_timestamps.items())
            if idx == current_sentence_start_idx:
                sentence_start_time = time_frame[0]
            if idx == next_sentence_start_idx-1:
                sentence_end_time = time_frame[1]
                sentence_based_texts_timestamps.update({current_sentence:
                                                        (sentence_start_time, sentence_end_time)})
                current_sentence_idx += 1
            idx += 1

        self._recording.texts_timestamps = sentence_based_texts_timestamps
