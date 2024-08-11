from langchain.pydantic_v1 import BaseModel, Field
from typing import Optional, List, Union
from langchain.output_parsers import ResponseSchema


class ModelFieldNotFoundError(Exception):
    def __init__(self, field_name):
        super().__init__(f"Model field '{field_name}' not found in the response schema.")
        self.field_name = field_name


class SummaryEvaluationItem(BaseModel):
    metric: Optional[str] = Field(None, description="Name of the metric")
    score: Optional[int] = Field(None, description="The given score out of 10 (0 extremely poor and 10 extremely good)")
    reason: Optional[str] = Field(None, description="The reason for the score in less than 15 words")


summary_evaluation_item_schema: List[ResponseSchema] = [
    ResponseSchema(
        name="metric",
        description="Name of the metric",
        type="str"),
    ResponseSchema(
        name="score",
        description="The given score out of 10 (0 extremely poor and 10 extremely good)",
        type="int"
    ),
    ResponseSchema(
        name="reason",
        description="The reason for the score in less than 15 words",
        type="str")
]


class SummaryEvaluations(BaseModel):
    evaluations: List[Union[
        SummaryEvaluationItem, BaseModel]] = []  # Field(..., description="A list of scores of all considered evaluation metrics for the provided summary")


class ErrorItem(BaseModel):
    error: str = Field(..., description="A grammatical error in the sentence.")
    correction: str = Field(..., description="A correction of the grammatical error in the sentence.")
    category: Optional[str] = Field(None, description="An error category that the error belongs to.")


class Errors(BaseModel):
    error: List[ErrorItem] = Field(...,
                                   description="A list of errors representing all possible grammatical "
                                               "errors in the sentence.")


grammatical_errors_schema: List[ResponseSchema] = [
    ResponseSchema(name="grammatical_errors",
                   description="A list of strings. Each string corresponding to a grammatical error found in the sentence.",
                   type="List[string]"),
    ResponseSchema(name="grammatical_errors_correction",
                   description="A dictionary of strings as keys and strings as values. Each key corresponding to a grammatical error found in the sentence and each value corresponding to its correction",
                   type="Dict[string, string]"),
]
