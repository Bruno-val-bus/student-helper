from langchain.pydantic_v1 import BaseModel, Field
from typing import Optional, List


class SummaryEvaluationItem(BaseModel):
    metric: str = Field(..., description="The name of the metric used")
    score: int = Field(..., description="The score given based on the metric")
    reason: Optional[str] = Field(None, description="The provided reason for the given score")


class SummaryEvaluations(BaseModel):
    evaluations: List[SummaryEvaluationItem] = [] #Field(..., description="A list of scores of all considered evaluation metrics for the provided summary")


class ErrorItem(BaseModel):
    error: str = Field(..., description="A grammatical error in the sentence.")
    correction: str = Field(..., description="A correction of the grammatical error in the sentence.")
    category: Optional[str] = Field(None, description="An error category that the error belongs to.")


class Errors(BaseModel):
    error: List[ErrorItem] = Field(...,
                                   description="A list of errors representing all possible grammatical "
                                               "errors in the sentence.")
