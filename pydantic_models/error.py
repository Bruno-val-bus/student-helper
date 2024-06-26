from langchain.pydantic_v1 import BaseModel, Field
from typing import Optional, List


class ErrorItem(BaseModel):
    error: str = Field(..., description="A grammatical error in the sentence.")
    correction: str = Field(..., description="A correction of the grammatical error in the sentence.")
    category: Optional[str] = Field(None, description="An error category that the error belongs to.")


class Errors(BaseModel):
    error: List[ErrorItem] = Field(...,
                                   description="A list of errors representing all possible grammatical errors in the sentence.")
