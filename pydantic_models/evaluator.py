from langchain.pydantic_v1 import BaseModel, Field
from typing import Optional, List, Union


class SummaryEvaluationItem(BaseModel):
    metric: Optional[str] = Field(None, description="The Metric used score")
    score: Optional[int] = Field(None,
                                 description="The score for the summary evaluation metric (e.g., relevance, coherence), if applicable.")
    reason: Optional[str] = Field(None, description="The description for the given score")


class SummaryEvaluations(BaseModel):
    evaluations: List[SummaryEvaluationItem] = []

