from typing import Optional
from pydantic import BaseModel

from pydantic_models.error import Errors, Evaluation


class RecordingType:
    COMPREHENSION = "COMPREHENSION"
    LANGUAGE_PRODUCTION = "LANGUAGE_PRODUCTION"
    PRESENTATION = "PRESENTATION"


class RecordingStatus:
    AUDIO_PROCESSED = "AUDIO_PROCESSED"
    AUDIO_SAVED = "AUDIO_SAVED"
    SAVING_AUDIO = "SAVING_AUDIO"
    NO_AUDIO_SAVED = "NO_AUDIO_SAVED"
    PROCESSING_AUDIO = "PROCESSING_AUDIO"


class Recording(BaseModel):
    id: Optional[int] = None
    session_id: Optional[int] = None
    type: Optional[RecordingType] = None
    start: Optional[str] = None
    end: Optional[str] = None
    audio_file_path: Optional[str] = None
    status: Optional[RecordingStatus] = None
    evaluation: Optional[Evaluation] = None


class Session(BaseModel):
    id: Optional[int] = None
    start: Optional[str] = None
    end: Optional[str] = None
    facilitator_id: Optional[int] = None
    student_id: Optional[int] = None
    recordings: Optional[list[Recording]] = None
