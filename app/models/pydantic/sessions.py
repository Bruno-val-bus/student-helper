from typing import Optional, Dict, List
from pydantic import BaseModel


class RecordingType:
    COMPREHENSION: str = "COMPREHENSION"
    LANGUAGE_PRODUCTION: str = "LANGUAGE_PRODUCTION"
    PRESENTATION: str = "PRESENTATION"


class RecordingStatus(BaseModel):
    AUDIO_PROCESSED: str = "AUDIO_PROCESSED"
    AUDIO_SAVED: str = "AUDIO_SAVED"
    SAVING_AUDIO: str = "SAVING_AUDIO"
    NO_AUDIO_SAVED: str = "NO_AUDIO_SAVED"
    PROCESSING_AUDIO: str = "PROCESSING_AUDIO"


class Recording(BaseModel):
    id: Optional[int] = None
    session_id: Optional[int] = None
    type: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    audio_file_path: Optional[str] = None
    status: Optional[RecordingStatus] = None
    evaluation: Optional[BaseModel] = None
    texts_timestamps: Optional[Dict[str, float]] = None
    audio_segments_paths: Optional[List[str]] = None


class Session(BaseModel):
    id: Optional[int] = None
    start: Optional[str] = None
    end: Optional[str] = None
    facilitator_id: Optional[int] = None
    student_id: Optional[int] = None
    recordings: Optional[list[Recording]] = None
