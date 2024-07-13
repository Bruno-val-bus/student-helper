from fastapi import FastAPI, File, UploadFile, HTTPException
from app.models.pydantic.sessions import Session, Recording, RecordingStatus, RecordingType
from app.models.repositories.recording import RecordingRepo
from app.models.repositories.session import SessionRepo
from services.error_finder_factory import TextEvaluatorFactory
from langchain.pydantic_v1 import BaseModel

app = FastAPI()

recording_repo = RecordingRepo()
session_repo = SessionRepo()


def _attributes_not_none(obj, attributes: list):
    for attribute in attributes:
        if getattr(obj, attribute) is None:
            return False
    return True


@app.post("/session/")
async def create_session(session: Session) -> Session:
    client_atts = ["facilitator_id", "student_id"]
    if _attributes_not_none(session, client_atts):
        session.start = ""  # TODO implement
        session_repo.create_session(session)
        return session
    else:
        raise HTTPException(status_code=400,
                            detail=f"Missing required attributes {str(client_atts)} in {str(session.__class__.__name__)}")


@app.patch("/session/{session_id}")
async def finish_session(session_id: int) -> Session:
    session: Session = session_repo.patch_session_attributes(session_id, end="")
    return session


@app.post("/session/{session_id}/recording/")
async def create_recording(session_id: int, recording: Recording) -> Recording:
    client_atts = ["type"]
    if _attributes_not_none(recording, client_atts):
        recording.session_id = session_id
        recording.start = ""  # TODO implement
        recording.status = RecordingStatus.NO_AUDIO_SAVED
        recording_repo.create_recording(recording)
        return recording
    else:
        raise HTTPException(status_code=400,
                            detail=f"Missing required attributes {str(client_atts)} in {str(recording.__class__.__name__)}")


@app.patch("recording/{recording_id}/audio")
async def finish_recording(recording_id: int, file: UploadFile = File(...)) -> Recording:
    try:
        recording: Recording = recording_repo.get_recording(recording_id)
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Could not find recording with id: {recording_id}\n{str(e)}")
    #  finish recording
    recording_end = ""  # TODO implement
    #  check if currently saving audio
    if recording.status == RecordingStatus.SAVING_AUDIO:
        raise HTTPException(status_code=500,
                            detail=f"Currently saving audio for recording with id: {recording_id}")

    try:
        #  save audio
        await recording_repo.save_audio_file(file, recording)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File saving failed: {str(e)}")
    recording_status = RecordingStatus.SAVING_AUDIO
    # patch recording and return it
    recording = recording_repo.patch_recording_attributes(recording.id, end=recording_end, status=recording_status)
    return recording


@app.post("recording/{recording_id}/evaluation")
async def process_recording(recording_id: int) -> Recording:
    try:
        recording: Recording = recording_repo.get_recording(recording_id)
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Could not find recording with id: {recording_id}\n{str(e)}")

    if recording.status != RecordingStatus.AUDIO_SAVED:
        # processing only possible with status AUDIO_SAVED
        raise HTTPException(status_code=500,
                            detail=f"Cannot process audio because of current status {recording.status} of recording id"
                                   f": {recording_id}")
    txt_proc_fact = TextEvaluatorFactory(recording.type)
    evaluator = txt_proc_fact.get_evaluator()
    # TODO logic with processor: async processor.process(recording: Recording, recording_repo: RecordingRepo)
    #  recording.evaluation = evaluator.evaluate(summary)
    # update status to PROCESSING_AUDIO
    recording_status = RecordingStatus.PROCESSING_AUDIO
    recording = recording_repo.patch_recording_attributes(recording.id, status=recording_status)
    return recording


@app.get("recording/{recording_id}/evaluation")
async def get_recording_evaluation(recording_id: int) -> BaseModel:
    try:
        recording: Recording = recording_repo.get_recording(recording_id)
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Could not find recording with id: {recording_id}\n{str(e)}")
    return recording.evaluation


@app.get("recording/{recording_id}/status")
async def get_recording_status(recording_id: int) -> RecordingStatus:
    try:
        recording: Recording = recording_repo.get_recording(recording_id)
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Could not find recording with id: {recording_id}\n{str(e)}")
    return recording.status


@app.get("/session/{session_id}/recording/")
async def get_session_recordings(session_id: int) -> list[Recording]:
    pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
