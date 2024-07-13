import os
from fastapi import UploadFile
from app.models.pydantic.sessions import Recording

# Directory to save uploaded files
UPLOAD_DIRECTORY = "uploads/"


class RecordingRepo:

    def __init__(self):
        # Ensure the directory exists
        os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

    def get_recording(self, recording_id: int) -> Recording:
        """
        TODO implement
        :param recording_id:
        :return:
        """
        return Recording()

    def update_recording(self, recording: Recording) -> Recording:
        """
        TODO implement
        :param recording:
        :return:
        """
        return Recording()

    def patch_recording_attributes(self, recording_id: int, **attributes) -> Recording:
        """
         TODO implement
         :param recording_id:
         :return:
         """
        for attr, value in attributes.items():
            pass
        return self.get_recording(recording_id)

    def create_recording(self, recording: Recording) -> Recording:
        """
        TODO implement
        :param recording:
        :return:
        """
        return Recording()

    async def save_audio_file(self, file: UploadFile, recording: Recording) -> Recording:
        file_path = f"{UPLOAD_DIRECTORY}/{recording.session_id}_{recording.id}_{file.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        recording.audio_file_path = file_path
        return recording
