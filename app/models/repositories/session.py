from app.models.pydantic.sessions import Session


class SessionRepo:

    def __init__(self):
        pass

    def get_session(self, session_id: int) -> Session:
        """
        TODO implement
        :param session_id:
        :return:
        """
        return Session()

    def update_session(self, session: Session) -> Session:
        """
        TODO implement
        :param session:
        :return:
        """
        return Session()

    def patch_session_attributes(self, session_id: int, **attributes) -> Session:
        """
         TODO implement
         :param session_id:
         :return:
         """
        for attr, value in attributes.items():
            pass
        return self.get_session(session_id)

    def create_session(self, session: Session) -> Session:
        """
        TODO implement
        :param session:
        :return:
        """
        return Session()
