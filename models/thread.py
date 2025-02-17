from sqlalchemy import Boolean, Column, String, Text
from sqlalchemy.orm import declarative_base


Base = declarative_base()


class Thread(Base):
    __tablename__ = "threads"
    thread_id = Column(String, primary_key=True, index=True)
    question_asked = Column(Boolean, default=False)
    question = Column(String, nullable=True)
    answer = Column(Text, nullable=True)
    confirmed = Column(Boolean, default=False)
    error = Column(Boolean, default=False)