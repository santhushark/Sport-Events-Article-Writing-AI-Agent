
from typing import Optional
from pydantic import BaseModel



class ThreadResponse(BaseModel):
    thread_id: str
    question_asked: bool
    question: Optional[str] = None
    answer: Optional[str] = None
    confirmed: bool
    error: bool