
from typing import Optional
from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: Optional[str] = None