
from typing import Optional
from pydantic import BaseModel


class ChatRequest(BaseModel):
    sport_event: Optional[str] = None