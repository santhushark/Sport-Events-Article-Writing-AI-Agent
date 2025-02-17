from pydantic import BaseModel



class StartThreadResponse(BaseModel):
    thread_id: str