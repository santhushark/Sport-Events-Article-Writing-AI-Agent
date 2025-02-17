from pydantic import BaseModel



class UpdateStateRequest(BaseModel):
    answer: str