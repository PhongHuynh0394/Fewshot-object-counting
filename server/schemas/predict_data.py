from pydantic import BaseModel, Field

class Exampler(BaseModel):
    boxes: list[list[float]]