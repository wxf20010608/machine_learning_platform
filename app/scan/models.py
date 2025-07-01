from pydantic import BaseModel

class ScanResponse(BaseModel):
    original_image: str
    processed_image: str
    message: str
