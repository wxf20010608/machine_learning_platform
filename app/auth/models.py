from pydantic import BaseModel, EmailStr
from typing import Optional

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

class EmailVerificationRequest(BaseModel):
    email: EmailStr

class VerificationLoginRequest(BaseModel):
    email: EmailStr
    verification_code: str

class GitHubLoginRequest(BaseModel):
    code: str

