from pydantic import BaseModel
from typing import Optional, List, Literal

class RecommendRequest(BaseModel):
    query: Optional[str] = None
    user_id: Optional[int] = None
    k: int = 5
    alpha: float = 0.5
    include_reasons: bool = True


class UserCreate(BaseModel):
    user_id: int
    skills: List[str] = []
    interests: List[str] = []
    experience: str = ""
    profile_text: str = ""


class OpportunityCreate(BaseModel):
    opportunity_id: int
    title: str
    description: str
    skills_required: List[str] = []
    category: str = ""
    location: str = ""
    opportunity_type: str = ""


class InteractionCreate(BaseModel):
    user_id: int
    opportunity_id: int
    event_type: Literal["view", "click", "save", "apply", "implicit"] = "implicit"
    weight: float = 1.0

class OpportunityOut(BaseModel):
    opportunity_id: int
    title: str
    description: str
    score: Optional[float] = None
    reason: Optional[str] = None

class RecommendResponse(BaseModel):
    results: List[OpportunityOut]
