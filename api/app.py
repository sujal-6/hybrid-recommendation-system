from fastapi import FastAPI, Depends

from api.schemas import (
    InteractionCreate,
    OpportunityCreate,
    OpportunityOut,
    RecommendRequest,
    RecommendResponse,
    UserCreate,
)
from api.deps import load_models, reset_models
from src.data.db import DEFAULT_DB_PATH, insert_interaction, insert_opportunity, upsert_user

app = FastAPI(
    title="Opportunity Recommendation API",
    version="1.0.0"
)

@app.get("/")
def health():
    return {"status": "OK", "message": "Recommender API is running"}

@app.post("/reload")
def reload_models():
    reset_models()
    return {"status": "OK", "note": "Models will be rebuilt on next request to /recommend."}

@app.post("/users")
def create_user(payload: UserCreate):
    upsert_user(
        db_path=DEFAULT_DB_PATH,
        user_id=payload.user_id,
        skills=payload.skills,
        interests=payload.interests,
        experience=payload.experience,
        profile_text=payload.profile_text,
    )
    return {"status": "OK", "user_id": payload.user_id}


@app.post("/opportunities")
def create_opportunity(payload: OpportunityCreate):
    insert_opportunity(
        db_path=DEFAULT_DB_PATH,
        opportunity_id=payload.opportunity_id,
        title=payload.title,
        description=payload.description,
        skills_required=payload.skills_required,
        category=payload.category,
        location=payload.location,
        opportunity_type=payload.opportunity_type,
    )
    # ModelStore is cached; simplest MVP approach: restart API to pick up changes
    return {"status": "OK", "opportunity_id": payload.opportunity_id, "note": "Restart API to refresh models."}


@app.post("/interactions")
def create_interaction(payload: InteractionCreate):
    insert_interaction(
        db_path=DEFAULT_DB_PATH,
        user_id=payload.user_id,
        opportunity_id=payload.opportunity_id,
        event_type=payload.event_type,
        weight=payload.weight,
    )
    return {"status": "OK"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest, store=Depends(load_models)):
    opps = store.opps
    system = store.recommender
    recs = system.recommend(user_id=req.user_id, query=req.query, k=req.k, alpha=req.alpha)

    results = []
    for r in recs:
        row = opps.iloc[r.item_idx]
        results.append(
            OpportunityOut(
                opportunity_id=int(row["opportunity_id"]),
                title=row["title"],
                description=row["description"],
                score=r.score,
                reason=r.reason if req.include_reasons else None,
            )
        )

    return RecommendResponse(results=results)
