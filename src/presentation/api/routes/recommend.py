from fastapi import APIRouter
from src.application.recommendation.use_case import get_recommendations

router = APIRouter(prefix="/reco", tags=["recommendations"])

@router.post("/get_recommendations")
async def get_recommendations_endpoint(user_id: int, top_k: int = 10, page_size: int = 200):
    return await get_recommendations(user_id=user_id, top_k=top_k, page_size=page_size)
