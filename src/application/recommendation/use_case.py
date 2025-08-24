from typing import Dict, Any
from src.infrastructure.http_client import RailwayClient
from src.domain.state import api_data, api_lock
from src.application.recommendation.transformers import add_user_behaviors_from_api, update_products_from_api
from src.ml.scoring import recommend_for_user

async def get_recommendations(user_id: int, top_k: int = 10, page_size: int = 200) -> Dict[str, Any]:
    client = RailwayClient()
    behaviors_data = await client.get_user_actions(user_id)
    products_payload = await client.get_products(page_size=page_size, page_number=1)
    if not products_payload.get("Success") or not products_payload.get("Data"):
        return {"status": "error", "message": "Failed to fetch products from API", "user_id": user_id}
    with api_lock:
        api_data["behaviors"].clear()
        api_data["products"].clear()
        api_data["users"].clear()
        api_data["sellers"].clear()
        add_user_behaviors_from_api(behaviors_data)
        products_list = products_payload["Data"]
        update_products_from_api(products_list)
        recs = recommend_for_user(user_id_raw=user_id, top_k=top_k, candidate_limit=len(products_list))
    return {
        "status": "success",
        "user_id": user_id,
        "recommendations": recs,
        "behaviors_processed": len(behaviors_data),
        "products_available": len(products_list),
        "total_products_in_api": products_payload.get("TotalCount", 0),
    }
