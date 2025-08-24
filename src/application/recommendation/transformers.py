from typing import List, Dict, Any
from src.domain.state import api_data

def add_user_behaviors_from_api(api_data_list: List[Dict[str, Any]]):
    for behavior in api_data_list:
        gender_raw = behavior.get("UserGender")
        behavior_data = {
            "user_id": behavior["UserId"],
            "product_id": behavior["ProductId"],
            "behavior_type": behavior["ActionType"],
            "behavior_time": behavior["Timestamp"],
            "gender": gender_raw,
            "product_name": behavior.get("ProductName"),
            "category": behavior.get("ProductCategory") or behavior.get("CategoryName"),
            "type": ("digital" if behavior.get("IsDigital") else "physical") if behavior.get("IsDigital") is not None else behavior.get("Type"),
            "price": behavior.get("ProductPrice"),
            "trader_id": behavior.get("SellerId"),
            "shop_id": behavior.get("SellerId"),
        }
        api_data["behaviors"].append(behavior_data)
        api_data["products"][behavior["ProductId"]] = {
            "product_id": behavior["ProductId"],
            "name": behavior.get("ProductName"),
            "price": behavior.get("ProductPrice"),
            "category": behavior.get("ProductCategory") or behavior.get("CategoryName"),
            "type": ("digital" if behavior.get("IsDigital") else "physical") if behavior.get("IsDigital") is not None else behavior.get("Type"),
            "trader_id": behavior.get("SellerId"),
        }
        api_data["users"][behavior["UserId"]] = {"user_id": behavior["UserId"], "gender": gender_raw}
        api_data["sellers"][behavior["SellerId"]] = {"user_id": behavior["SellerId"], "shop_id": behavior["SellerId"]}

def update_products_from_api(products_data: List[Dict[str, Any]]):
    for product in products_data:
        product_id = product.get("Id")
        if not product_id:
            continue
        existing = api_data["products"].get(product_id, {})
        category = product.get("Category") or product.get("CategoryName")
        if not category:
            cat_ids = product.get("CategoryIds") or []
            if isinstance(cat_ids, list) and len(cat_ids) > 0:
                category = str(cat_ids[0])
            else:
                category = existing.get("category")
        if "IsDigital" in product:
            prod_type = "digital" if product.get("IsDigital") else "physical"
        else:
            prod_type = existing.get("type")
        trader_id = product.get("SellerId") or product.get("TraderId") or existing.get("trader_id")
        api_data["products"][product_id] = {
            "product_id": product_id,
            "name": product.get("Name", existing.get("name", "")),
            "price": product.get("Price", existing.get("price")),
            "category": category,
            "type": prod_type,
            "trader_id": trader_id,
            "description": product.get("Description", existing.get("description")),
            "stock_quantity": product.get("StockQuantity", existing.get("stock_quantity")),
            "image_url": product.get("ImageUrl", existing.get("image_url")),
            "discount_percentage": product.get("DiscountPercentage", existing.get("discount_percentage")),
            "is_available": product.get("IsAvailable", existing.get("is_available")),
        }
