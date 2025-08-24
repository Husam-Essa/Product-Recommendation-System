from dataclasses import dataclass
from typing import Optional, Any

@dataclass
class Behavior:
    user_id: int
    product_id: int
    behavior_type: str
    behavior_time: Any
    gender: Optional[Any] = None
    product_name: Optional[str] = None
    category: Optional[Any] = None
    type: Optional[Any] = None
    price: Optional[float] = None
    trader_id: Optional[int] = None
    shop_id: Optional[int] = None

@dataclass
class Product:
    product_id: int
    name: Optional[str] = None
    price: Optional[float] = None
    category: Optional[Any] = None
    type: Optional[Any] = None
    trader_id: Optional[int] = None
    description: Optional[str] = None
    stock_quantity: Optional[int] = None
    image_url: Optional[str] = None
    discount_percentage: Optional[float] = None
    is_available: Optional[bool] = None

@dataclass
class User:
    user_id: int
    gender: Optional[Any] = None
