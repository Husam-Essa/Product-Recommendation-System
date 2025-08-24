import httpx

class RailwayClient:
    def __init__(self, base_url: str = "https://tsecommerceapi-production.up.railway.app"):
        self.base_url = base_url

    async def get_user_actions(self, user_id: int):
        url = f"{self.base_url}/api/CustomerAction/{user_id}/actions"
        async with httpx.AsyncClient() as client:
            r = await client.get(url)
            r.raise_for_status()
            return r.json()

    async def get_products(self, page_size: int = 200, page_number: int = 1):
        url = f"{self.base_url}/api/Products/GetAllProducts?pageNumber={page_number}&pageSize={page_size}"
        async with httpx.AsyncClient() as client:
            r = await client.get(url)
            r.raise_for_status()
            return r.json()
