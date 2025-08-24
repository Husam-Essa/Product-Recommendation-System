import nest_asyncio, uvicorn
from src.presentation.api.app import app

nest_asyncio.apply()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
