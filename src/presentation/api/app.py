from fastapi import FastAPI
from .routes.recommend import router as reco_router

app = FastAPI(title="Reco API")
app.include_router(reco_router)
