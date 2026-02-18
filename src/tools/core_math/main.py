from routers.router_kmeans import router as kmeans_router

from fastapi import FastAPI

app = FastAPI()

app.include_router(kmeans_router)


