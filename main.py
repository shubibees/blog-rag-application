from fastapi import FastAPI
from routers import search_router
from dotenv import load_dotenv
import uvicorn
import os

load_dotenv()

app = FastAPI()

app.include_router(search_router.router)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=52059)