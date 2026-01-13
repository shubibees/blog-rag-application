from fastapi import FastAPI
from routers.similarity_router import similarity_router
from routers.ai_router import ai_router
from routers.database_operation import database_operation_router
from dotenv import load_dotenv
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()

    # Add CORS middleware to allow all origins, methods, and headers
app.add_middleware(
    CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
)
app.include_router(database_operation_router, prefix="/database")
app.include_router(similarity_router, prefix="/similarity")
app.include_router(ai_router, prefix="/ai")
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=52059)

