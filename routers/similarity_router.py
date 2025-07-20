from fastapi import APIRouter, Depends
import asyncpg
from models.search import Query, SearchResponse, ProductSimilarityResult
from controllers.search_controller import SearchController
from database.connection import get_db

similarity_router = APIRouter()

@similarity_router.post("/blog/similar", response_model=SearchResponse)
async def find_similar(
    query_data: Query,
    db: asyncpg.Connection = Depends(get_db)
):
    context = await SearchController.find_similar(query_data.query, db)
    return SearchResponse(
        message="Similar blogs found",
        results=context
    )

@similarity_router.post("/product/similar", response_model=list[ProductSimilarityResult])
async def find_similar_product(
    query_data: Query,
    db: asyncpg.Connection = Depends(get_db)
):
    return await SearchController.find_similar_product(query_data.query, db) 
