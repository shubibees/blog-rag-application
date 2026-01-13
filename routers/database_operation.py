from fastapi import APIRouter, Depends
import asyncpg
from controllers.embedding_generator import create_embedding_of_blog_in_database, create_embedding_of_product_in_database
from database.connection import get_db
from models.search import EmbeddingsResponse

database_operation_router = APIRouter()

@database_operation_router.get("/blog/embeddings",response_model=EmbeddingsResponse)
async def generate_embeddings_for_blog(
    db: asyncpg.Connection = Depends(get_db)
):
    return await create_embedding_of_blog_in_database(db)

@database_operation_router.get("/product/embeddings",response_model=EmbeddingsResponse)
async def generate_embeddings_for_product(
    db: asyncpg.Connection = Depends(get_db)
):
    return await create_embedding_of_product_in_database(db)

@database_operation_router.get("/connection/test")
async def test_connection(
    db: asyncpg.Connection = Depends(get_db)
):
    return await db.fetch("SELECT * FROM blogs where published_at is not null")


