from fastapi import APIRouter, Depends
import asyncpg
from models.search import *
from controllers.search_controller import SearchController
from controllers.stream_controller import StreamController
from database.connection import get_db
from fastapi.responses import StreamingResponse
from database.queries import perform_similarity_search


router = APIRouter()

@router.post("/blog/similar", response_model=SearchResponse)
async def find_similar(
    query_data: Query,
    db: asyncpg.Connection = Depends(get_db)
):
    context = await SearchController.find_similar(query_data.query, db)
    return SearchResponse(
        message="Similar blogs found",
        results=context
    )

@router.post("/blog/ai-response")
async def generate_ai_response(
    query_data: Query,
    db: asyncpg.Connection = Depends(get_db)
):
    return await SearchController.generate_ai_response(query_data.query, db)

@router.post("/blog/ai-streaming-response")
async def generate_ai_streaming_response(
    query_data: Query,
    db: asyncpg.Connection = Depends(get_db)
):
    # Do all DB/embedding work here
    query_embedding = await StreamController.generate_embedding(query_data.query)
    vector_string = f"[{','.join(map(str, query_embedding))}]"
    context = await perform_similarity_search(db, vector_string, 5)
    return StreamingResponse(
        StreamController.openai_stream(query_data.query, context),
        media_type="text/markdown",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # For nginx
        }
    )

@router.post("/blog/related-question", response_model=RelatedQuestionResponse)
async def generate_related_question(
    request: RelatedQuestionRequest
):
    related_questions = await SearchController.generate_related_question(request.question, request.context)
    return RelatedQuestionResponse(related_questions=related_questions)

@router.post("/blog/recommend-product-blog", response_model=RecommendProductBlogResponse)
async def recommend_product_blog(
    request: RecommendProductBlogRequest,
    db: asyncpg.Connection = Depends(get_db)
):
    result = await SearchController.recommend_product_blog(request.query, request.context, db)
    return RecommendProductBlogResponse(**result)


