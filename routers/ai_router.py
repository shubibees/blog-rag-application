from fastapi import APIRouter, Depends
import asyncpg
from models.search import Query, RelatedQuestionRequest, RelatedQuestionResponse, RecommendProductBlogRequest, RecommendProductBlogResponse
from controllers.search_controller import SearchController
from controllers.stream_controller import StreamController
from database.connection import get_db
from fastapi.responses import StreamingResponse
from database.queries import perform_blogs_similarity_search

ai_router = APIRouter()

@ai_router.post("/blog/ai-response")
async def generate_ai_response(
    query_data: Query,
    db: asyncpg.Connection = Depends(get_db)
):
    return await SearchController.generate_ai_response(query_data.query, db)

@ai_router.post("/blog/ai-streaming-response")
async def generate_ai_streaming_response(
    query_data: Query,
    db: asyncpg.Connection = Depends(get_db)
):
    query_embedding = await StreamController.generate_embedding(query_data.query)
    vector_string = f"[{','.join(map(str, query_embedding))}]"
    context = await perform_blogs_similarity_search(db, vector_string, 5)
    return StreamingResponse(
        StreamController.openai_stream(query_data.query, context, db),
        media_type="text/markdown",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@ai_router.post("/blog/related-question", response_model=RelatedQuestionResponse)
async def generate_related_question(
    request: RelatedQuestionRequest
):
    related_questions = await SearchController.generate_related_question(request.question, request.context)
    return RelatedQuestionResponse(related_questions=related_questions)

@ai_router.post("/blog/recommend-product-blog", response_model=RecommendProductBlogResponse)
async def recommend_product_blog(
    request: RecommendProductBlogRequest,
    db: asyncpg.Connection = Depends(get_db)
):
    result = await SearchController.recommend_product_blog(request.query, request.context, db)
    return RecommendProductBlogResponse(**result) 
