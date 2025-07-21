from fastapi import APIRouter, Depends
import asyncpg
from models.search import Query, RelatedQuestionRequest, RelatedQuestionResponse, RecommendProductBlogRequest, RecommendProductBlogResponse
from controllers.search_controller import SearchController
from controllers.stream_controller import StreamController
from database.connection import get_db
from fastapi.responses import StreamingResponse
from database.queries import perform_blogs_similarity_search
from helpers.helper import preprocess_query_with_openai
ai_router = APIRouter()

@ai_router.post("/ai-response")
async def generate_ai_response(
    query_data: Query,
    db: asyncpg.Connection = Depends(get_db)
):
    return await SearchController.generate_ai_response(query_data.query, db)

@ai_router.post("/ai-streaming-response")
async def generate_ai_streaming_response(
    query_data: Query,
    db: asyncpg.Connection = Depends(get_db)
):
    preprocessed_query = await preprocess_query_with_openai(query_data.query)
    expanded_query = preprocessed_query["expanded_query"]
    query_embedding = await StreamController.generate_embedding(expanded_query)
    vector_string = f"[{','.join(map(str, query_embedding))}]"
    context = await perform_blogs_similarity_search(db, vector_string, 2)
    print("context",context)
    recommended_products = await SearchController.find_similar_product(query_data.query, db)
    return StreamingResponse(
        StreamController.openai_stream(query_data.query, context, recommended_products),
        media_type="text/markdown",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@ai_router.post("/related-question", response_model=RelatedQuestionResponse)
async def generate_related_question(
    request: RelatedQuestionRequest
):
    related_questions = await SearchController.generate_related_question(request.question, request.context)
    return RelatedQuestionResponse(related_questions=related_questions)

@ai_router.post("/recommend-similar-products-and-blogs", response_model=RecommendProductBlogResponse)
async def recommend_similar_products_and_blogs(
    request: RecommendProductBlogRequest,
    db: asyncpg.Connection = Depends(get_db)
):
    preprocessed_query = await preprocess_query_with_openai(request.query)
    expanded_query = preprocessed_query["expanded_query"]
    result = await SearchController.recommend_similar_products_and_blogs(expanded_query, db)
    return RecommendProductBlogResponse(**result) 
