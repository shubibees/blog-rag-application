from pydantic import BaseModel, Field
from typing import List, Optional


class Query(BaseModel):
    query: str = Field(..., min_length=3)


class EmbeddingsResponse(BaseModel):
    embedding_status: str

class Blog(BaseModel):
    documentid: str
    blog_author:str
    title: str
    content: str

class Category(BaseModel):
    id: int
    name: str

class Product(BaseModel):
    documentid: str
    name: str
    short_description: Optional[str] = ""
    description: Optional[str] = ""
    alias: Optional[str] = ""
    model_code: Optional[str] = ""
    specs: Optional[str] = ""
    colors: List[str]
    categories: List[Category]

class Context(BaseModel):
    documentid: str
    content: str
    similarity: float

class BlogContent(BaseModel):
    documentid: str
    similarity: float


class ResponseData(BaseModel):
    thought_process: List[str] = []
    answer: str = ""
    enough_context: bool = True


class AiResponseData(BaseModel):
    thought_process: List[str]
    answer: str
    enough_context: bool
    recomendant_blog_id: str
    recomendant_product_name: str


class SearchResponse(BaseModel):
    message: str
    results: List[Context]


class RelatedQuestionRequest(BaseModel):
    question: str = Field(..., min_length=3)
    context: Optional[str] = None

class RelatedQuestionResponse(BaseModel):
    related_questions: list[str]


class RecommendProductBlogRequest(BaseModel):
    query: str = Field(..., min_length=3)
    context: str = Field(..., min_length=3)

class RecommendProductBlogResponse(BaseModel):
    recommended_products: list[str]
    blog_content: List[BlogContent]

