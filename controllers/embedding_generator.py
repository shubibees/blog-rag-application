from database.queries import get_all_blogs, get_all_products
from openai import OpenAI
from typing import List
import os
import json
import logging

async def generate_embedding(query: str) -> List[float]:
   client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
   response = client.embeddings.create(
       input=query,
       model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
   )
   return response.data[0].embedding

async def create_embedding_of_blog_in_database(db):
    # get all blogs from database
    blogs = await get_all_blogs(db)
    # let create a loop to generate embeddings for each blog and store in the database
    for blog in blogs:
        print("documentid",blog.documentid,"embeding generated")
        embedding_context = f"Blog Author: {blog.blog_author}\nBlog Title: {blog.title}\nBlog Content: {blog.content}"
        embedding = await generate_embedding(embedding_context)
        await db.execute("INSERT INTO blog_embedding_oai_small (documentid,title,embeddingContext, embedding) VALUES ($1, $2, $3, $4)", blog.documentid, blog.title, embedding_context, embedding)
    return {"embedding_status": "success"}


async def create_embedding_of_product_in_database(db):
    products = await get_all_products(db)
    for product in products:
        print("documentid", product.documentid, "embeding generated")
        # Parse colors
        if isinstance(product.colors, str):
            try:
                color_list = json.loads(product.colors)
            except Exception as e:
                logging.warning(f"Failed to parse colors for product {product.documentid}: {e}")
                color_list = []
        else:
            color_list = product.colors if product.colors is not None else []
        color_codes = ", ".join(color_list)

        # Parse categories
        if isinstance(product.categories, str):
            try:
                category_list = json.loads(product.categories)
            except Exception as e:
                logging.warning(f"Failed to parse categories for product {product.documentid}: {e}")
                category_list = []
        else:
            category_list = product.categories if product.categories is not None else []
        category_names = ", ".join([cat["name"] if isinstance(cat, dict) and "name" in cat else str(cat) for cat in category_list])

        embedding_context = (
            f"Product Name: {product.name}\n"
            f"Product Short Description: {product.short_description}\n"
            f"Product Description: {product.description}\n"
            f"Product Alias: {product.alias}\n"
            f"Product Model Code: {product.model_code}\n"
            f"Product Specs: {product.specs}\n"
            f"Product Colors: {color_codes}\n"
            f"Product Categories: {category_names}"
        )
        embedding = await generate_embedding(embedding_context)
        # Convert embedding to string if needed for DB compatibility
        embedding_str = str(embedding)
        await db.execute(
            "INSERT INTO product_embedding_oai_small (documentid, name, alias, embeddingContext, embedding) VALUES ($1, $2, $3, $4, $5)",
            product.documentid, product.name, product.alias, embedding_context, embedding_str
        )
    return {"embedding_status": "success"}