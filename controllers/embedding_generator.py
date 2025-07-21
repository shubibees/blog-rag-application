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


BATCH_SIZE = 10
def truncate_to_n_words(text, n):
    if not text:
        return ""
    words = text.split()
    return " ".join(words[:n])

async def create_embedding_of_blog_in_database(db):
    #delete the table data 
    #await db.execute("DELETE FROM blog_embedding_oai_small")
    #new embedding
    blogs = await get_all_blogs(db)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    for i in range(0, len(blogs), BATCH_SIZE):
        batch = blogs[i:i+BATCH_SIZE]
        embedding_contexts = [
            f"Blog Author: {blog.blog_author or ''}\nBlog Title: {blog.title or ''}\nBlog Content: {truncate_to_n_words(blog.content,400)}"
            for blog in batch
        ]
        response = client.embeddings.create(
            input=embedding_contexts,
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        )
        embeddings = [item.embedding for item in response.data]
        print("embedding length",len(embeddings))
        for blog, embedding, context in zip(batch, embeddings, embedding_contexts):
            embedding_str = str(embedding)
            await db.execute(
                "INSERT INTO blog_embedding_oai_small (documentid,title,content, embedding) VALUES ($1, $2, $3, $4::vector)",
                blog.documentid, blog.title, context, embedding_str
            )
    return {"embedding_status": "success"}



async def create_embedding_of_product_in_database(db):
    #delete all the data
    #await db.execute("DELETE FROM product_embedding_oai_small")
    #get all the product from database
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
        category_names = ", ".join([cat["name"] if isinstance(cat, dict) and "name" in cat else getattr(cat,"name",str(cat)) for cat in category_list])


        embedding_context = (
                 f" Product Name: {product.name}\n"
                 f" Color Focus (parsed from name): (infer_color_from_name{product.name})\n"
                 f" Product Colors: {color_codes or 'Not specified'}\n"
                 f" This product is mainly known for its color: (infer_color_from_name{product.name})\n"
                 f" Categories (human-readable): {category_names}\n"
                 f" This product belongs to the following types/categories: {category_names}\n"
                 f" Product Alias: {product.alias}\n"
                 f" Model Code: {product.model_code}\n"
                 f" Short Description: {product.short_description or ''}\n"
                 f"Description: {product.description or ''}\n"
                 f" Specs: {product.specs or ''}"
        )

        print("embedding_context",embedding_context)
        embedding = await generate_embedding(embedding_context)
        print("embedding",len(embedding),type(embedding),embedding[:5])
        embedding_str = f"[{','.join(map(str,embedding))}]"
        await db.execute(
                "INSERT INTO product_embedding_oai_small (documentid, name, alias, embeddingContext, embedding) VALUES ($1, $2, $3, $4, $5::vector)",
            product.documentid, product.name, product.alias, embedding_context, embedding_str
        )
    return {"embedding_status": "success"}
