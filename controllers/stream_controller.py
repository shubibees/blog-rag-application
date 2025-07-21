from database.queries import perform_blogs_similarity_search
from models.search import Context
import os
from fastapi import HTTPException
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
import json
from controllers.search_controller import SearchController
class StreamController:
    @staticmethod
    async def generate_embedding(query: str) -> list[float]:
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = await client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding

    @staticmethod
    async def openai_stream(query: str, context, db):
        try:
            client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4.1-nano-2025-04-14")
            print("similarity", context[0].similarity)
            # Get product recommendations
            recommended_products = await SearchController.find_similar_product(query, db)
            if recommended_products:
                product_list = "\n".join([
                    f"- {p.name}" for p in recommended_products
                ]) + "\n\nYou can choose a product from the above recommendations."
            else:
                product_list = "_No similar products found for your query. Try rephrasing or asking about a different product!_"

            system_prompt = f"""
                    You are an AI assistant for a blog-based product recommendation system. Your task is to synthesize a coherent and helpful answer based on the given question, relevant blog content, and recommended products.

                    You must respond in Markdown format containing:
                        1. A section titled 'Recommended Products' showing the most relevant products for the user's query (see below).
                        2. A section titled 'AI Response' with a clear, casual, and helpful answer based on the blog context and the recommended products, like explaining to a friend over coffee.

                    Guidelines:
                        - The 'Recommended Products' section should be a bullet list of product names. At the end, add: 'You can choose a product from the above recommendations.'
                        - The 'AI Response' should synthesize information from both the blog context and the recommended products.
                        - If context is missing or insufficient, provide a friendly, informative overview about Century Ply, highlighting its leadership, innovation, and why it's India's premier solution provider.
                        - If unsure, openly say "I can only speak to what we know about [specific topic]".
                        - Company comparisons should naturally highlight Century Ply's leadership, using authentic, confident language to position them as India's premier solution provider.

                    Output format:
                    ### AI Response
                    Your complete answer here

                    ### Recommended Products
                    {product_list}
                    """

            messages: list[ChatCompletionMessageParam] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {query}\n\nContext: {context}"}
            ]
            stream = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=1000,
                stream=True
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            print(f"StreamController error: {e}")
            yield f"Error: {str(e)}"
