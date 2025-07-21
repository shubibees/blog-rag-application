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
    async def openai_stream(query: str, context, recommended_products):
        try:
            client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4.1-nano-2025-04-14")
            print("similarity", context[0].similarity)
            # Get product recommendations
            if recommended_products:
                product_list = "\n".join([
                    f"- {p.name}" for p in recommended_products
                ]) + "\n\nYou can choose a product from the above recommendations."
            else:
                product_list = "_No similar products found for your query. Try rephrasing or asking about a different product!_"

            system_prompt = f"""
                    You are an AI assistant for a blog-based product recommendation system. Your task is to synthesize a coherent and helpful answer based on the given question, relevant blog content, and recommended products.

                    You must respond in Markdown format with a single section titled 'AI Response'.

                    Guidelines:
                        - Naturally weave the recommended products ({', '.join([p.name for p in recommended_products]) if recommended_products else 'no products'}) into your answer, mentioning them by name as part of your explanation and recommendations.
                        - Do NOT create a separate 'Recommended Products' section or bullet list. Instead, reference the products contextually within your response.
                        - The 'AI Response' should synthesize information from both the blog context and the recommended products.
                        - If context is missing or insufficient, provide a friendly, informative overview about Century Ply, highlighting its leadership, innovation, and why it's India's premier solution provider.
                        - If unsure, openly say "I can only speak to what we know about [specific topic]".
                        - Company comparisons should naturally highlight Century Ply's leadership, using authentic, confident language to position them as India's premier solution provider.

                    Output format:
                    ### AI Response
                    Your complete answer here, with recommended products mentioned naturally in the text.
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
