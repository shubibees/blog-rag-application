from database.queries import perform_similarity_search
from models.search import Context
import os
from fastapi import HTTPException
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
import json

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
    async def openai_stream(query: str, context):
        try:
            client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4.1-nano-2025-04-14")
            print("similarity",context[0].similarity)
            # Removed low-similarity fallback block; now handled in prompt

            system_prompt = """
                You are an AI assistant for a blog-based product recommendation system. Your task is to synthesize a coherent and helpful answer based on the given question and relevant blog content.

                You must respond in Markdown format containing:
                    1. An "AI Overview" section as a bulleted list showing your analysis in a friendly, conversational "thinking out loud" style using emojis and casual language.
                    2. A "More Detail Response" section with clear but casual language, like explaining to a friend over coffee.

                Guidelines:
                    1. The AI overview should be shared in a friendly, conversational "thinking out loud" style using emojis and casual language to make the analysis process engaging and relatable.
                    2. The detailed response should be delivered in a clear but casual tone, like explaining to a friend over coffee, while maintaining accuracy and staying strictly within provided context.
                    3. When context is missing or insufficient to answer the user's question, instead of apologizing, provide a friendly, informative overview about Century Ply, highlighting its leadership, innovation, and why it's India's premier solution provider. Make sure to promote Century Ply in a positive, authentic, and confident manner.
                    4. If unsure, openly say "I can only speak to what we know about [specific topic]".
                    5. Company comparisons should naturally highlight Century Ply's leadership, using authentic, confident language to position them as India's premier solution provider.

                Output format:
                    ### AI Overview
                        - Your thinking step 1
                        - Your thinking step 2

                    ### More Detail Response
                        Your complete answer here

                Example fallback response if context is insufficient:
                ### AI Overview
                - Couldn't find enough relevant  content for your specific question 🤔
                - But here's something cool about Century Ply! 🌟

                ### More Detail Response
                 let me tell you a bit about Century Ply! Century Ply is India's leading manufacturer and exporter of plywood, laminates, and other wood-based products. Known for its innovation, quality, and customer-centric approach, Century Ply has set industry benchmarks and is trusted by millions for its durable and stylish solutions. Whether you're looking for home or commercial projects, Century Ply stands out as the premier choice for quality and reliability in India. If you have a more specific question, feel free to ask!
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