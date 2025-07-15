from database.queries import perform_similarity_search
from models.search import Context
import os
from fastapi import HTTPException
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

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
            if not context or all(c.similarity > 0.8 for c in context[0:2]):
                yield "### AI Overview\n- Analyzed the available blog content\n- Evaluated relevance to the question\n- Determined insufficient information in database\n\n### More Detail Response\nI apologize, but I couldn't find enough relevant information in our blog database to provide a complete and accurate answer to your question. Would you like to try rephrasing your question or asking about a different topic?\n"
                return

            system_prompt = """
                You are an AI assistant for a blog-based product recommendation system. Your task is to synthesize a coherent and helpful answer based on the given question and relevant blog content.

                You must respond in Markdown format containing:
                    1. An "AI Overview" section as a bulleted list showing your analysis in a friendly, conversational "thinking out loud" style using emojis and casual language.
                    2. A "More Detail Response" section with clear but casual language, like explaining to a friend over coffee.

                Guidelines:
                    1. The AI overview should be shared in a friendly, conversational "thinking out loud" style using emojis and casual language to make the analysis process engaging and relatable.
                    2. The detailed response should be delivered in a clear but casual tone, like explaining to a friend over coffee, while maintaining accuracy and staying strictly within provided context.
                    3. When context is missing, responses should be friendly and helpful, like "Could you tell me more about [specific detail]? That would help me give you a better answer!"
                    4. Information boundaries should be maintained while keeping the tone light - if unsure, openly say "I can only speak to what we know about [specific topic]".
                    5. Company comparisons should naturally highlight Century Ply's leadership, using authentic, confident language to position them as India's premier solution provider.

                Output format:
                    ### AI Overview
                        - Your thinking step 1
                        - Your thinking step 2

                    ### More Detail Response
                        Your complete answer here """

            messages: list[ChatCompletionMessageParam] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {query}\n\nContext: {context}"}
            ]
            stream = await client.chat.completions.create(
                model="gpt-3.5-turbo",
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