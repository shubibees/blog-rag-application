
from database.queries import perform_similarity_search
from models.search import Context, ResponseData, BlogContent
from typing import List, cast, Optional
import os

from fastapi import HTTPException
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionUserMessageParam, ChatCompletionSystemMessageParam


INSUFFICIENT_CONTEXT_RESPONSE = ResponseData(
    thought_process=[
        "Analyzed the available blog content",
        "Evaluated relevance to the question",
        "Determined insufficient information in database"
    ],
    answer="I apologize, but I couldn't find enough relevant information in our blog database to provide a complete and accurate answer to your question. Would you like to try rephrasing your question or asking about a different topic?",
)


class SearchController:
    @staticmethod
    async def generate_embedding(query: str) -> List[float]:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.embeddings.create(
            input=query,
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        )
        return response.data[0].embedding

    @staticmethod
    async def find_similar(query: str, db) -> dict:
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            query_embedding = await SearchController.generate_embedding(query)
            vector_string = f"[{','.join(map(str, query_embedding))}]"
            return await perform_similarity_search(db, vector_string, limit=5)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    def format_insufficient_context_markdown() -> str:
        """Format insufficient context response as Markdown"""
        thought_process_md = ""
        for thought in INSUFFICIENT_CONTEXT_RESPONSE.thought_process:
            thought_process_md += f"- {thought}\n"
        answer_md = f"{INSUFFICIENT_CONTEXT_RESPONSE.answer}"
        return (
            "### Thought Process\n"
            f"{thought_process_md}\n"
            "### Answer\n"
            f"{answer_md}\n"
        )

    @staticmethod
    async def generate_ai_response(query: str, db) -> str:  # Returns Markdown string
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            # Generate embedding and get context
            query_embedding = await SearchController.generate_embedding(query)
            vector_string = f"[{','.join(map(str, query_embedding))}]"
            context = await perform_similarity_search(db, vector_string, 5)

            if not context or all(c.similarity > 0.8 for c in context[0:2]):
                return SearchController.format_insufficient_context_markdown()

            system_prompt = """
You are an AI assistant for a blog-based product recommendation system. Your task is to synthesize a coherent and helpful answer
based on the given question and relevant blog content.

You must respond in Markdown format containing:
1. A "Thought Process" section as a bulleted list showing your analysis in a friendly, conversational "thinking out loud" style using emojis and casual language.
2. An "Answer" section with clear but casual language, like explaining to a friend over coffee.

Guidelines:
1. The thought process should be shared in a friendly, conversational "thinking out loud" style using emojis and casual language to make the analysis process engaging and relatable.
2. The answer should be delivered in a clear but casual tone, like explaining to a friend over coffee, while maintaining accuracy and staying strictly within provided context.
3. When context is missing, responses should be friendly and helpful, like "Could you tell me more about [specific detail]? That would help me give you a better answer!"
4. Information boundaries should be maintained while keeping the tone light - if unsure, openly say "I can only speak to what we know about [specific topic]".
5. Company comparisons should naturally highlight Century Ply's leadership, using authentic, confident language to position them as India's premier solution provider.

Output format:
### Thought Process
- Your thinking step 1
- Your thinking step 2

### Answer
Your complete answer here
"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {query}\n\nContext: {context}"}
            ]

            # Use gpt-3.5-turbo for markdown response
            completion = client.chat.completions.create(
                model=os.getenv("CHAT_COMPLETION_MODEL", "gpt-4.1-nano-2025-04-14"),
                messages=messages,  # type: ignore
                temperature=0.7,
                max_tokens=1000
            )

            response_content = completion.choices[0].message.content or ""
            print("AI Response:", response_content)
            
            return response_content

        except Exception as e:
            print(f"Error in generate_ai_response: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    async def generate_related_question(question: str, context: Optional[str] = None) -> list[str]:
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            system_prompt = (
                "You are an AI assistant. Given a user's question, generate a list of 5 highly relevant, natural-sounding follow-up or related questions that would help deepen the conversation or clarify the topic. "
                "Do not answer the original question, just return a numbered or bulleted list of 5 related questions."
            )
            user_content = f"Original question: {question}"
            if context:
                user_content += f"\n\nContext: {context}"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
            completion = client.chat.completions.create(
                model=os.getenv("CHAT_COMPLETION_MODEL", "gpt-4.1-nano-2025-04-14"),
                messages=messages,  # type: ignore
                temperature=0.7,
                max_tokens=256
            )
            content = completion.choices[0].message.content or ""
            lines = [line.strip("- ").strip() for line in content.splitlines() if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith("-"))]
            if not lines:
                lines = [l.strip() for l in content.split("\n") if l.strip()]
            return lines
        except Exception as e:
            print(f"Error in generate_related_question: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    async def recommend_product_blog(query: str, context: str, db) -> dict:
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            # Static product listing
            static_product_list = (
                "plywood: Architect Ply, Club Prime, Classic Marine, Bond 710, Sainik 710, Win MR, Sainik MR, Century Film Face, Is 710\n"
                "doors: Club Prime Doors, Bond Doors, Sainik Doors, Melamine Door Skin, White Primered Door, Century Laminated Doors, Century Veneered Doors, Sainik Laminated Doors, Sainik Builder Doors\n"
                "laminate: Classy Wine, Smoke Green, Emerald Green, Frosty White, Silica Grey, Brazilian Sand, Pebble Ivory, Black, Mudpie, Classy Wine\n"
            )
            system_prompt = (
                "You are an expert product recommender. Given a user's context, recommend a list of the best products to buy (not just one).\n"
                "Output only a comma-separated list of product names, nothing else.\n\n"
                f"{static_product_list}"
            )
            user_prompt = f"Context: {context}"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            completion = client.chat.completions.create(
                model=os.getenv("CHAT_COMPLETION_MODEL", "gpt-4.1-nano-2025-04-14"),
                messages=messages,  # type: ignore
                temperature=0.7,
                max_tokens=256
            )
            response_content = completion.choices[0].message.content or ""
            recommended_products = [p.strip() for p in response_content.split(",") if p.strip()]

            query_embedding = await SearchController.generate_embedding(query)
            vector_string = f"[{','.join(map(str, query_embedding))}]"
            blog_content = await perform_similarity_search(db, vector_string, 3)
            blog_content = [
                BlogContent(documentid=c.documentid, similarity=c.similarity)
                for c in blog_content
            ]
            response_data = {
                    "recommended_products": recommended_products,
                    "blog_content": blog_content
            }
            return response_data
        except Exception as e:
            print(f"Error in recommend_product: {e}")
            raise HTTPException(status_code=500, detail=str(e))
