from openai import OpenAI
import os
import json
import re

async def preprocess_query_with_openai(query: str) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = (
        "Given the following user query, extract the color (if any) and the product category like laminate ,sanik laminate ,doors,new age prodcut ,centruyexteria .. (if any). "
        "Emphasize and clearly state the color and category in your response. "
        "If the query is ambiguous, briefly clarify or expand it for a product search. "
        "Always include lines like: color: \"<color>\" and category: \"<category>\" in your response, but you may include other helpful text as well.\n\n"
        f"User query: \"{query}\""
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant for query preprocessing."},
        {"role": "user", "content": prompt}
    ]
    completion = client.chat.completions.create(
        model=os.getenv("CHAT_COMPLETION_MODEL", "gpt-3.5-turbo"),
        messages=messages,
        temperature=0,
        max_tokens=256
    )
    content = completion.choices[0].message.content.strip()
    # Extract color and category from anywhere in the response
    return content 