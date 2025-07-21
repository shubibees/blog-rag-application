from openai import OpenAI
import os
import json

async def preprocess_query_with_openai(query: str) -> dict:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = (
        "Given the following user query, extract the color (if any) and the product category (if any). "
        "If the query is ambiguous, try to make it more descriptive for a product search. "
        "Return a JSON object with keys: 'color', 'category', and 'expanded_query'.\n\n"
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
    import json
    content = completion.choices[0].message.content
    try:
        return json.loads(content)
    except Exception:
        # fallback: just return the original query
        return {"color": None, "category": None, "expanded_query": query}