from openai import OpenAI
import os
import json

from pydantic import BaseModel, Field
from typing import List, Optional

class returnModel(BaseModel):
    color: Optional[str] = None
    color_hex: Optional[str] = None
    category: Optional[str] = None

async def preprocess_query_with_openai(query: str) -> returnModel:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    function_schema = [
        {
            "type": "function",
            "function": {
                "name": "extract_color_and_category",
                "description": (
                    "Extracts the human-readable color and product category from a user query. "
                    "Also provides the color as a hex code if possible. "
                    "For example: 'red' -> '#FF0000', 'ivory' -> '#FFFFF0', 'brown' -> '#A52A2A'. "
                    "If the color cannot be converted, leave color_hex empty. "
                    "Examples of categories: plywood, laminate, sanik laminate, century exteria, doors, new age product, etc."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "color_name": {
                            "type": "string",
                            "description": "The human-readable color mentioned in the query, if any (e.g., brown, red, ivory)."
                        },
                        "color_hex": {
                            "type": "string",
                            "description": (
                                "The hex code for the color_name (e.g., #271911 for brown, #FF0000 for red, #FFFFF0 for ivory). "
                                "try to convert color mentioned in the querey to hex code return an empty string."
                            )
                        },
                        "category": {
                            "type": "string",
                            "description": (
                                "The product category mentioned in the query, if any. "
                                "Examples: plywood, laminate, sanik laminate, century exteria, doors, new age product, etc."
                            )
                        }
                    },
                    "required": ["color_name", "color_hex", "category"]
                }
            }
        }
    ]
    messages = [
        {"role": "system", "content": "You are a helpful assistant for query preprocessing."},
        {"role": "user", "content": query}
    ]
    completion = client.chat.completions.create(
        model=os.getenv("CHAT_COMPLETION_MODEL", "gpt-3.5-turbo-0613"),
        messages=messages,
        tools=function_schema,
        tool_choice={"type": "function", "function": {"name": "extract_color_and_category"}},
        temperature=0,
        max_tokens=256
    )
    tool_calls = completion.choices[0].message.tool_calls
    if tool_calls and tool_calls[0].function.name == "extract_color_and_category":
        json_output = json.loads(tool_calls[0].function.arguments)
        print("json_output:", str(json_output))
        return returnModel(
            color=json_output.get("color_name", ""),
            color_hex=json_output.get("color_hex", ""),
            category=json_output.get("category", "")
        )
    else:
        return returnModel(color="", color_hex="", category="") 