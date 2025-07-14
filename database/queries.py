from typing import List

import asyncpg
from models.search import Context


async def perform_similarity_search(conn: asyncpg.Connection, vector_string: str, limit: int) -> List[Context]:
    rows = await conn.fetch("""
        SELECT documentid, content, embedding <=> $1::vector AS similarity
        FROM blog_embedding_oai_small
        ORDER BY similarity ASC
        LIMIT $2
    """, vector_string, limit)

    return [
        Context(
            documentid=row['documentid'],
            content=row['content'],
            similarity=row['similarity']
        ) for row in rows
    ]
