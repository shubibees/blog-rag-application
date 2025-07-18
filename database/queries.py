from typing import List

import asyncpg
from models.search import Context, Blog, Product


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


async def get_all_blogs(conn: asyncpg.Connection) -> List[Blog]:
    rows = await conn.fetch("SELECT * FROM blogs where published_at is not null")
    return [
        Blog(
            documentid=row['document_id'],
            title=row['title'],
            content=row['content']
        ) for row in rows
    ]


async def get_all_products(conn: asyncpg.Connection) -> List[Product]:
    rows = await conn.fetch("""SELECT
        p.id,
        p.name,
        p.short_description,
        p.description,
        p.alias,
        p.model_code,
        p.specs,
        COALESCE(
            JSON_AGG(DISTINCT pc.color) FILTER (WHERE pc.color IS NOT NULL),
            '[]'
        ) AS colors,
        COALESCE(
            JSON_AGG(
                DISTINCT jsonb_build_object(
                    'id', cat.id,
                    'name', cat.name
                )
            ) FILTER (WHERE cat.id IS NOT NULL),
            '[]'
        ) AS categories
    FROM
        products p
    LEFT JOIN
        products_colors_lnk pcl
    ON
        p.id = pcl.product_id
    LEFT JOIN
        product_colors pc
    ON
        pcl.product_color_id = pc.id
    LEFT JOIN
        product_categories_products_lnk pcpl
    ON
        p.id = pcpl.product_id
    LEFT JOIN
        product_categories cat
    ON
        pcpl.product_category_id = cat.id
    WHERE
        p.published_at IS NOT NULL
    GROUP BY
        p.id
    ORDER BY
        p.id ASC""")
    return [
        Product(
            documentid=row['document_id'],
            name=row['name'],
            short_description=row['short_description'],
            description=row['description'],
            alias=row['alias'],
            model_code=row['model_code'],
            specs=row['specs'],
            colors=row['colors'],
            categories=row['categories']
        ) for row in rows
    ]