# app/database/connection.py
import os
import asyncpg
from fastapi import HTTPException
import logging
from typing import AsyncGenerator
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

async def ensure_embedding_tables_exist(conn):
    # Create blog_embedding_oai_small table if it does not exist
    await conn.execute('''
        CREATE TABLE IF NOT EXISTS blog_embedding_oai_small (
            documentid TEXT PRIMARY KEY,
            title TEXT,
            embeddingContext TEXT,
            embedding VECTOR(1536)
        )
    ''')
    # Create product_embedding_oai_small table if it does not exist
    await conn.execute('''
        CREATE TABLE IF NOT EXISTS product_embedding_oai_small (
            documentid TEXT PRIMARY KEY,
            name TEXT,
            alias TEXT,
            embeddingContext TEXT,
            embedding VECTOR(1536)
        )
    ''')

async def get_db() -> AsyncGenerator[asyncpg.Connection, None]:
    try:
        load_dotenv()
        conn = await asyncpg.connect(
            host=os.getenv("DB_HOST", "localhost"),
            database=os.getenv("DB_NAME", "your_database_name"),
            user=os.getenv("DB_USER", "your_username"),
            password=os.getenv("DB_PASSWORD", "your_password"),
            port=int(os.getenv("DB_PORT", 5432))
        )
        logger.info("Database connection established")
        await ensure_embedding_tables_exist(conn)
        yield conn
    except asyncpg.PostgresError as e:
        logger.error(f"Database connection error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Could not connect to database"
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )
    finally:
        if 'conn' in locals():
            await conn.close()
            logger.info("Database connection closed")