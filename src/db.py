# src/db.py
"""
Database connection management for Supabase and data access functions.

This module provides a reusable SQLAlchemy engine configured to connect
to a Supabase instance. It relies on environment variables for database credentials.
It also includes functions for interacting with trademark-related data,
including storing and searching vector embeddings for goods/services.

Required Environment Variables:
    SUPABASE_URL: The Supabase project URL.
    SUPABASE_KEY: The Supabase API key (service role key recommended for backend).

Usage:
    from src.db import get_engine, store_goods_service_embedding, find_similar_goods_services

    engine = get_engine()
    # ... (Example usage for storing/searching)
"""

import os
from typing import Optional, List, Tuple

import sqlalchemy
from sqlalchemy import select, Index, cast, String # Add index import
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker # Import async components
from sqlalchemy.exc import OperationalError, IntegrityError
from supabase import create_client, Client

# Import the logger
from src.logger import get_logger, info, warning, error, exception

# Import ORM models and embedding function
from src.models import Base, GoodsServiceOrm, VectorEmbeddingOrm
from src.embeddings import generate_embedding

# Load environment variables from .env file for local development
# In Cloud Functions, set these variables in the runtime environment settings.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # dotenv is optional, mainly for local development
    logger.info("dotenv not installed, skipping .env file loading")
    pass


# Module-level variables to store instances for reuse
# within the same Cloud Function instance (improves efficiency)
_async_engine: Optional[sqlalchemy.ext.asyncio.AsyncEngine] = None # Rename for clarity
_supabase: Optional[Client] = None
_async_session_local: Optional[async_sessionmaker[AsyncSession]] = None # Use async_sessionmaker


def get_supabase() -> Client:
    """
    Initializes and returns a Supabase client.

    Creates a Supabase client using the project URL and API key.
    Ensures the client is created only once per Cloud Function instance.

    Raises:
        ValueError: If required environment variables are not set.

    Returns:
        A configured Supabase client instance.
    """
    global _supabase

    # Return existing client if already initialized
    if _supabase:
        return _supabase

    # Get Configuration from Environment Variables
    supabase_url: Optional[str] = os.getenv("SUPABASE_URL")
    # Consider using SUPABASE_SERVICE_KEY for backend operations
    supabase_key: Optional[str] = os.getenv("SUPABASE_KEY")

    if not all([supabase_url, supabase_key]):
        error_msg = "Missing required Supabase environment variables: SUPABASE_URL, SUPABASE_KEY"
        logger.error(error_msg)
        raise ValueError(error_msg)

    try:
        # Create and store the Supabase client
        _supabase = create_client(supabase_url, supabase_key)
        logger.info("Successfully created Supabase client")
        return _supabase
    except Exception as e:
        exception("Failed to create Supabase client", exc=e)
        raise OperationalError(f"Failed to create Supabase client: {e}", params={}, orig=e) from e


async def get_async_engine() -> sqlalchemy.ext.asyncio.AsyncEngine: # Make async (optional, but good practice if init involves IO)
    """
    Initializes and returns an asynchronous SQLAlchemy Engine configured for Supabase.

    Creates an async connection pool using the Supabase connection details with asyncpg.
    Ensures the engine is created only once per Cloud Function instance.

    Raises:
        ValueError: If required environment variables are not set.
        OperationalError: If the database connection fails.

    Returns:
        A configured asynchronous SQLAlchemy Engine instance.
    """
    global _async_engine

    # Return existing engine if already initialized
    if _async_engine:
        return _async_engine

    # Get Configuration from Environment Variables (same as before)
    supabase_url: Optional[str] = os.getenv("SUPABASE_URL")
    db_password: Optional[str] = os.getenv("SUPABASE_SERVICE_KEY")
    db_user: str = "postgres"
    db_host: Optional[str] = None
    db_name: str = "postgres"

    if supabase_url:
        try:
            host_part = supabase_url.split("//")[1]
            db_host = f"db.{host_part}"
        except IndexError:
            error_msg = "Invalid SUPABASE_URL format."
            logger.error(error_msg)
            raise ValueError(error_msg)

    if not all([db_host, db_password]):
        error_msg = "Missing required environment variables for DB engine: SUPABASE_URL, SUPABASE_SERVICE_KEY"
        logger.error(error_msg)
        raise ValueError(error_msg)

    try:
        # Use postgresql+asyncpg dialect
        db_url = f"postgresql+asyncpg://{db_user}:{db_password}@{db_host}:5432/{db_name}"
        logger.info("Initializing async database engine")

        # Use create_async_engine
        async_engine = create_async_engine(
            db_url,
            pool_size=5,
            max_overflow=2,
            pool_timeout=30,
            pool_recycle=1800,
            # Async engines typically don't need json serializers specified here
            # echo=True # Uncomment for debugging SQL
        )

        # Optional: Test connection asynchronously
        async with async_engine.connect() as connection:
            await connection.execute(sqlalchemy.text("SELECT 1"))

        # Store the engine globally for reuse
        _async_engine = async_engine
        logger.info("Async database engine created successfully.")
        return _async_engine
    except OperationalError as e:
        exception("Database connection failed", exc=e)
        raise
    except Exception as e:
        # Ensure the original exception type is preserved if possible
        if isinstance(e, OperationalError):
             raise
        exception("Failed to create async database engine", exc=e)
        raise OperationalError(f"Failed to create async database engine: {e}", params={}, orig=e) from e


async def get_async_session() -> async_sessionmaker[AsyncSession]: # Make async (consistent with get_engine)
    """
    Returns an asynchronous SQLAlchemy sessionmaker instance bound to the async engine.

    Creates the sessionmaker only once.
    """
    global _async_session_local
    if _async_session_local is None:
        engine = await get_async_engine() # Await the async engine getter
        _async_session_local = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False # Recommended for async sessions
        )
        logger.info("Async session maker created.")
    return _async_session_local


# --- Embedding Storage and Search Functions ---

async def store_goods_service_embedding(goods_service_id: int, term: str, session: AsyncSession) -> bool: # Make async, accept AsyncSession
    """
    Generates an embedding for the given term and stores it async in the vector_embeddings table,
    linked to the specified goods_services ID. Skips if embedding already exists.

    Args:
        goods_service_id: The ID of the GoodsServiceOrm record.
        term: The text term of the goods/service to embed.
        session: The asynchronous SQLAlchemy session to use for database operations.

    Returns:
        True if the embedding was successfully generated and stored (or already existed),
        False otherwise.
    """
    try:
        # Check if embedding already exists using async session
        stmt = select(VectorEmbeddingOrm).filter_by(
            entity_type='goods_services',
            entity_id=goods_service_id
        )
        result = await session.execute(stmt)
        existing = result.scalars().first() # Use await and scalars()

        if existing:
            logger.info(f"Embedding already exists for goods_services ID {goods_service_id}")
            return True

        # Generate the embedding (Use the async version now)
        embedding_vector: Optional[List[float]] = await generate_embedding(term)

        if embedding_vector is None:
            logger.warning(f"Failed to generate embedding for term", term=term)
            return False

        # Create and store the new embedding record
        new_embedding = VectorEmbeddingOrm(
            entity_type='goods_services',
            entity_id=goods_service_id,
            embedding=embedding_vector
        )
        session.add(new_embedding)
        await session.flush() # Use await
        logger.info(f"Successfully stored embedding for goods_services ID {goods_service_id}")
        # No explicit commit here, assuming caller manages the transaction boundary
        return True

    except IntegrityError as e:
        # Handle potential race conditions if another process inserted concurrently
        exception("IntegrityError storing embedding", exc=e, goods_service_id=goods_service_id)
        await session.rollback() # Use await
        # Optionally, re-query async to confirm if it exists now
        stmt = select(VectorEmbeddingOrm).filter_by(
            entity_type='goods_services',
            entity_id=goods_service_id
        )
        result = await session.execute(stmt)
        return result.scalars().first() is not None # Check if exists after rollback
    except Exception as e:
        exception("Error storing embedding", exc=e, goods_service_id=goods_service_id)
        await session.rollback() # Use await
        return False


async def find_similar_goods_services( # Make async
    query_embedding: List[float],
    limit: int = 10,
    distance_threshold: float = 0.3, # Add threshold parameter
    session: Optional[AsyncSession] = None, # Accept optional AsyncSession
) -> List[Tuple[GoodsServiceOrm, float]]:
    """
    Finds goods/services terms similar to the provided query embedding asynchronously.

    Performs a vector similarity search (cosine distance) against the
    `vector_embeddings` table where entity_type is 'goods_services'.

    Args:
        query_embedding: The vector embedding to search against.
        limit: The maximum number of similar items to return.
        distance_threshold: Maximum cosine distance to consider (0.0-2.0, lower is more similar).
        session: An optional existing asynchronous SQLAlchemy session. If None, a new one is created and managed.

    Returns:
        A list of tuples, where each tuple contains:
        (GoodsServiceOrm object, similarity_score (cosine distance, lower is better)).
        Returns an empty list if an error occurs or no matches are found.
    """
    # Determine if we need to manage the session locally
    manage_session = session is None
    db_session: AsyncSession

    if manage_session:
        AsyncSessionLocal = await get_async_session() # Await the async factory
        db_session = AsyncSessionLocal() # Create a new session instance
    else:
        # Use the provided session, ensuring it's not None (checked by type hint Optional[AsyncSession])
        db_session = session # type: ignore

    results: List[Tuple[GoodsServiceOrm, float]] = []
    try:
        # Use the <=> operator for cosine distance (provided by pgvector)
        # Query VectorEmbeddingOrm, get distance, join with GoodsServiceOrm
        # Add WHERE clause with distance threshold
        stmt = (
            select( # Use select() from sqlalchemy
                GoodsServiceOrm,
                VectorEmbeddingOrm.embedding.cosine_distance(query_embedding).label("distance")
            )
            .join(VectorEmbeddingOrm,
                  (VectorEmbeddingOrm.entity_id == GoodsServiceOrm.id) &
                  (VectorEmbeddingOrm.entity_type == 'goods_services'))
            .where(VectorEmbeddingOrm.embedding.cosine_distance(query_embedding) <= distance_threshold)
            .order_by(sqlalchemy.asc("distance")) # Use sqlalchemy.asc explicitly
            .limit(limit)
        )

        # Log search parameters
        logger.info("Executing vector similarity search", limit=limit, distance_threshold=distance_threshold)

        # Execute the query asynchronously
        query_result = await db_session.execute(stmt)
        results = query_result.all() # Returns List[Row], Row contains (GoodsServiceOrm, float)
        
        # Log search results
        logger.info(f"Vector search found {len(results)} similar items")

    except Exception as e:
        exception("Error finding similar goods/services", exc=e)
        if manage_session: # Only rollback if we created the session
            await db_session.rollback() # Use await
    finally:
        if manage_session: # Only close if we created the session
            await db_session.close() # Use await

    # Ensure the return type matches the signature
    # query_result.all() gives List[Row], need List[Tuple[GoodsServiceOrm, float]]
    # In SQLAlchemy 2.0, Row behaves like a tuple, so this should be fine.
    # Explicit conversion if needed: return [(row[0], row[1]) for row in results]
    return results # type: ignore


# --- CREATE INDEXES FOR VECTOR SEARCH OPTIMIZATION ---
async def create_vector_indexes(session: AsyncSession) -> None:
    """
    Creates optimized indexes for vector similarity searches if they don't exist.
    This includes both B-tree indexes for traditional filtering and an IVFFlat index
    for efficient approximate nearest neighbor searches with pgvector.
    
    Args:
        session: An asynchronous SQLAlchemy session.
    """
    try:
        logger.info("Checking and creating vector search indexes")
        
        # Check if indexes exist first
        check_index_query = sqlalchemy.text("""
            SELECT indexname FROM pg_indexes 
            WHERE tablename = 'vector_embeddings' AND indexname = 'idx_vector_embeddings_embedding_ivfflat'
        """)
        result = await session.execute(check_index_query)
        if result.first() is None:
            # Create IVFFlat index for faster ANN searches
            # Lists 100 is a good starting point for most collections, can be tuned based on data size
            ivfflat_query = sqlalchemy.text("""
                CREATE INDEX IF NOT EXISTS idx_vector_embeddings_embedding_ivfflat 
                ON vector_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)
            """)
            await session.execute(ivfflat_query)
            logger.info("Created IVFFlat index for vector embeddings")
        
        # Check for entity type+id index
        check_entity_index_query = sqlalchemy.text("""
            SELECT indexname FROM pg_indexes 
            WHERE tablename = 'vector_embeddings' AND indexname = 'idx_vector_embeddings_entity_combined'
        """)
        result = await session.execute(check_entity_index_query)
        if result.first() is None:
            # Create optimized index for entity type + id searches (frequently used in joins)
            entity_index_query = sqlalchemy.text("""
                CREATE INDEX IF NOT EXISTS idx_vector_embeddings_entity_combined 
                ON vector_embeddings (entity_type, entity_id)
            """)
            await session.execute(entity_index_query)
            logger.info("Created combined entity type+id index")
        
        # Add GIN index for text search on goods_services terms if needed
        check_gin_index_query = sqlalchemy.text("""
            SELECT indexname FROM pg_indexes 
            WHERE tablename = 'goods_services' AND indexname = 'idx_goods_services_term_gin'
        """)
        result = await session.execute(check_gin_index_query)
        if result.first() is None:
            gin_index_query = sqlalchemy.text("""
                CREATE INDEX IF NOT EXISTS idx_goods_services_term_gin 
                ON goods_services USING gin (term gin_trgm_ops)
            """)
            
            # Add the trgm extension if not present
            await session.execute(sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
            
            # Create the index
            await session.execute(gin_index_query)
            logger.info("Created GIN index for text search on goods_services terms")
        
        # Commit all index changes
        await session.commit()
        logger.info("Successfully created all vector search indexes")
            
    except Exception as e:
        exception("Error creating vector indexes", exc=e)
        await session.rollback()
        raise


# --- Initialize Database with Indexes ---
async def initialize_database() -> None:
    """
    Initializes the database by:
    1. Creating tables if they don't exist
    2. Adding optimized indexes for vector searches
    
    Should be called during application startup.
    """
    logger.info("Initializing database")
    try:
        # Get engine and create tables
        engine = await get_async_engine()
        async with engine.begin() as conn:
            # Create tables if they don't exist
            # Note: This assumes your models have __table_args__ with 'extend_existing=True'
            # to avoid errors if tables already exist
            await conn.run_sync(Base.metadata.create_all)
        
        # Create session and add indexes
        AsyncSessionLocal = await get_async_session()
        async with AsyncSessionLocal() as session:
            async with session.begin():
                await create_vector_indexes(session)
        
        logger.info("Database initialization completed successfully")
    except Exception as e:
        exception("Database initialization failed", exc=e)
        raise


# Need to import Base for potential Base.metadata.create_all usage
from src.models import Base
# Remove psycopg2 import as we now use asyncpg via the connection string
# import psycopg2
# Import asyncpg for type checking or direct use if needed, though SQLAlchemy handles it
import asyncpg # Add asyncpg import
