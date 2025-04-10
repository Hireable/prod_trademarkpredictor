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
from sqlalchemy import select # Add select import
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker # Import async components
from sqlalchemy.exc import OperationalError, IntegrityError
from supabase import create_client, Client

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
        raise ValueError(
            "Missing required Supabase environment variables: "
            "SUPABASE_URL, SUPABASE_KEY"
        )

    try:
        # Create and store the Supabase client
        _supabase = create_client(supabase_url, supabase_key)
        return _supabase
    except Exception as e:
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
            raise ValueError("Invalid SUPABASE_URL format.")

    if not all([db_host, db_password]):
        raise ValueError(
            "Missing required environment variables for DB engine: "
            "SUPABASE_URL, SUPABASE_SERVICE_KEY"
        )

    try:
        # Use postgresql+asyncpg dialect
        db_url = f"postgresql+asyncpg://{db_user}:{db_password}@{db_host}:5432/{db_name}"

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
        print("Async database engine created successfully.") # Added log
        return _async_engine
    except OperationalError as e:
        print(f"Database connection failed: {e}")
        raise
    except Exception as e:
        # Ensure the original exception type is preserved if possible
        if isinstance(e, OperationalError):
             raise
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
        print("Async session maker created.") # Added log
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
            print(f"Embedding already exists for goods_services ID {goods_service_id}.")
            return True

        # Generate the embedding (assuming generate_embedding is sync, which is okay if CPU-bound)
        embedding_vector: Optional[List[float]] = generate_embedding(term)

        if embedding_vector is None:
            print(f"Failed to generate embedding for term: {term}")
            return False

        # Create and store the new embedding record
        new_embedding = VectorEmbeddingOrm(
            entity_type='goods_services',
            entity_id=goods_service_id,
            embedding=embedding_vector
        )
        session.add(new_embedding)
        await session.flush() # Use await
        print(f"Successfully stored embedding for goods_services ID {goods_service_id}.")
        # No explicit commit here, assuming caller manages the transaction boundary
        return True

    except IntegrityError as e:
        # Handle potential race conditions if another process inserted concurrently
        print(f"IntegrityError storing embedding for goods_services ID {goods_service_id}: {e}")
        await session.rollback() # Use await
        # Optionally, re-query async to confirm if it exists now
        stmt = select(VectorEmbeddingOrm).filter_by(
            entity_type='goods_services',
            entity_id=goods_service_id
        )
        result = await session.execute(stmt)
        return result.scalars().first() is not None # Check if exists after rollback
    except Exception as e:
        print(f"Error storing embedding for goods_services ID {goods_service_id}: {e}")
        await session.rollback() # Use await
        return False


async def find_similar_goods_services( # Make async
    query_embedding: List[float],
    limit: int = 10,
    session: Optional[AsyncSession] = None, # Accept optional AsyncSession
) -> List[Tuple[GoodsServiceOrm, float]]:
    """
    Finds goods/services terms similar to the provided query embedding asynchronously.

    Performs a vector similarity search (cosine distance) against the
    `vector_embeddings` table where entity_type is 'goods_services'.

    Args:
        query_embedding: The vector embedding to search against.
        limit: The maximum number of similar items to return.
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
        stmt = (
            select( # Use select() from sqlalchemy
                GoodsServiceOrm,
                VectorEmbeddingOrm.embedding.cosine_distance(query_embedding).label("distance")
            )
            .join(VectorEmbeddingOrm,
                  (VectorEmbeddingOrm.entity_id == GoodsServiceOrm.id) &
                  (VectorEmbeddingOrm.entity_type == 'goods_services'))
            .order_by(sqlalchemy.asc("distance")) # Use sqlalchemy.asc explicitly
            .limit(limit)
        )

        # Execute the query asynchronously
        query_result = await db_session.execute(stmt)
        results = query_result.all() # Returns List[Row], Row contains (GoodsServiceOrm, float)

    except Exception as e:
        print(f"Error finding similar goods/services: {e}")
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


# --- Example Usage (Optional, for testing/scripts) ---
# Need to be adapted for async if run directly
# async def populate_all_embeddings_async():
#     """Generates and stores embeddings for all existing goods_services asynchronously."""
#     AsyncSessionLocal = await get_async_session()
#     async with AsyncSessionLocal() as session:
#         async with session.begin(): # Manage transaction
#             try:
#                 stmt = select(GoodsServiceOrm)
#                 result = await session.execute(stmt)
#                 goods_services_records = result.scalars().all()
#                 print(f"Found {len(goods_services_records)} goods/services records.")
#                 count_success = 0
#                 count_fail = 0
#                 for record in goods_services_records:
#                     if record.id is not None and record.term:
#                         # Pass the session explicitly
#                         if await store_goods_service_embedding(record.id, record.term, session):
#                             count_success += 1
#                         else:
#                             count_fail += 1
#                     else:
#                         print(f"Skipping record with missing ID or term: {record}")
#                         count_fail += 1
#
#                 print(f"Attempted embedding storage: {count_success} succeeded, {count_fail} failed/skipped.")
#                 # Commit happens automatically with session.begin() context exiting successfully
#
#             except Exception as e:
#                 print(f"An error occurred during async embedding population: {e}")
#                 # Rollback happens automatically with session.begin() context exiting with exception
#                 raise # Re-raise after logging
#
# # if __name__ == "__main__":
# #     import asyncio
# #     print("Populating embeddings asynchronously...")
# #     asyncio.run(populate_all_embeddings_async())
# #     print("Async embedding population attempt finished.")
# #
# #     # Example async search
# #     async def run_search():
# #         print("\nTesting async search...")
# #         example_term = "computer software"
# #         example_embedding = generate_embedding(example_term) # Sync embedding gen is ok
# #         if example_embedding:
# #             print(f"Searching for terms similar to '{example_term}'...")
# #             # Use the find function directly, it manages its own session if needed
# #             similar_items = await find_similar_goods_services(example_embedding, limit=5)
# #             if similar_items:
# #                 print("Found similar items:")
# #                 for gs, distance in similar_items: # Unpack tuple
# #                     print(f"  - Term: '{gs.term}', Class: {gs.nice_class}, Distance: {distance:.4f}")
# #             else:
# #                 print("No similar items found or error occurred.")
# #         else:
# #             print(f"Could not generate embedding for '{example_term}'.")
# #
# #     asyncio.run(run_search())


# Need to import Base for potential Base.metadata.create_all usage
from src.models import Base
# Remove psycopg2 import as we now use asyncpg via the connection string
# import psycopg2
# Import asyncpg for type checking or direct use if needed, though SQLAlchemy handles it
import asyncpg # Add asyncpg import
