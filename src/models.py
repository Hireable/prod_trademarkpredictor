# src/models.py
"""
Pydantic models for representing trademark data, similarity inputs/outputs,
and prediction results within the UK/EU Trademark AI Agent.

SQLAlchemy models for database interaction.
"""

from typing import List, Optional

from pydantic import BaseModel, Field
import sqlalchemy  # Add import for sqlalchemy module

# --- Pydantic Models ---

class GoodsService(BaseModel):
    """
    Represents a single item of goods or services associated with a trademark.

    Attributes:
        term: The specific description of the good or service.
        nice_class: The NICE classification class number.
        # Removed embedding: it's stored separately according to the schema
    """

    term: str = Field(..., description="The specific description of the good or service.")
    nice_class: int = Field(..., description="The NICE classification class number.")
    # embedding: Optional[List[float]] = Field( <-- Removed
    #     default=None,
    #     description="Vector embedding for semantic similarity using pgvector.",
    # )


class Wordmark(BaseModel):
    """
    Represents the textual element (wordmark) of a trademark.

    This is used for comparisons based on visual, aural, and conceptual
    similarity algorithms (e.g., Levenshtein, phonetic matching).

    Attributes:
        mark_text: The literal text of the wordmark.
    """

    mark_text: str = Field(..., description="The literal text of the wordmark.")


class Trademark(BaseModel):
    """
    Represents the core comparable elements of a trademark application or registration.
    (Note: This Pydantic model represents input/transfer data, not the DB structure directly)

    Attributes:
        identifier: A unique identifier for the trademark (e.g., application number).
        wordmark: The textual Wordmark component of the trademark.
        goods_services: A list of GoodsService items associated with the trademark.
    """

    identifier: str = Field(
        ..., description="Unique identifier (e.g., application/registration number)."
    )
    wordmark: Wordmark = Field(..., description="The textual Wordmark component.")
    goods_services: List[GoodsService] = Field(
        ..., description="List of goods and services associated with the trademark."
    )


class SimilarityTaskInput(BaseModel):
    """
    Input data structure for the trademark similarity analysis task.

    Bundles the two trademarks (applicant's and opponent's) to be compared.

    Attributes:
        applicant_trademark: The Trademark object representing the applicant's mark.
        opponent_trademark: The Trademark object representing the opponent's mark.
    """

    applicant_trademark: Trademark = Field(
        ..., description="The applicant's trademark data."
    )
    opponent_trademark: Trademark = Field(
        ..., description="The opponent's trademark data."
    )


class SimilarityScores(BaseModel):
    """
    Holds the calculated similarity scores between two trademarks across various dimensions.

    These scores result from different comparison algorithms (e.g., vector distance
    for goods/services, Levenshtein for visual, phonetic algorithms for aural).

    Attributes:
        goods_services_similarity: Similarity score based on goods/services (often semantic/vector).
        visual_similarity: Similarity score based on visual appearance of wordmarks (e.g., Levenshtein).
        aural_similarity: Similarity score based on phonetic similarity of wordmarks.
        conceptual_similarity: Similarity score based on the meaning/concept of wordmarks.
    """

    goods_services_similarity: Optional[float] = Field(
        default=None, description="Similarity score for goods and services."
    )
    visual_similarity: Optional[float] = Field(
        default=None, description="Visual similarity score for wordmarks."
    )
    aural_similarity: Optional[float] = Field(
        default=None, description="Aural similarity score for wordmarks."
    )
    conceptual_similarity: Optional[float] = Field(
        default=None, description="Conceptual similarity score for wordmarks."
    )


class PredictionTaskInput(BaseModel):
    """
    Input data structure for the opposition outcome prediction task.

    Combines the details of the compared trademarks and their calculated
    similarity scores, serving as features for the prediction model/logic.

    Attributes:
        applicant_trademark: The Trademark object representing the applicant's mark.
        opponent_trademark: The Trademark object representing the opponent's mark.
        similarity_scores: The calculated SimilarityScores between the two trademarks.
    """

    applicant_trademark: Trademark = Field(
        ..., description="The applicant's trademark data."
    )
    opponent_trademark: Trademark = Field(
        ..., description="The opponent's trademark data."
    )
    similarity_scores: SimilarityScores = Field(
        ..., description="Calculated similarity scores between the trademarks."
    )


class PredictionResult(BaseModel):
    """
    Represents the output of the trademark opposition outcome prediction.

    Attributes:
        predicted_outcome: The predicted outcome string (e.g., 'Opposition Likely Succeeds').
        confidence_score: An optional score indicating the prediction confidence (0.0 to 1.0).
        reasoning: An optional explanation or justification for the predicted outcome.
    """

    predicted_outcome: str = Field(..., description="The predicted outcome string.")
    confidence_score: Optional[float] = Field(
        default=None, description="Confidence score of the prediction (0.0-1.0)."
    )
    reasoning: Optional[str] = Field(
        default=None, description="Explanation for the prediction."
    )


# --- SQLAlchemy ORM Definitions ---

from sqlalchemy import Column, Integer, String, Float, ForeignKey, Index, VARCHAR, TIMESTAMP
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import ARRAY
from pgvector.sqlalchemy import Vector  # Changed from sqlalchemy_pgvector.types import VECTOR

# Embedding dimension based on the migration file
# EMBEDDING_DIMENSION = 1536
# Update dimension based on the chosen model (all-MiniLM-L6-v2)
EMBEDDING_DIMENSION = 384

Base = declarative_base()


# Removed TrademarkOrm as it doesn't align with the provided schema's structure
# (trademark_cases, applicant_marks, etc.)
# class TrademarkOrm(Base):
#     ...


class GoodsServiceOrm(Base):
    """
    SQLAlchemy ORM model representing the 'goods_services' table.

    Stores individual goods or services items, including their term,
    NICE class, and creation/update timestamps. Embeddings are stored
    in the separate 'vector_embeddings' table according to the schema.
    """
    __tablename__ = 'goods_services'

    id: Column[int] = Column(Integer, primary_key=True)
    term: Column[str] = Column(String, nullable=False) # Assuming String is appropriate, adjust if Text needed
    nice_class: Column[int] = Column(Integer, index=True, nullable=False)
    # Removed embedding column:
    # embedding: Column[ARRAY] = Column(ARRAY(Float), nullable=True)
    # trademark_id: Column[int] = Column(Integer, ForeignKey('trademarks.id'), nullable=False) <-- Removed, schema links differently
    created_at: Column[TIMESTAMP] = Column(TIMESTAMP(timezone=True), server_default=sqlalchemy.func.now())
    updated_at: Column[TIMESTAMP] = Column(TIMESTAMP(timezone=True), server_default=sqlalchemy.func.now(), onupdate=sqlalchemy.func.now())


    # Relationship back to the parent trademark (REMOVED, needs rethinking based on full schema)
    # trademark = relationship("TrademarkOrm", back_populates="goods_services")

    # Relationship to embeddings (if needed, needs explicit setup)
    # embeddings = relationship("VectorEmbeddingOrm", primaryjoin="and_(VectorEmbeddingOrm.entity_type=='goods_services', foreign(VectorEmbeddingOrm.entity_id)==GoodsServiceOrm.id)", overlaps="embeddings")


    def __repr__(self) -> str:
        # return f"<GoodsServiceOrm(id={self.id}, term='{self.term}', nice_class={self.nice_class}, trademark_id={self.trademark_id})>" <-- Updated
        return f"<GoodsServiceOrm(id={self.id}, term='{self.term}', nice_class={self.nice_class})>"


class VectorEmbeddingOrm(Base):
    """
    SQLAlchemy ORM model representing the 'vector_embeddings' table.

    Stores vector embeddings linked to different entity types (e.g., goods_services).
    """
    __tablename__ = 'vector_embeddings'

    id: Column[int] = Column(Integer, primary_key=True)
    entity_type: Column[str] = Column(VARCHAR(50), nullable=False, index=True) # Added index based on schema
    entity_id: Column[int] = Column(Integer, nullable=False, index=True) # Added index based on schema
    embedding: Column[Vector] = Column(Vector(EMBEDDING_DIMENSION), nullable=False) # Use Vector type (updated from VECTOR)
    created_at: Column[TIMESTAMP] = Column(TIMESTAMP(timezone=True), server_default=sqlalchemy.func.now())
    updated_at: Column[TIMESTAMP] = Column(TIMESTAMP(timezone=True), server_default=sqlalchemy.func.now(), onupdate=sqlalchemy.func.now())

    # TODO: Define relationships back to specific entity tables if needed, e.g.:
    # goods_service = relationship(
    #     "GoodsServiceOrm",
    #     primaryjoin="and_(VectorEmbeddingOrm.entity_type=='goods_services', foreign(VectorEmbeddingOrm.entity_id)==GoodsServiceOrm.id)",
    #     back_populates="embeddings", # Requires adding `embeddings` relationship to GoodsServiceOrm
    #     uselist=False
    # )

    __table_args__ = (
        Index('idx_vector_embeddings_entity', 'entity_type', 'entity_id'), # Explicit index based on schema
    )

    def __repr__(self) -> str:
        return f"<VectorEmbeddingOrm(id={self.id}, entity_type='{self.entity_type}', entity_id={self.entity_id})>"

# Need to import sqlalchemy for server_default=sqlalchemy.func.now()
import sqlalchemy