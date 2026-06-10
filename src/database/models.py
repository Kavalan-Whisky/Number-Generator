"""
SQLAlchemy models for generation history and analysis results.
"""

import json
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Column, DateTime, Float, ForeignKey, Integer,
    String, Text, Boolean, JSON, create_engine
)
from sqlalchemy.orm import DeclarativeBase, relationship, Session


class Base(DeclarativeBase):
    pass


class GenerationSession(Base):
    """
    Records a number generation session.
    Stores metadata about what was generated and how.
    """
    __tablename__ = "generation_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Generation type and subtype
    generator_type = Column(String(50), nullable=False)  # random, prime, fibonacci, etc.
    algorithm = Column(String(100))  # specific algorithm used
    variant = Column(String(100))  # variant/subtype

    # Parameters
    count = Column(Integer, nullable=False)
    parameters_json = Column(Text)  # JSON string of generation parameters
    seed = Column(Integer, nullable=True)

    # Results metadata
    checksum = Column(String(64))  # SHA-256 hash of values
    min_value = Column(Float)
    max_value = Column(Float)
    mean_value = Column(Float)

    # Status
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)

    # Relationships
    sequences = relationship("GeneratedSequence", back_populates="session", cascade="all, delete-orphan")
    analysis_results = relationship("AnalysisResult", back_populates="session", cascade="all, delete-orphan")

    @property
    def parameters(self) -> Dict:
        if self.parameters_json:
            return json.loads(self.parameters_json)
        return {}

    @parameters.setter
    def parameters(self, value: Dict) -> None:
        self.parameters_json = json.dumps(value)

    def compute_checksum(self, values: List[float]) -> str:
        """Compute SHA-256 checksum of the generated values."""
        data_str = ",".join(str(v) for v in values)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def __repr__(self) -> str:
        return f"<GenerationSession(id={self.id}, type={self.generator_type}, count={self.count})>"


class GeneratedSequence(Base):
    """
    Stores the actual generated number sequence.
    Linked to a GenerationSession.
    """
    __tablename__ = "generated_sequences"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("generation_sessions.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # The sequence data
    values_json = Column(Text, nullable=False)  # JSON array of values
    length = Column(Integer, nullable=False)

    # Optional metadata
    name = Column(String(200), nullable=True)
    description = Column(Text, nullable=True)
    tags = Column(String(500), nullable=True)  # comma-separated tags

    # Statistics cache
    stats_json = Column(Text, nullable=True)  # Cached statistics as JSON

    # Relationships
    session = relationship("GenerationSession", back_populates="sequences")

    @property
    def values(self) -> List[float]:
        return json.loads(self.values_json)

    @values.setter
    def values(self, v: List[float]) -> None:
        self.values_json = json.dumps(v)
        self.length = len(v)

    @property
    def stats(self) -> Optional[Dict]:
        if self.stats_json:
            return json.loads(self.stats_json)
        return None

    @stats.setter
    def stats(self, v: Dict) -> None:
        self.stats_json = json.dumps(v)

    def __repr__(self) -> str:
        return f"<GeneratedSequence(id={self.id}, session_id={self.session_id}, length={self.length})>"


class AnalysisResult(Base):
    """
    Stores results of statistical analysis on a sequence.
    """
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("generation_sessions.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Analysis type
    analysis_type = Column(String(50), nullable=False)  # statistical, randomness, patterns, etc.

    # Results
    results_json = Column(Text, nullable=False)
    passed = Column(Boolean, nullable=True)  # For tests with pass/fail
    score = Column(Float, nullable=True)  # Overall score/quality metric

    # Relationships
    session = relationship("GenerationSession", back_populates="analysis_results")

    @property
    def results(self) -> Dict:
        return json.loads(self.results_json)

    @results.setter
    def results(self, v: Dict) -> None:
        self.results_json = json.dumps(v, default=str)

    def __repr__(self) -> str:
        return f"<AnalysisResult(id={self.id}, type={self.analysis_type})>"


def create_tables(engine) -> None:
    """Create all database tables."""
    Base.metadata.create_all(engine)


def get_engine(url: str = "sqlite:///numgen.db"):
    """Create SQLAlchemy engine."""
    return create_engine(url, echo=False)


def get_session(engine) -> Session:
    """Create a new database session."""
    from sqlalchemy.orm import sessionmaker
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()
