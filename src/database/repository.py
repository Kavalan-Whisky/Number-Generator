"""
Repository layer for CRUD operations on database models.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from src.database.models import AnalysisResult, GeneratedSequence, GenerationSession


class GenerationSessionRepository:
    """CRUD operations for GenerationSession."""

    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        generator_type: str,
        algorithm: str,
        count: int,
        values: List[float],
        parameters: Dict = None,
        seed: int = None,
        variant: str = None,
    ) -> GenerationSession:
        """Create a new generation session with values."""
        session = GenerationSession(
            generator_type=generator_type,
            algorithm=algorithm,
            variant=variant,
            count=count,
            seed=seed,
            success=True,
        )
        if parameters:
            session.parameters = parameters

        if values:
            session.checksum = session.compute_checksum(values)
            session.min_value = float(min(values))
            session.max_value = float(max(values))
            session.mean_value = sum(values) / len(values)

        self.db.add(session)
        self.db.flush()

        # Add sequence
        seq = GeneratedSequence(session_id=session.id)
        seq.values = values
        self.db.add(seq)
        self.db.commit()
        self.db.refresh(session)
        return session

    def get_by_id(self, session_id: int) -> Optional[GenerationSession]:
        return self.db.get(GenerationSession, session_id)

    def list_recent(self, limit: int = 20) -> List[GenerationSession]:
        return (
            self.db.query(GenerationSession)
            .order_by(GenerationSession.created_at.desc())
            .limit(limit)
            .all()
        )

    def list_by_type(self, generator_type: str, limit: int = 20) -> List[GenerationSession]:
        return (
            self.db.query(GenerationSession)
            .filter(GenerationSession.generator_type == generator_type)
            .order_by(GenerationSession.created_at.desc())
            .limit(limit)
            .all()
        )

    def delete(self, session_id: int) -> bool:
        session = self.get_by_id(session_id)
        if session:
            self.db.delete(session)
            self.db.commit()
            return True
        return False

    def count(self) -> int:
        return self.db.query(GenerationSession).count()

    def search_by_checksum(self, checksum: str) -> Optional[GenerationSession]:
        return self.db.query(GenerationSession).filter(
            GenerationSession.checksum == checksum
        ).first()


class SequenceRepository:
    """CRUD operations for GeneratedSequence."""

    def __init__(self, db: Session):
        self.db = db

    def get_by_session(self, session_id: int) -> List[GeneratedSequence]:
        return self.db.query(GeneratedSequence).filter(
            GeneratedSequence.session_id == session_id
        ).all()

    def get_by_id(self, seq_id: int) -> Optional[GeneratedSequence]:
        return self.db.get(GeneratedSequence, seq_id)

    def update_stats(self, seq_id: int, stats: Dict) -> bool:
        seq = self.get_by_id(seq_id)
        if seq:
            seq.stats = stats
            self.db.commit()
            return True
        return False

    def tag_sequence(self, seq_id: int, tags: List[str]) -> bool:
        seq = self.get_by_id(seq_id)
        if seq:
            seq.tags = ",".join(tags)
            self.db.commit()
            return True
        return False

    def search_by_tags(self, tag: str) -> List[GeneratedSequence]:
        return self.db.query(GeneratedSequence).filter(
            GeneratedSequence.tags.contains(tag)
        ).all()


class AnalysisRepository:
    """CRUD operations for AnalysisResult."""

    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        session_id: int,
        analysis_type: str,
        results: Dict,
        passed: bool = None,
        score: float = None,
    ) -> AnalysisResult:
        result = AnalysisResult(
            session_id=session_id,
            analysis_type=analysis_type,
            passed=passed,
            score=score,
        )
        result.results = results
        self.db.add(result)
        self.db.commit()
        self.db.refresh(result)
        return result

    def get_by_session(self, session_id: int) -> List[AnalysisResult]:
        return self.db.query(AnalysisResult).filter(
            AnalysisResult.session_id == session_id
        ).all()

    def get_by_type(self, analysis_type: str, limit: int = 50) -> List[AnalysisResult]:
        return (
            self.db.query(AnalysisResult)
            .filter(AnalysisResult.analysis_type == analysis_type)
            .order_by(AnalysisResult.created_at.desc())
            .limit(limit)
            .all()
        )


class UnitOfWork:
    """Unit of Work pattern combining all repositories."""

    def __init__(self, db: Session):
        self.db = db
        self.sessions = GenerationSessionRepository(db)
        self.sequences = SequenceRepository(db)
        self.analyses = AnalysisRepository(db)

    def commit(self) -> None:
        self.db.commit()

    def rollback(self) -> None:
        self.db.rollback()

    def close(self) -> None:
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.rollback()
        self.close()
