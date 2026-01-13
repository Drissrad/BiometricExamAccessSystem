"""
Modèles pour les examens
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, Enum, Float
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

from app.database import Base


class ExamStatus(str, enum.Enum):
    """Statuts d'un examen"""
    DRAFT = "draft"
    PUBLISHED = "published"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class Exam(Base):
    """Modèle Examen"""
    __tablename__ = "exams"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    duration_minutes = Column(Integer, nullable=False)  # Durée en minutes
    status = Column(Enum(ExamStatus), default=ExamStatus.DRAFT)
    
    # Questions (stockées en JSON)
    questions_json = Column(Text, nullable=True)
    
    # Configuration de surveillance
    face_check_enabled = Column(Boolean, default=True)
    voice_check_enabled = Column(Boolean, default=True)
    face_check_interval = Column(Integer, default=5)  # secondes (vérification temps réel)
    voice_check_interval = Column(Integer, default=120)  # secondes
    
    # Dates
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Créateur (Admin)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Relations
    sessions = relationship("ExamSession", back_populates="exam")
    
    def __repr__(self):
        return f"<Exam {self.title}>"


class SessionStatus(str, enum.Enum):
    """Statuts d'une session d'examen"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SUSPENDED = "suspended"
    TERMINATED = "terminated"
    DISQUALIFIED = "disqualified"  # Triche détectée


class ExamSession(Base):
    """Session d'examen d'un candidat"""
    __tablename__ = "exam_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    exam_id = Column(Integer, ForeignKey("exams.id"), nullable=False)
    
    status = Column(Enum(SessionStatus), default=SessionStatus.PENDING)
    
    # Réponses (stockées en JSON)
    answers_json = Column(Text, nullable=True)
    score = Column(Float, nullable=True)
    
    # Statistiques de surveillance
    total_face_checks = Column(Integer, default=0)
    successful_face_checks = Column(Integer, default=0)
    total_voice_checks = Column(Integer, default=0)
    successful_voice_checks = Column(Integer, default=0)
    anomaly_count = Column(Integer, default=0)
    
    # Dates
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relations
    user = relationship("User", back_populates="exam_sessions")
    exam = relationship("Exam", back_populates="sessions")
    
    def __repr__(self):
        return f"<ExamSession user={self.user_id} exam={self.exam_id}>"
