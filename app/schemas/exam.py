"""
Schémas Pydantic pour les examens
"""
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from app.models.exam import ExamStatus, SessionStatus


class QuestionSchema(BaseModel):
    """Schéma d'une question"""
    id: int
    text: str
    options: List[str]
    correct_answer: int  # Index de la bonne réponse
    points: float = 1.0


class ExamBase(BaseModel):
    """Schéma de base examen"""
    title: str
    description: Optional[str] = None
    duration_minutes: int


class ExamCreate(ExamBase):
    """Création d'un examen"""
    questions: List[QuestionSchema] = []
    face_check_enabled: bool = True
    voice_check_enabled: bool = True
    face_check_interval: int = 30
    voice_check_interval: int = 120
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class ExamUpdate(BaseModel):
    """Mise à jour d'un examen"""
    title: Optional[str] = None
    description: Optional[str] = None
    duration_minutes: Optional[int] = None
    status: Optional[ExamStatus] = None
    questions: Optional[List[QuestionSchema]] = None


class ExamResponse(ExamBase):
    """Réponse examen"""
    id: int
    status: ExamStatus
    face_check_enabled: bool
    voice_check_enabled: bool
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    created_at: datetime
    
    class Config:
        from_attributes = True


class ExamWithQuestions(ExamResponse):
    """Examen avec questions"""
    questions: List[QuestionSchema] = []


class AnswerSubmit(BaseModel):
    """Soumission d'une réponse"""
    question_id: int
    answer: int  # Index de la réponse choisie


class ExamSessionResponse(BaseModel):
    """Réponse session d'examen"""
    id: int
    exam_id: int
    status: SessionStatus
    score: Optional[float]
    total_face_checks: int
    successful_face_checks: int
    total_voice_checks: int
    successful_voice_checks: int
    anomaly_count: int
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    
    class Config:
        from_attributes = True
