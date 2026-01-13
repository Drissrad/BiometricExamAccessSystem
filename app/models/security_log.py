"""
Modèle pour les journaux de sécurité
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Enum, Float
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

from app.database import Base


class LogType(str, enum.Enum):
    """Types de logs de sécurité"""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILED = "login_failed"
    FACE_CHECK_SUCCESS = "face_check_success"
    FACE_CHECK_FAILED = "face_check_failed"
    VOICE_CHECK_SUCCESS = "voice_check_success"
    VOICE_CHECK_FAILED = "voice_check_failed"
    ABSENCE_DETECTED = "absence_detected"
    PERSON_CHANGED = "person_changed"
    EXAM_STARTED = "exam_started"
    EXAM_COMPLETED = "exam_completed"
    EXAM_SUSPENDED = "exam_suspended"
    ANOMALY_DETECTED = "anomaly_detected"
    ENROLLMENT_SUCCESS = "enrollment_success"
    ENROLLMENT_FAILED = "enrollment_failed"
    CHEATING_DETECTED = "cheating_detected"  # Triche détectée
    # Types pour FAR/FRR
    IMPOSTOR_ATTEMPT = "impostor_attempt"  # Tentative d'imposteur (FAR)
    GENUINE_REJECTED = "genuine_rejected"  # Utilisateur légitime rejeté (FRR)
    GENUINE_ACCEPTED = "genuine_accepted"  # Utilisateur légitime accepté (True Positive)
    IMPOSTOR_REJECTED = "impostor_rejected"  # Imposteur rejeté (True Negative)
    # Anti-spoofing
    SPOOFING_DETECTED = "spoofing_detected"  # Tentative de fraude biométrique détectée


class SecurityLog(Base):
    """Journaux de sécurité"""
    __tablename__ = "security_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    exam_session_id = Column(Integer, ForeignKey("exam_sessions.id"), nullable=True)
    
    log_type = Column(Enum(LogType), nullable=False)
    message = Column(Text, nullable=True)
    
    # Scores biométriques (si applicable)
    face_score = Column(Float, nullable=True)
    voice_score = Column(Float, nullable=True)
    combined_score = Column(Float, nullable=True)
    
    # Métadonnées
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(255), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relations
    user = relationship("User", back_populates="security_logs")
    
    def __repr__(self):
        return f"<SecurityLog {self.log_type.value}>"
