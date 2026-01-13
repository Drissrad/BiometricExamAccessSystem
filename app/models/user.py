"""
Modèle Utilisateur (Admin et Candidat)
"""
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Enum, LargeBinary
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

from app.database import Base


class UserRole(str, enum.Enum):
    """Rôles des utilisateurs"""
    ADMIN = "admin"
    CANDIDAT = "candidat"


class User(Base):
    """Modèle Utilisateur"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    nom = Column(String(100), nullable=False)
    prenom = Column(String(100), nullable=False)
    role = Column(Enum(UserRole), default=UserRole.CANDIDAT, nullable=False)
    is_active = Column(Boolean, default=True)
    is_enrolled = Column(Boolean, default=False)  # Enrôlement biométrique effectué
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relations
    biometric_data = relationship("BiometricData", back_populates="user", uselist=False)
    exam_sessions = relationship("ExamSession", back_populates="user")
    security_logs = relationship("SecurityLog", back_populates="user")
    
    def __repr__(self):
        return f"<User {self.email}>"
