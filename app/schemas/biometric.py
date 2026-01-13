"""
Schémas Pydantic pour la biométrie
"""
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime


class BiometricEnrollRequest(BaseModel):
    """Requête d'enrôlement biométrique"""
    face_image_base64: str  # Image du visage en base64
    voice_audio_base64: str  # Audio de la voix en base64


class BiometricVerifyRequest(BaseModel):
    """Requête de vérification biométrique"""
    face_image_base64: Optional[str] = None
    voice_audio_base64: Optional[str] = None


class BiometricLoginRequest(BaseModel):
    """Requête de connexion avec vérification biométrique"""
    email: EmailStr
    password: str
    face_image_base64: str
    voice_audio_base64: str


class BiometricScoreResponse(BaseModel):
    """Réponse avec les scores biométriques"""
    face_score: Optional[float] = None
    voice_score: Optional[float] = None
    combined_score: Optional[float] = None
    is_verified: bool
    message: str


class FaceCheckRequest(BaseModel):
    """Requête de vérification faciale"""
    image_base64: str


class VoiceChallengeResponse(BaseModel):
    """Défi vocal à lire"""
    challenge_id: str
    text_to_read: str
    expires_at: datetime


class VoiceChallengeSubmit(BaseModel):
    """Soumission du défi vocal"""
    challenge_id: str
    audio_base64: str


class SurveillanceStatus(BaseModel):
    """Statut de surveillance"""
    face_verified: bool
    voice_verified: bool
    last_face_check: Optional[datetime]
    last_voice_check: Optional[datetime]
    anomalies_count: int
    is_active: bool
