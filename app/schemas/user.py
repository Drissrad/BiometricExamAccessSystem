"""
Schémas Pydantic pour les utilisateurs
"""
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime
from app.models.user import UserRole


class UserBase(BaseModel):
    """Schéma de base utilisateur"""
    email: EmailStr
    nom: str
    prenom: str


class UserCreate(UserBase):
    """Création d'un utilisateur"""
    password: str
    role: UserRole = UserRole.CANDIDAT


class UserUpdate(BaseModel):
    """Mise à jour d'un utilisateur"""
    nom: Optional[str] = None
    prenom: Optional[str] = None
    is_active: Optional[bool] = None


class UserResponse(UserBase):
    """Réponse utilisateur"""
    id: int
    role: UserRole
    is_active: bool
    is_enrolled: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class UserLogin(BaseModel):
    """Connexion utilisateur"""
    email: EmailStr
    password: str


class Token(BaseModel):
    """Token JWT"""
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Données du token"""
    user_id: Optional[int] = None
    email: Optional[str] = None
    role: Optional[str] = None
