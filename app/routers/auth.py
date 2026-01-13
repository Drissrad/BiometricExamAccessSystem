"""
Routes d'authentification
"""
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import timedelta

from app.database import get_db
from app.schemas.user import UserCreate, UserResponse, UserLogin, Token
from app.schemas.biometric import BiometricEnrollRequest, BiometricVerifyRequest, BiometricScoreResponse, BiometricLoginRequest
from app.services.auth_service import (
    authenticate_user, create_access_token, decode_access_token,
    get_user_by_id, get_user_by_email, create_user
)
from app.services.biometric_service import biometric_service
from app.models.user import User, UserRole
from app.config import settings

router = APIRouter(prefix="/auth", tags=["Authentification"])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db)
) -> User:
    """Récupérer l'utilisateur courant à partir du token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token invalide",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    print(f"DEBUG - Token reçu: {token[:50]}..." if len(token) > 50 else f"DEBUG - Token reçu: {token}")
    token_data = decode_access_token(token)
    print(f"DEBUG - Token data: {token_data}")
    
    if token_data is None:
        print("DEBUG - Token data is None!")
        raise credentials_exception
    
    user = await get_user_by_id(db, token_data.user_id)
    if user is None:
        print(f"DEBUG - User not found for id: {token_data.user_id}")
        raise credentials_exception
    
    print(f"DEBUG - User found: {user.email}")
    return user


async def get_current_admin(current_user: User = Depends(get_current_user)) -> User:
    """Vérifier que l'utilisateur est un admin"""
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Accès réservé aux administrateurs"
        )
    return current_user


@router.post("/register", response_model=UserResponse)
async def register(user_data: UserCreate, db: AsyncSession = Depends(get_db)):
    """Inscription d'un nouvel utilisateur"""
    # Vérifier si l'email existe déjà
    existing = await get_user_by_email(db, user_data.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cet email est déjà utilisé"
        )
    
    user = await create_user(
        db,
        email=user_data.email,
        password=user_data.password,
        nom=user_data.nom,
        prenom=user_data.prenom,
        role=user_data.role
    )
    
    return user


@router.post("/token", response_model=Token)
async def login_for_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    """Connexion et obtention du token (sans biométrie)"""
    user = await authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email ou mot de passe incorrect",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(
        data={"sub": str(user.id), "email": user.email, "role": user.role.value}
    )
    
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/login-biometric", response_model=Token)
async def login_with_biometric(
    data: BiometricLoginRequest,
    db: AsyncSession = Depends(get_db)
):
    """Connexion avec vérification biométrique (visage + voix)"""
    # Authentification par mot de passe
    user = await authenticate_user(db, data.email, data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email ou mot de passe incorrect"
        )
    
    # Vérifier si l'utilisateur est enrôlé
    if not user.is_enrolled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Vos données biométriques ne sont pas encore enregistrées. Contactez l'administrateur."
        )
    
    # Vérification biométrique avec anti-spoofing
    is_verified, face_score, voice_score, combined_score, message, antispoof = await biometric_service.verify_user(
        db,
        user.id,
        data.face_image_base64,
        data.voice_audio_base64
    )
    
    if not is_verified:
        # Vérifier si c'est du spoofing
        if antispoof.get("enabled") and (not antispoof.get("face_passed") or not antispoof.get("voice_passed")):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Fraude biométrique détectée: {message}"
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Vérification biométrique échouée: {message}"
        )
    
    access_token = create_access_token(
        data={"sub": str(user.id), "email": user.email, "role": user.role.value}
    )
    
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/enroll", response_model=dict)
async def enroll_biometric(
    data: BiometricEnrollRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Enrôlement biométrique (visage + voix)"""
    success, message = await biometric_service.enroll_user(
        db,
        current_user.id,
        data.face_image_base64,
        data.voice_audio_base64
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )
    
    return {"success": True, "message": message}


@router.post("/verify-biometric", response_model=BiometricScoreResponse)
async def verify_biometric(
    data: BiometricVerifyRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Vérifier l'identité biométrique avec anti-spoofing"""
    is_verified, face_score, voice_score, combined_score, message, antispoof = await biometric_service.verify_user(
        db,
        current_user.id,
        data.face_image_base64,
        data.voice_audio_base64
    )
    
    return BiometricScoreResponse(
        face_score=face_score,
        voice_score=voice_score,
        combined_score=combined_score,
        is_verified=is_verified,
        message=message
    )


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    """Récupérer les informations de l'utilisateur connecté"""
    return current_user
