"""
Configuration de l'application
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Paramètres de configuration"""
    
    # Application
    APP_NAME: str = "Biométrie Examen"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Base de données
    DATABASE_URL: str = "sqlite+aiosqlite:///./biometrie_exam.db"
    
    # Sécurité
    SECRET_KEY: str = "cle-secrete-tres-longue-et-complexe-a-changer"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    
    # Biométrie - Seuils (TRÈS SÉCURISÉS)
    FACE_RECOGNITION_THRESHOLD: float = 0.75  # Seuil STRICT pour le score facial
    VOICE_RECOGNITION_THRESHOLD: float = 0.80  # Seuil STRICT pour le score vocal
    MULTIMODAL_THRESHOLD: float = 0.75
    
    # Seuils minimaux individuels - chaque modalité doit les atteindre
    MIN_FACE_SCORE: float = 0.75  # Score facial minimum STRICT (empêche imposteurs)
    MIN_VOICE_SCORE: float = 0  # Score vocal minimum STRICT (empêche imposteurs)
    
    # Poids pour la fusion multimodale
    FACE_WEIGHT: float = 0.6
    VOICE_WEIGHT: float = 0.4
    
    # Surveillance (intervalles en secondes)
    FACE_CHECK_INTERVAL_SECONDS: int = 5  # Vérification faciale toutes les 5s (temps réel)
    VOICE_CHALLENGE_INTERVAL_SECONDS: int = 120
    MAX_ABSENCE_DURATION_SECONDS: int = 15  # Absence max 15s (3 échecs consécutifs)
    
    # Stockage
    UPLOAD_DIR: str = "uploads"
    
    # Chiffrement des données biométriques (AES-256)
    # IMPORTANT: Générer une clé unique et la stocker dans .env
    # Utiliser: python -c "from app.services.encryption_service import EncryptionService; print(EncryptionService.generate_key())"
    BIOMETRIC_ENCRYPTION_KEY: str = "CHANGE-THIS-KEY-IN-ENV-FILE-USE-GENERATE-KEY"
    
    # ============ ANTI-SPOOFING (Détection de fraude biométrique) ============
    # Activer/désactiver les vérifications anti-spoofing
    ANTISPOOF_ENABLED: bool = True
    ANTISPOOF_STRICT_MODE: bool = True  # Si True, rejette si l'anti-spoofing échoue à analyser
    
    # Seuils pour le visage (0.0 à 1.0, plus haut = plus strict)
    ANTISPOOF_FACE_TEXTURE_THRESHOLD: float = 0.3  # Analyse de texture LBP
    ANTISPOOF_FACE_EDGE_THRESHOLD: float = 0.2      # Détection de bords
    ANTISPOOF_FACE_COLOR_THRESHOLD: float = 0.3     # Distribution des couleurs
    ANTISPOOF_FACE_REFLECTION_THRESHOLD: float = 0.4  # Reflets d'écran
    
    # Seuils pour la voix (0.0 à 1.0, plus haut = plus strict)
    ANTISPOOF_VOICE_QUALITY_THRESHOLD: float = 0.3   # Qualité audio
    ANTISPOOF_VOICE_REPLAY_THRESHOLD: float = 0.35   # Détection de replay
    ANTISPOOF_VOICE_SPECTRAL_THRESHOLD: float = 0.3  # Analyse spectrale
    ANTISPOOF_VOICE_SYNTHETIC_THRESHOLD: float = 0.4 # Voix synthétique
    
    # Score global minimum pour passer l'anti-spoofing
    ANTISPOOF_MIN_FACE_SCORE: float = 0.5  # Score liveness visage minimum
    ANTISPOOF_MIN_VOICE_SCORE: float = 0.5  # Score liveness voix minimum
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

# Créer le dossier uploads s'il n'existe pas
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
