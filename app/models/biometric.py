"""
Modèle pour les données biométriques
"""
from sqlalchemy import Column, Integer, ForeignKey, LargeBinary, DateTime, Float
from sqlalchemy.orm import relationship
from datetime import datetime

from app.database import Base


class BiometricData(Base):
    """
    Stocke les descripteurs biométriques (pas les images/audios bruts)
    - Face encoding: vecteur de 128 dimensions
    - Voice encoding: caractéristiques vocales (MFCC moyens)
    """
    __tablename__ = "biometric_data"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    
    # Descripteur facial (numpy array sérialisé)
    face_encoding = Column(LargeBinary, nullable=True)
    face_encoding_quality = Column(Float, default=0.0)  # Qualité de l'enrôlement facial
    
    # Descripteur vocal (numpy array sérialisé)
    voice_encoding = Column(LargeBinary, nullable=True)
    voice_encoding_quality = Column(Float, default=0.0)  # Qualité de l'enrôlement vocal
    
    # Métadonnées
    enrolled_at = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relation
    user = relationship("User", back_populates="biometric_data")
    
    def __repr__(self):
        return f"<BiometricData user_id={self.user_id}>"
