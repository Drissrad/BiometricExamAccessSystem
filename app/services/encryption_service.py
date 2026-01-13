"""
Service de chiffrement AES-256 pour les données biométriques
Utilise Fernet (AES-128-CBC avec HMAC pour l'authentification)
"""
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64
import os
import logging

logger = logging.getLogger(__name__)


class EncryptionService:
    """
    Service de chiffrement/déchiffrement pour les données biométriques
    Utilise Fernet qui implémente AES-128-CBC avec HMAC-SHA256
    """
    
    def __init__(self, encryption_key: str = None):
        """
        Initialise le service avec une clé de chiffrement
        
        Args:
            encryption_key: Clé de chiffrement (base64). Si None, génère une nouvelle clé.
        """
        if encryption_key:
            # Dériver une clé Fernet à partir de la clé fournie
            self._fernet = self._create_fernet_from_key(encryption_key)
        else:
            # Générer une nouvelle clé (pour la première exécution)
            logger.warning("Aucune clé de chiffrement fournie - utilisation d'une clé par défaut (NON SÉCURISÉ)")
            self._fernet = self._create_fernet_from_key("default-encryption-key-change-this")
    
    def _create_fernet_from_key(self, key: str) -> Fernet:
        """
        Crée un objet Fernet à partir d'une clé string
        Utilise PBKDF2 pour dériver une clé de 32 bytes
        """
        # Sel fixe pour la dérivation (pourrait être stocké séparément)
        salt = b'biometrie_exam_salt_v1'
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        
        key_bytes = kdf.derive(key.encode())
        fernet_key = base64.urlsafe_b64encode(key_bytes)
        
        return Fernet(fernet_key)
    
    def encrypt(self, data: bytes) -> bytes:
        """
        Chiffre des données binaires
        
        Args:
            data: Données à chiffrer (bytes)
            
        Returns:
            Données chiffrées (bytes)
        """
        if data is None:
            return None
        
        try:
            encrypted = self._fernet.encrypt(data)
            logger.debug(f"Données chiffrées: {len(data)} bytes -> {len(encrypted)} bytes")
            return encrypted
        except Exception as e:
            logger.error(f"Erreur lors du chiffrement: {e}")
            raise
    
    def decrypt(self, encrypted_data: bytes) -> bytes:
        """
        Déchiffre des données chiffrées
        
        Args:
            encrypted_data: Données chiffrées (bytes)
            
        Returns:
            Données déchiffrées (bytes)
        """
        if encrypted_data is None:
            return None
        
        try:
            decrypted = self._fernet.decrypt(encrypted_data)
            logger.debug(f"Données déchiffrées: {len(encrypted_data)} bytes -> {len(decrypted)} bytes")
            return decrypted
        except InvalidToken:
            logger.error("Échec du déchiffrement: token invalide (clé incorrecte ou données corrompues)")
            raise ValueError("Impossible de déchiffrer les données biométriques. Clé incorrecte ou données corrompues.")
        except Exception as e:
            logger.error(f"Erreur lors du déchiffrement: {e}")
            raise
    
    @staticmethod
    def generate_key() -> str:
        """
        Génère une nouvelle clé de chiffrement sécurisée
        
        Returns:
            Clé de chiffrement (string base64)
        """
        # Génère 32 bytes aléatoires et les encode en base64
        random_bytes = os.urandom(32)
        key = base64.urlsafe_b64encode(random_bytes).decode()
        return key


# Instance globale - sera initialisée avec la clé de config
encryption_service = None


def get_encryption_service():
    """
    Retourne l'instance du service de chiffrement
    Lazy initialization pour attendre que la config soit chargée
    """
    global encryption_service
    
    if encryption_service is None:
        from app.config import settings
        encryption_service = EncryptionService(settings.BIOMETRIC_ENCRYPTION_KEY)
    
    return encryption_service
