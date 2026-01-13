"""
Service de reconnaissance faciale
"""
import numpy as np
import cv2
import base64
import face_recognition
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FaceRecognitionService:
    """Service pour la reconnaissance faciale"""
    
    def __init__(self, threshold: float = 0.6):
        """
        Initialiser le service
        Args:
            threshold: Seuil de similarité (distance). Plus petit = plus similaire.
        """
        self.threshold = threshold
        # Initialiser le détecteur de visage Haar Cascade (rapide)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def decode_base64_image(self, image_base64: str) -> Optional[np.ndarray]:
        """
        Décoder une image base64 en array numpy
        """
        try:
            # Retirer le préfixe data:image si présent
            if ',' in image_base64:
                image_base64 = image_base64.split(',')[1]
            
            # Décoder base64
            image_data = base64.b64decode(image_base64)
            
            # Convertir en numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            
            # Décoder l'image avec OpenCV
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Convertir BGR (OpenCV) en RGB (face_recognition)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            return image
        except Exception as e:
            logger.error(f"Erreur de décodage image: {e}")
            return None
    
    def _resize_image_for_speed(self, image: np.ndarray, max_width: int = 320) -> Tuple[np.ndarray, float]:
        """
        Redimensionner l'image pour accélérer le traitement
        Returns:
            Tuple (image redimensionnée, facteur de scale)
        """
        height, width = image.shape[:2]
        if width > max_width:
            scale = max_width / width
            new_height = int(height * scale)
            resized = cv2.resize(image, (max_width, new_height))
            return resized, scale
        return image, 1.0
    
    def extract_face_encoding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extraire le descripteur facial (encoding) d'une image
        Returns:
            Numpy array de 128 dimensions ou None si aucun visage détecté
        """
        try:
            # Redimensionner pour accélérer
            small_image, _ = self._resize_image_for_speed(image, max_width=480)
            
            # Détecter les visages avec le modèle HOG (plus rapide que CNN)
            face_locations = face_recognition.face_locations(small_image, model="hog")
            
            if len(face_locations) == 0:
                logger.warning("Aucun visage détecté dans l'image")
                return None
            
            if len(face_locations) > 1:
                logger.warning(f"Plusieurs visages détectés ({len(face_locations)}), utilisation du premier")
            
            # Extraire l'encoding du premier visage
            face_encodings = face_recognition.face_encodings(small_image, face_locations)
            
            if len(face_encodings) == 0:
                return None
            
            return face_encodings[0]
            
        except Exception as e:
            logger.error(f"Erreur d'extraction du descripteur facial: {e}")
            return None
    
    def encode_to_bytes(self, encoding: np.ndarray) -> bytes:
        """Convertir un encoding numpy en bytes pour stockage"""
        return encoding.tobytes()
    
    def decode_from_bytes(self, data: bytes) -> np.ndarray:
        """Reconvertir des bytes en encoding numpy"""
        return np.frombuffer(data, dtype=np.float64)
    
    def compare_faces(
        self,
        known_encoding: np.ndarray,
        unknown_encoding: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Comparer deux descripteurs faciaux
        Args:
            known_encoding: Encoding enregistré
            unknown_encoding: Encoding à vérifier
        Returns:
            Tuple (match: bool, score: float entre 0 et 1)
        """
        try:
            # Calculer la distance euclidienne
            distance = face_recognition.face_distance([known_encoding], unknown_encoding)[0]
            
            # Convertir la distance en score de similarité (0 à 1)
            # Distance 0 = identique, Distance > 0.6 = différent
            similarity_score = max(0, 1 - distance)
            
            # Vérifier si c'est un match
            is_match = distance <= self.threshold
            
            return is_match, similarity_score
            
        except Exception as e:
            logger.error(f"Erreur de comparaison faciale: {e}")
            return False, 0.0
    
    def enroll_face(self, image_base64: str) -> Tuple[Optional[bytes], float]:
        """
        Enrôler un visage à partir d'une image base64
        Returns:
            Tuple (encoding en bytes ou None, qualité)
        """
        # Décoder l'image
        image = self.decode_base64_image(image_base64)
        if image is None:
            return None, 0.0
        
        # Extraire l'encoding
        encoding = self.extract_face_encoding(image)
        if encoding is None:
            return None, 0.0
        
        # Évaluer la qualité (basée sur la détection)
        quality = 1.0  # Simplification - pourrait être amélioré
        
        return self.encode_to_bytes(encoding), quality
    
    def verify_face(
        self,
        stored_encoding_bytes: bytes,
        image_base64: str
    ) -> Tuple[bool, float]:
        """
        Vérifier un visage par rapport à l'encoding stocké
        Args:
            stored_encoding_bytes: Encoding stocké en bytes
            image_base64: Image à vérifier en base64
        Returns:
            Tuple (match: bool, score: float)
        """
        # Décoder l'image
        image = self.decode_base64_image(image_base64)
        if image is None:
            return False, 0.0
        
        # Extraire l'encoding
        unknown_encoding = self.extract_face_encoding(image)
        if unknown_encoding is None:
            return False, 0.0
        
        # Récupérer l'encoding stocké
        known_encoding = self.decode_from_bytes(stored_encoding_bytes)
        
        # Comparer
        return self.compare_faces(known_encoding, unknown_encoding)
    
    def detect_face_presence(self, image_base64: str) -> bool:
        """
        Détecter rapidement si un visage est présent dans l'image
        Utilise Haar Cascade (très rapide) au lieu de face_recognition
        """
        try:
            image = self.decode_base64_image(image_base64)
            if image is None:
                return False
            
            # Redimensionner pour accélérer
            small_image, _ = self._resize_image_for_speed(image, max_width=240)
            
            # Convertir en niveaux de gris pour Haar Cascade
            gray = cv2.cvtColor(small_image, cv2.COLOR_RGB2GRAY)
            
            # Détecter les visages avec Haar Cascade (très rapide)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(30, 30)
            )
            
            return len(faces) > 0
        except Exception as e:
            logger.error(f"Erreur de détection de présence: {e}")
            return False


# Instance globale du service
face_service = FaceRecognitionService()
