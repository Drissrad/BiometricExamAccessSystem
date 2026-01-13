"""
Service de biométrie multimodale
Combine la reconnaissance faciale et vocale
Inclut la détection anti-spoofing
"""
from typing import Optional, Tuple, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import base64
import io

from app.models.biometric import BiometricData
from app.models.user import User
from app.models.security_log import SecurityLog, LogType
from app.services.face_service import face_service
from app.services.voice_service import voice_service
from app.services.encryption_service import get_encryption_service
from app.services.antispoof_service import face_antispoof_service, voice_antispoof_service
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class BiometricService:
    """Service de biométrie multimodale avec anti-spoofing"""
    
    def __init__(self):
        self.face_weight = settings.FACE_WEIGHT
        self.voice_weight = settings.VOICE_WEIGHT
        self.threshold = settings.MULTIMODAL_THRESHOLD
        # Seuils minimaux individuels - protection contre les imposteurs
        self.min_face_score = settings.MIN_FACE_SCORE
        self.min_voice_score = settings.MIN_VOICE_SCORE
        # Anti-spoofing
        self.antispoof_enabled = settings.ANTISPOOF_ENABLED
        self.antispoof_min_face = settings.ANTISPOOF_MIN_FACE_SCORE
        self.antispoof_min_voice = settings.ANTISPOOF_MIN_VOICE_SCORE
    
    async def enroll_user(
        self,
        db: AsyncSession,
        user_id: int,
        face_image_base64: str,
        voice_audio_base64: str
    ) -> Tuple[bool, str]:
        """
        Enrôler les données biométriques d'un utilisateur
        """
        try:
            # Extraire le descripteur facial
            face_encoding, face_quality = face_service.enroll_face(face_image_base64)
            if face_encoding is None:
                return False, "Impossible de détecter un visage dans l'image"
            
            # Extraire le descripteur vocal
            voice_encoding, voice_quality = voice_service.enroll_voice(voice_audio_base64)
            if voice_encoding is None:
                return False, "Impossible d'extraire les caractéristiques vocales"
            
            # Chiffrer les données biométriques avant stockage
            encryption = get_encryption_service()
            encrypted_face = encryption.encrypt(face_encoding)
            encrypted_voice = encryption.encrypt(voice_encoding)
            logger.info("Données biométriques chiffrées avec AES-256")
            
            # Vérifier si l'utilisateur a déjà des données biométriques
            result = await db.execute(
                select(BiometricData).where(BiometricData.user_id == user_id)
            )
            existing = result.scalar_one_or_none()
            
            if existing:
                # Mettre à jour avec données chiffrées
                existing.face_encoding = encrypted_face
                existing.face_encoding_quality = face_quality
                existing.voice_encoding = encrypted_voice
                existing.voice_encoding_quality = voice_quality
            else:
                # Créer avec données chiffrées
                biometric = BiometricData(
                    user_id=user_id,
                    face_encoding=encrypted_face,
                    face_encoding_quality=face_quality,
                    voice_encoding=encrypted_voice,
                    voice_encoding_quality=voice_quality
                )
                db.add(biometric)
            
            # Marquer l'utilisateur comme enrôlé
            user_result = await db.execute(select(User).where(User.id == user_id))
            user = user_result.scalar_one_or_none()
            if user:
                user.is_enrolled = True
            
            # Logger l'enrôlement
            log = SecurityLog(
                user_id=user_id,
                log_type=LogType.ENROLLMENT_SUCCESS,
                message=f"Enrôlement biométrique réussi (face: {face_quality:.2f}, voice: {voice_quality:.2f})",
                face_score=face_quality,
                voice_score=voice_quality
            )
            db.add(log)
            
            await db.commit()
            
            return True, "Enrôlement biométrique réussi"
            
        except Exception as e:
            logger.error(f"Erreur lors de l'enrôlement: {e}")
            return False, f"Erreur lors de l'enrôlement: {str(e)}"
    
    async def verify_user(
        self,
        db: AsyncSession,
        user_id: int,
        face_image_base64: Optional[str] = None,
        voice_audio_base64: Optional[str] = None
    ) -> Tuple[bool, float, float, float, str, Dict[str, Any]]:
        """
        Vérifier l'identité d'un utilisateur avec anti-spoofing
        Returns:
            Tuple (is_verified, face_score, voice_score, combined_score, message, antispoof_results)
        """
        antispoof_results = {
            "enabled": self.antispoof_enabled,
            "face_liveness": None,
            "voice_liveness": None,
            "face_passed": True,
            "voice_passed": True,
            "details": {}
        }
        
        try:
            # ==================== ANTI-SPOOFING CHECKS ====================
            if self.antispoof_enabled:
                logger.info("=== VÉRIFICATION ANTI-SPOOFING ===")
                
                # Anti-spoofing facial
                if face_image_base64:
                    is_live, liveness_score, details = face_antispoof_service.check_liveness(face_image_base64)
                    face_liveness = {
                        "is_live": is_live,
                        "liveness_score": liveness_score,
                        "details": details,
                        "reason": "Image authentique" if is_live else "Spoofing facial détecté"
                    }
                    antispoof_results["face_liveness"] = face_liveness
                    antispoof_results["details"]["face"] = face_liveness
                    
                    if not is_live:
                        antispoof_results["face_passed"] = False
                        logger.warning(f"❌ ANTI-SPOOF FACE: Score {liveness_score:.2f} < {self.antispoof_min_face}")
                        logger.warning(f"   Détails: {details}")
                        
                        # Logger la tentative de spoofing
                        log = SecurityLog(
                            user_id=user_id,
                            log_type=LogType.SPOOFING_DETECTED,
                            message=f"Tentative de spoofing facial détectée (score: {liveness_score:.2f})",
                            face_score=liveness_score
                        )
                        db.add(log)
                        await db.commit()
                        
                        return False, 0.0, 0.0, 0.0, f"Fraude détectée: Spoofing facial", antispoof_results
                    else:
                        logger.info(f"✅ ANTI-SPOOF FACE: Score {liveness_score:.2f}")
                
                # Anti-spoofing vocal
                if voice_audio_base64:
                    # Utiliser le décodeur audio du voice_service qui gère tous les formats
                    try:
                        audio_result = voice_service.decode_base64_audio(voice_audio_base64)
                        if audio_result is None:
                            raise Exception("Impossible de décoder l'audio")
                        audio, sr = audio_result
                        is_live, liveness_score, details = voice_antispoof_service.check_liveness(audio, sr)
                    except Exception as e:
                        logger.warning(f"Erreur anti-spoofing vocal: {e}")
                        # MODE STRICT: Rejeter si on ne peut pas analyser
                        if settings.ANTISPOOF_STRICT_MODE:
                            logger.error("❌ MODE STRICT: Rejet car anti-spoofing vocal impossible")
                            return False, 0.0, 0.0, 0.0, "Erreur analyse audio - réessayez", antispoof_results
                        else:
                            is_live, liveness_score, details = True, 1.0, {"skipped": str(e)}
                    
                    voice_liveness = {
                        "is_live": is_live,
                        "liveness_score": liveness_score,
                        "details": details,
                        "reason": "Audio authentique" if is_live else "Spoofing vocal détecté"
                    }
                    antispoof_results["voice_liveness"] = voice_liveness
                    antispoof_results["details"]["voice"] = voice_liveness
                    
                    if not is_live:
                        antispoof_results["voice_passed"] = False
                        logger.warning(f"❌ ANTI-SPOOF VOICE: Score {liveness_score:.2f} < {self.antispoof_min_voice}")
                        logger.warning(f"   Détails: {details}")
                        
                        # Logger la tentative de spoofing
                        log = SecurityLog(
                            user_id=user_id,
                            log_type=LogType.SPOOFING_DETECTED,
                            message=f"Tentative de spoofing vocal détectée (score: {liveness_score:.2f})",
                            voice_score=liveness_score
                        )
                        db.add(log)
                        await db.commit()
                        
                        return False, 0.0, 0.0, 0.0, f"Fraude détectée: Spoofing vocal", antispoof_results
                    else:
                        logger.info(f"✅ ANTI-SPOOF VOICE: Score {liveness_score:.2f}")
            
            # ==================== VÉRIFICATION BIOMÉTRIQUE ====================
            # Récupérer les données biométriques stockées
            result = await db.execute(
                select(BiometricData).where(BiometricData.user_id == user_id)
            )
            biometric = result.scalar_one_or_none()
            
            if biometric is None:
                return False, 0.0, 0.0, 0.0, "Utilisateur non enrôlé", antispoof_results
            
            face_score = 0.0
            voice_score = 0.0
            
            logger.info(f"=== VÉRIFICATION BIOMÉTRIQUE pour user_id={user_id} ===")
            logger.info(f"Face image fournie: {bool(face_image_base64)}, Voice audio fourni: {bool(voice_audio_base64)}")
            logger.info(f"Face encoding stocké: {bool(biometric.face_encoding)}, Voice encoding stocké: {bool(biometric.voice_encoding)}")
            
            # Déchiffrer les données biométriques stockées
            encryption = get_encryption_service()
            
            # Vérification faciale
            if face_image_base64 and biometric.face_encoding:
                # Déchiffrer l'encoding facial
                decrypted_face = encryption.decrypt(biometric.face_encoding)
                _, face_score = face_service.verify_face(
                    decrypted_face,
                    face_image_base64
                )
                logger.info(f"Score facial: {face_score:.4f}")
            else:
                logger.warning("Vérification faciale NON effectuée - données manquantes")
            
            # Vérification vocale
            if voice_audio_base64 and biometric.voice_encoding:
                # Déchiffrer l'encoding vocal
                decrypted_voice = encryption.decrypt(biometric.voice_encoding)
                _, voice_score = voice_service.verify_voice(
                    decrypted_voice,
                    voice_audio_base64
                )
                logger.info(f"Score vocal: {voice_score:.4f}")
            else:
                logger.warning("Vérification vocale NON effectuée - données manquantes")
            
            # Calcul du score combiné (fusion multimodale)
            if face_image_base64 and voice_audio_base64:
                # Les deux modalités présentes
                combined_score = (
                    self.face_weight * face_score +
                    self.voice_weight * voice_score
                )
            elif face_image_base64:
                # Seulement le visage
                combined_score = face_score
            elif voice_audio_base64:
                # Seulement la voix
                combined_score = voice_score
            else:
                return False, 0.0, 0.0, 0.0, "Aucune donnée biométrique fournie", antispoof_results
            
            logger.info(f"Score combiné: {combined_score:.4f} (seuil multimodal: {self.threshold})")
            
            # Décision - IMPORTANT: Chaque modalité doit atteindre son seuil minimum
            # Cela empêche un imposteur avec une voix similaire (ex: frère) de passer
            individual_checks_passed = True
            rejection_reason = ""
            
            if face_image_base64:
                logger.info(f"Vérification seuil facial: {face_score:.4f} >= {self.min_face_score} ?")
                if face_score < self.min_face_score:
                    individual_checks_passed = False
                    rejection_reason = f"Score facial insuffisant ({face_score:.2f} < {self.min_face_score})"
                    logger.warning(f"❌ REJET: {rejection_reason}")
            
            if voice_audio_base64:
                logger.info(f"Vérification seuil vocal: {voice_score:.4f} >= {self.min_voice_score} ?")
                if voice_score < self.min_voice_score:
                    individual_checks_passed = False
                    rejection_reason = f"Score vocal insuffisant ({voice_score:.2f} < {self.min_voice_score}). Voix non reconnue comme appartenant à l'utilisateur."
                    logger.warning(f"❌ REJET: {rejection_reason}")
            
            # Les deux conditions doivent être vraies:
            # 1. Chaque modalité fournie doit passer son seuil individuel
            # 2. Le score combiné doit atteindre le seuil multimodal
            is_verified = individual_checks_passed and (combined_score >= self.threshold)
            logger.info(f"=== RÉSULTAT: {'✅ ACCEPTÉ' if is_verified else '❌ REFUSÉ'} ===")
            
            # Logger la vérification
            log_type = LogType.LOGIN_SUCCESS if is_verified else LogType.LOGIN_FAILED
            log = SecurityLog(
                user_id=user_id,
                log_type=log_type,
                message=f"Vérification biométrique: {'réussie' if is_verified else 'échouée'}",
                face_score=face_score,
                voice_score=voice_score,
                combined_score=combined_score
            )
            db.add(log)
            await db.commit()
            
            # Message détaillé pour l'utilisateur
            if is_verified:
                message = "Vérification réussie"
            elif rejection_reason:
                message = f"Vérification échouée: {rejection_reason}"
            else:
                message = f"Vérification échouée: score combiné insuffisant ({combined_score:.2f} < {self.threshold})"
            
            return is_verified, face_score, voice_score, combined_score, message, antispoof_results
            
        except Exception as e:
            logger.error(f"Erreur lors de la vérification: {e}")
            return False, 0.0, 0.0, 0.0, f"Erreur: {str(e)}", antispoof_results
    
    async def check_face(
        self,
        db: AsyncSession,
        user_id: int,
        exam_session_id: int,
        face_image_base64: str
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Vérification faciale pendant l'examen (surveillance continue) avec anti-spoofing
        Returns: (is_match, score, antispoof_result)
        """
        antispoof_result = {"enabled": self.antispoof_enabled, "passed": True, "details": None}
        
        try:
            # ===== ANTI-SPOOFING CHECK =====
            if self.antispoof_enabled:
                # check_liveness retourne un tuple (is_live, score, details)
                is_live, liveness_score, liveness_details = face_antispoof_service.check_liveness(face_image_base64)
                face_liveness = {
                    "is_live": is_live,
                    "liveness_score": liveness_score,
                    "reason": liveness_details.get("reason", "unknown") if isinstance(liveness_details, dict) else "unknown"
                }
                antispoof_result["details"] = face_liveness
                
                if not is_live:
                    antispoof_result["passed"] = False
                    logger.warning(f"❌ SPOOFING DÉTECTÉ (exam_session={exam_session_id}): {face_liveness['reason']}")
                    
                    # Logger la tentative de spoofing
                    log = SecurityLog(
                        user_id=user_id,
                        exam_session_id=exam_session_id,
                        log_type=LogType.SPOOFING_DETECTED,
                        message=f"Spoofing facial pendant examen: {face_liveness['reason']}",
                        face_score=liveness_score
                    )
                    db.add(log)
                    await db.commit()
                    
                    return False, 0.0, antispoof_result
            
            # ===== VÉRIFICATION BIOMÉTRIQUE =====
            # Récupérer les données biométriques
            result = await db.execute(
                select(BiometricData).where(BiometricData.user_id == user_id)
            )
            biometric = result.scalar_one_or_none()
            
            if biometric is None or biometric.face_encoding is None:
                return False, 0.0, antispoof_result
            
            # Déchiffrer l'encoding facial
            encryption = get_encryption_service()
            decrypted_face = encryption.decrypt(biometric.face_encoding)
            
            # Vérifier le visage
            is_match, score = face_service.verify_face(
                decrypted_face,
                face_image_base64
            )
            
            # Logger
            log_type = LogType.FACE_CHECK_SUCCESS if is_match else LogType.FACE_CHECK_FAILED
            log = SecurityLog(
                user_id=user_id,
                exam_session_id=exam_session_id,
                log_type=log_type,
                message=f"Vérification faciale: score={score:.2f}",
                face_score=score
            )
            db.add(log)
            await db.commit()
            
            return is_match, score, antispoof_result
            
        except Exception as e:
            logger.error(f"Erreur vérification faciale: {e}")
            return False, 0.0, antispoof_result
    
    async def check_voice(
        self,
        db: AsyncSession,
        user_id: int,
        exam_session_id: int,
        voice_audio_base64: str
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Vérification vocale pendant l'examen (surveillance continue) avec anti-spoofing
        Returns: (is_match, score, antispoof_result)
        """
        antispoof_result = {"enabled": self.antispoof_enabled, "passed": True, "details": None}
        
        try:
            # ===== ANTI-SPOOFING CHECK =====
            if self.antispoof_enabled:
                # Utiliser le décodeur du voice_service qui gère tous les formats
                try:
                    audio_result = voice_service.decode_base64_audio(voice_audio_base64)
                    if audio_result is None:
                        raise Exception("Impossible de décoder l'audio")
                    audio_data, sr = audio_result
                    
                    # check_liveness retourne un tuple (is_live, score, details)
                    is_live, liveness_score, liveness_details = voice_antispoof_service.check_liveness(audio_data, sr)
                    voice_liveness = {
                        "is_live": is_live,
                        "liveness_score": liveness_score,
                        "reason": liveness_details.get("reason", "unknown") if isinstance(liveness_details, dict) else "unknown"
                    }
                    antispoof_result["details"] = voice_liveness
                    
                    if not is_live:
                        antispoof_result["passed"] = False
                        logger.warning(f"❌ SPOOFING VOCAL DÉTECTÉ (exam_session={exam_session_id}): {voice_liveness['reason']}")
                        
                        # Logger la tentative de spoofing
                        log = SecurityLog(
                            user_id=user_id,
                            exam_session_id=exam_session_id,
                            log_type=LogType.SPOOFING_DETECTED,
                            message=f"Spoofing vocal pendant examen: {voice_liveness['reason']}",
                            voice_score=liveness_score
                        )
                        db.add(log)
                        await db.commit()
                        
                        return False, 0.0, antispoof_result
                except Exception as e:
                    logger.warning(f"Erreur anti-spoofing vocal: {e}")
                    if settings.ANTISPOOF_STRICT_MODE:
                        antispoof_result["passed"] = False
                        return False, 0.0, antispoof_result
            
            # ===== VÉRIFICATION BIOMÉTRIQUE =====
            # Récupérer les données biométriques
            result = await db.execute(
                select(BiometricData).where(BiometricData.user_id == user_id)
            )
            biometric = result.scalar_one_or_none()
            
            if biometric is None or biometric.voice_encoding is None:
                return False, 0.0, antispoof_result
            
            # Déchiffrer l'encoding vocal
            encryption = get_encryption_service()
            decrypted_voice = encryption.decrypt(biometric.voice_encoding)
            
            # Vérifier la voix
            is_match, score = voice_service.verify_voice(
                decrypted_voice,
                voice_audio_base64
            )
            
            # Logger
            log_type = LogType.VOICE_CHECK_SUCCESS if is_match else LogType.VOICE_CHECK_FAILED
            log = SecurityLog(
                user_id=user_id,
                exam_session_id=exam_session_id,
                log_type=log_type,
                message=f"Vérification vocale: score={score:.2f}",
                voice_score=score
            )
            db.add(log)
            await db.commit()
            
            return is_match, score, antispoof_result
            
        except Exception as e:
            logger.error(f"Erreur vérification vocale: {e}")
            return False, 0.0, antispoof_result


# Instance globale
biometric_service = BiometricService()
