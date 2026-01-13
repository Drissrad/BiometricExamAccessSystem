"""
Routes de surveillance biométrique pendant l'examen
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime

from app.database import get_db
from app.schemas.biometric import (
    FaceCheckRequest, VoiceChallengeResponse, VoiceChallengeSubmit,
    SurveillanceStatus
)
from app.models.exam import ExamSession, SessionStatus
from app.models.user import User
from app.models.security_log import SecurityLog, LogType
from app.services.biometric_service import biometric_service
from app.services.voice_service import voice_service
from app.services.face_service import face_service
from app.routers.auth import get_current_user

router = APIRouter(prefix="/surveillance", tags=["Surveillance"])

# Nombre maximum de tentatives échouées TOTALES avant disqualification
MAX_FACE_FAILURES = 3  # 3 échecs sur le visage = triche
MAX_VOICE_FAILURES = 3  # 3 échecs sur la voix = triche

# Dictionnaire pour stocker les échecs totaux (session_id -> count)
total_face_failures = {}
total_voice_failures = {}


async def disqualify_session(session: ExamSession, db: AsyncSession, user_id: int, reason: str):
    """Disqualifier un candidat et mettre sa note à 0"""
    session.status = SessionStatus.DISQUALIFIED
    session.score = 0.0
    session.completed_at = datetime.utcnow()
    
    # Logger la triche
    log = SecurityLog(
        user_id=user_id,
        exam_session_id=session.id,
        log_type=LogType.CHEATING_DETECTED,
        message=f"Triche détectée: {reason}"
    )
    db.add(log)
    
    await db.commit()
    await db.refresh(session)
    
    # Nettoyer les compteurs
    if session.id in total_face_failures:
        del total_face_failures[session.id]
    if session.id in total_voice_failures:
        del total_voice_failures[session.id]


@router.post("/face-check/{session_id}")
async def check_face_during_exam(
    session_id: int,
    data: FaceCheckRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Vérification faciale pendant l'examen (surveillance continue)"""
    # Vérifier la session
    result = await db.execute(
        select(ExamSession).where(
            ExamSession.id == session_id,
            ExamSession.user_id == current_user.id,
            ExamSession.status == SessionStatus.IN_PROGRESS
        )
    )
    session = result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session d'examen non trouvée ou inactive"
        )
    
    # Vérifier d'abord si un visage est présent dans l'image
    has_face = face_service.detect_face_presence(data.image_base64)
    
    if not has_face:
        # Aucun visage détecté - incrémenter le compteur d'échecs TOTAUX
        total_face_failures[session_id] = total_face_failures.get(session_id, 0) + 1
        session.total_face_checks += 1
        session.anomaly_count += 1
        
        failures = total_face_failures[session_id]
        
        if failures >= MAX_FACE_FAILURES:
            await disqualify_session(
                session, db, current_user.id, 
                f"Aucun visage détecté {failures} fois au total"
            )
            return {
                "success": False,
                "score": 0.0,
                "message": "Vous êtes en train de tricher ! Examen terminé avec note 0.",
                "disqualified": True,
                "reason": "Aucun visage détecté"
            }
        
        await db.commit()
        return {
            "success": False,
            "score": 0.0,
            "message": f"⚠️ Aucun visage détecté ! Restez devant la caméra. Échecs: {failures}/{MAX_FACE_FAILURES}",
            "disqualified": False,
            "remaining_attempts": MAX_FACE_FAILURES - failures
        }
    
    # Un visage est détecté, effectuer la vérification d'identité avec anti-spoofing
    is_match, score, antispoof_result = await biometric_service.check_face(
        db,
        current_user.id,
        session_id,
        data.image_base64
    )
    
    # Vérifier si du spoofing a été détecté
    if antispoof_result.get("enabled") and not antispoof_result.get("passed"):
        # Spoofing détecté - disqualification immédiate
        await disqualify_session(
            session, db, current_user.id,
            f"Tentative de fraude biométrique détectée (photo/vidéo)"
        )
        return {
            "success": False,
            "score": 0.0,
            "message": "🚨 FRAUDE DÉTECTÉE ! Utilisation de photo/vidéo. Examen terminé.",
            "disqualified": True,
            "reason": "spoofing_detected",
            "antispoof_details": antispoof_result.get("details")
        }
    
    # Mettre à jour les statistiques
    session.total_face_checks += 1
    
    if is_match:
        # Succès - ne pas réinitialiser le compteur (échecs totaux)
        session.successful_face_checks += 1
        await db.commit()
        
        failures = total_face_failures.get(session_id, 0)
        return {
            "success": True,
            "score": score,
            "message": "Visage vérifié",
            "disqualified": False,
            "total_failures": failures,
            "max_failures": MAX_FACE_FAILURES
        }
    else:
        # Visage détecté mais pas reconnu - possiblement une autre personne
        total_face_failures[session_id] = total_face_failures.get(session_id, 0) + 1
        session.anomaly_count += 1
        
        failures = total_face_failures[session_id]
        
        if failures >= MAX_FACE_FAILURES:
            await disqualify_session(
                session, db, current_user.id, 
                f"Visage non reconnu {failures} fois au total"
            )
            return {
                "success": False,
                "score": score,
                "message": "Vous êtes en train de tricher ! Examen terminé avec note 0.",
                "disqualified": True,
                "reason": "Visage non reconnu - possiblement une autre personne"
            }
        
        await db.commit()
        return {
            "success": False,
            "score": score,
            "message": f"⚠️ Visage non reconnu. Échecs: {failures}/{MAX_FACE_FAILURES}",
            "disqualified": False,
            "remaining_attempts": MAX_FACE_FAILURES - failures
        }


@router.get("/voice-challenge/{session_id}", response_model=VoiceChallengeResponse)
async def get_voice_challenge(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Obtenir un défi vocal à lire"""
    # Vérifier la session
    result = await db.execute(
        select(ExamSession).where(
            ExamSession.id == session_id,
            ExamSession.user_id == current_user.id,
            ExamSession.status == SessionStatus.IN_PROGRESS
        )
    )
    session = result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session d'examen non trouvée ou inactive"
        )
    
    # Générer un défi
    challenge_id, text, expires_at = voice_service.generate_challenge(current_user.id)
    
    return VoiceChallengeResponse(
        challenge_id=challenge_id,
        text_to_read=text,
        expires_at=expires_at
    )


@router.post("/voice-check/{session_id}")
async def submit_voice_challenge(
    session_id: int,
    data: VoiceChallengeSubmit,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Soumettre le défi vocal pour vérification"""
    # Vérifier la session
    result = await db.execute(
        select(ExamSession).where(
            ExamSession.id == session_id,
            ExamSession.user_id == current_user.id,
            ExamSession.status == SessionStatus.IN_PROGRESS
        )
    )
    session = result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session d'examen non trouvée ou inactive"
        )
    
    # Valider le défi
    expected_text = voice_service.validate_challenge(data.challenge_id, current_user.id)
    if expected_text is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Défi expiré ou invalide"
        )
    
    # Effectuer la vérification vocale avec anti-spoofing
    is_match, score, antispoof_result = await biometric_service.check_voice(
        db,
        current_user.id,
        session_id,
        data.audio_base64
    )
    
    # Vérifier si du spoofing vocal a été détecté
    if antispoof_result.get("enabled") and not antispoof_result.get("passed"):
        # Spoofing vocal détecté - disqualification immédiate
        await disqualify_session(
            session, db, current_user.id,
            f"Tentative de fraude vocale détectée (enregistrement/synthèse)"
        )
        return {
            "success": False,
            "score": 0.0,
            "message": "🚨 FRAUDE VOCALE DÉTECTÉE ! Utilisation d'enregistrement. Examen terminé.",
            "disqualified": True,
            "reason": "voice_spoofing_detected",
            "antispoof_details": antispoof_result.get("details")
        }
    
    # Mettre à jour les statistiques
    session.total_voice_checks += 1
    
    if is_match:
        # Succès - ne pas réinitialiser le compteur (échecs totaux)
        session.successful_voice_checks += 1
        await db.commit()
        
        failures = total_voice_failures.get(session_id, 0)
        return {
            "success": True,
            "score": score,
            "message": "Voix vérifiée",
            "disqualified": False,
            "total_failures": failures,
            "max_failures": MAX_VOICE_FAILURES
        }
    else:
        # Incrémenter le compteur d'échecs TOTAUX
        total_voice_failures[session_id] = total_voice_failures.get(session_id, 0) + 1
        session.anomaly_count += 1
        
        failures = total_voice_failures[session_id]
        
        if failures >= MAX_VOICE_FAILURES:
            await disqualify_session(
                session, db, current_user.id,
                f"Voix non reconnue {failures} fois au total"
            )
            return {
                "success": False,
                "score": score,
                "message": "Vous êtes en train de tricher ! Examen terminé avec note 0.",
                "disqualified": True,
                "reason": "Voix non reconnue"
            }
        
        await db.commit()
        return {
            "success": False,
            "score": score,
            "message": f"⚠️ Voix non reconnue. Échecs: {failures}/{MAX_VOICE_FAILURES}",
            "disqualified": False,
            "remaining_attempts": MAX_VOICE_FAILURES - failures
        }


@router.post("/absence/{session_id}")
async def report_absence(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Signaler une absence détectée"""
    # Vérifier la session
    result = await db.execute(
        select(ExamSession).where(
            ExamSession.id == session_id,
            ExamSession.user_id == current_user.id,
            ExamSession.status == SessionStatus.IN_PROGRESS
        )
    )
    session = result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session d'examen non trouvée"
        )
    
    # Logger l'absence
    log = SecurityLog(
        user_id=current_user.id,
        exam_session_id=session_id,
        log_type=LogType.ABSENCE_DETECTED,
        message="Absence détectée - aucun visage visible"
    )
    db.add(log)
    
    session.anomaly_count += 1
    
    await db.commit()
    
    return {"message": "Absence signalée"}


@router.get("/status/{session_id}", response_model=SurveillanceStatus)
async def get_surveillance_status(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Obtenir le statut de surveillance d'une session"""
    # Vérifier la session
    result = await db.execute(
        select(ExamSession).where(ExamSession.id == session_id)
    )
    session = result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session non trouvée"
        )
    
    # Récupérer les derniers logs
    face_log_result = await db.execute(
        select(SecurityLog).where(
            SecurityLog.exam_session_id == session_id,
            SecurityLog.log_type.in_([LogType.FACE_CHECK_SUCCESS, LogType.FACE_CHECK_FAILED])
        ).order_by(SecurityLog.created_at.desc()).limit(1)
    )
    last_face_log = face_log_result.scalar_one_or_none()
    
    voice_log_result = await db.execute(
        select(SecurityLog).where(
            SecurityLog.exam_session_id == session_id,
            SecurityLog.log_type.in_([LogType.VOICE_CHECK_SUCCESS, LogType.VOICE_CHECK_FAILED])
        ).order_by(SecurityLog.created_at.desc()).limit(1)
    )
    last_voice_log = voice_log_result.scalar_one_or_none()
    
    return SurveillanceStatus(
        face_verified=last_face_log.log_type == LogType.FACE_CHECK_SUCCESS if last_face_log else False,
        voice_verified=last_voice_log.log_type == LogType.VOICE_CHECK_SUCCESS if last_voice_log else False,
        last_face_check=last_face_log.created_at if last_face_log else None,
        last_voice_check=last_voice_log.created_at if last_voice_log else None,
        anomalies_count=session.anomaly_count,
        is_active=session.status == SessionStatus.IN_PROGRESS
    )
