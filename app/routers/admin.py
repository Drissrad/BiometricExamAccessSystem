"""
Routes d'administration
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, EmailStr

from app.database import get_db
from app.schemas.user import UserResponse, UserUpdate
from app.models.user import User, UserRole
from app.models.exam import Exam, ExamSession, SessionStatus
from app.models.security_log import SecurityLog, LogType
from app.models.biometric import BiometricData
from app.routers.auth import get_current_admin
from app.services.auth_service import get_password_hash
from app.services.biometric_service import biometric_service


class AssignCandidatRequest(BaseModel):
    """Requête pour assigner un candidat à un examen"""
    user_id: int
    exam_id: int


class CreateCandidatWithBiometricRequest(BaseModel):
    """Requête pour créer un candidat avec données biométriques"""
    nom: str
    prenom: str
    email: EmailStr
    password: str
    face_image_base64: str
    voice_audio_base64: str


class ImpostorTestRequest(BaseModel):
    """Requête pour tester un imposteur (FAR test)"""
    target_user_id: int  # L'utilisateur dont on usurpe l'identité
    face_image_base64: Optional[str] = None  # Visage de l'imposteur
    voice_audio_base64: Optional[str] = None  # Voix de l'imposteur


router = APIRouter(prefix="/admin", tags=["Administration"])


@router.post("/create-candidate", response_model=UserResponse)
async def create_candidate_with_biometric(
    data: CreateCandidatWithBiometricRequest,
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Créer un candidat avec ses données biométriques (visage + voix)"""
    # Vérifier si l'email existe déjà
    result = await db.execute(select(User).where(User.email == data.email))
    existing = result.scalar_one_or_none()
    
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cet email est déjà utilisé"
        )
    
    # Créer l'utilisateur
    user = User(
        email=data.email,
        hashed_password=get_password_hash(data.password),
        nom=data.nom,
        prenom=data.prenom,
        role=UserRole.CANDIDAT,
        is_active=True,
        is_enrolled=False  # Sera mis à True après l'enrôlement
    )
    db.add(user)
    await db.flush()  # Pour obtenir l'ID
    
    # Enrôler les données biométriques
    success, message = await biometric_service.enroll_user(
        db,
        user.id,
        data.face_image_base64,
        data.voice_audio_base64
    )
    
    if not success:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Erreur lors de l'enrôlement biométrique: {message}"
        )
    
    await db.commit()
    await db.refresh(user)
    
    return user


@router.get("/users", response_model=List[UserResponse])
async def list_users(
    role: Optional[UserRole] = None,
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Lister tous les utilisateurs"""
    query = select(User)
    if role:
        query = query.where(User.role == role)
    
    result = await db.execute(query)
    return result.scalars().all()


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Récupérer un utilisateur"""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Utilisateur non trouvé"
        )
    
    return user


@router.put("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    user_data: UserUpdate,
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Mettre à jour un utilisateur"""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Utilisateur non trouvé"
        )
    
    if user_data.nom is not None:
        user.nom = user_data.nom
    if user_data.prenom is not None:
        user.prenom = user_data.prenom
    if user_data.is_active is not None:
        user.is_active = user_data.is_active
    
    await db.commit()
    await db.refresh(user)
    
    return user


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Supprimer un utilisateur"""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Utilisateur non trouvé"
        )
    
    if user.id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Vous ne pouvez pas vous supprimer vous-même"
        )
    
    # Supprimer les données biométriques associées
    biometric_result = await db.execute(
        select(BiometricData).where(BiometricData.user_id == user_id)
    )
    biometric = biometric_result.scalar_one_or_none()
    if biometric:
        await db.delete(biometric)
    
    # Supprimer les logs de sécurité associés
    logs_result = await db.execute(
        select(SecurityLog).where(SecurityLog.user_id == user_id)
    )
    for log in logs_result.scalars().all():
        await db.delete(log)
    
    # Supprimer les sessions d'examen associées
    sessions_result = await db.execute(
        select(ExamSession).where(ExamSession.user_id == user_id)
    )
    for session in sessions_result.scalars().all():
        await db.delete(session)
    
    # Supprimer l'utilisateur
    await db.delete(user)
    await db.commit()
    
    return {"message": f"Utilisateur {user.prenom} {user.nom} et toutes ses données supprimés"}


@router.post("/assign-candidate")
async def assign_candidate_to_exam(
    data: AssignCandidatRequest,
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Assigner un candidat à un examen"""
    # Vérifier que l'utilisateur existe
    result = await db.execute(select(User).where(User.id == data.user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Utilisateur non trouvé"
        )
    
    if user.role != UserRole.CANDIDAT:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Seuls les candidats peuvent être assignés à des examens"
        )
    
    # Vérifier que l'examen existe
    result = await db.execute(select(Exam).where(Exam.id == data.exam_id))
    exam = result.scalar_one_or_none()
    
    if not exam:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Examen non trouvé"
        )
    
    # Vérifier si le candidat est déjà assigné
    result = await db.execute(
        select(ExamSession).where(
            ExamSession.user_id == data.user_id,
            ExamSession.exam_id == data.exam_id
        )
    )
    existing = result.scalar_one_or_none()
    
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Ce candidat est déjà assigné à cet examen"
        )
    
    # Créer la session d'examen
    session = ExamSession(
        user_id=data.user_id,
        exam_id=data.exam_id,
        status=SessionStatus.PENDING
    )
    
    db.add(session)
    await db.commit()
    await db.refresh(session)
    
    return {
        "message": f"Candidat {user.prenom} {user.nom} assigné à l'examen {exam.title}",
        "session_id": session.id
    }


@router.get("/exam-sessions/{exam_id}")
async def get_exam_sessions(
    exam_id: int,
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Récupérer les sessions d'un examen"""
    result = await db.execute(
        select(ExamSession, User)
        .join(User, ExamSession.user_id == User.id)
        .where(ExamSession.exam_id == exam_id)
    )
    
    sessions = []
    for session, user in result:
        sessions.append({
            "session_id": session.id,
            "user_id": user.id,
            "user_name": f"{user.prenom} {user.nom}",
            "user_email": user.email,
            "status": session.status.value,
            "score": session.score,
            "started_at": session.started_at.isoformat() if session.started_at else None,
            "completed_at": session.completed_at.isoformat() if session.completed_at else None,
            "anomaly_count": session.anomaly_count
        })
    
    return sessions


@router.get("/security-logs")
async def get_security_logs(
    user_id: Optional[int] = None,
    log_type: Optional[LogType] = None,
    limit: int = 100,
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Récupérer les journaux de sécurité"""
    query = select(SecurityLog)
    
    if user_id:
        query = query.where(SecurityLog.user_id == user_id)
    if log_type:
        query = query.where(SecurityLog.log_type == log_type)
    
    query = query.order_by(SecurityLog.created_at.desc()).limit(limit)
    
    result = await db.execute(query)
    logs = result.scalars().all()
    
    return [
        {
            "id": log.id,
            "user_id": log.user_id,
            "exam_session_id": log.exam_session_id,
            "log_type": log.log_type.value,
            "message": log.message,
            "face_score": log.face_score,
            "voice_score": log.voice_score,
            "combined_score": log.combined_score,
            "ip_address": log.ip_address,
            "created_at": log.created_at.isoformat()
        }
        for log in logs
    ]


@router.get("/statistics")
async def get_statistics(
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Obtenir les statistiques générales"""
    # Nombre d'utilisateurs
    users_count = await db.execute(select(func.count(User.id)))
    total_users = users_count.scalar()
    
    # Candidats enrôlés
    enrolled_count = await db.execute(
        select(func.count(User.id)).where(User.is_enrolled == True)
    )
    total_enrolled = enrolled_count.scalar()
    
    # Examens
    exams_count = await db.execute(select(func.count(Exam.id)))
    total_exams = exams_count.scalar()
    
    # Sessions complétées
    sessions_count = await db.execute(
        select(func.count(ExamSession.id)).where(
            ExamSession.status == SessionStatus.COMPLETED
        )
    )
    completed_sessions = sessions_count.scalar()
    
    # Score moyen
    avg_score = await db.execute(
        select(func.avg(ExamSession.score)).where(
            ExamSession.status == SessionStatus.COMPLETED
        )
    )
    average_score = avg_score.scalar() or 0
    
    # Anomalies totales
    anomalies = await db.execute(
        select(func.sum(ExamSession.anomaly_count))
    )
    total_anomalies = anomalies.scalar() or 0
    
    # Statistiques de la semaine
    week_ago = datetime.utcnow() - timedelta(days=7)
    
    weekly_sessions = await db.execute(
        select(func.count(ExamSession.id)).where(
            ExamSession.created_at >= week_ago
        )
    )
    sessions_this_week = weekly_sessions.scalar()
    
    weekly_logs = await db.execute(
        select(func.count(SecurityLog.id)).where(
            SecurityLog.created_at >= week_ago,
            SecurityLog.log_type.in_([
                LogType.FACE_CHECK_FAILED,
                LogType.VOICE_CHECK_FAILED,
                LogType.ABSENCE_DETECTED
            ])
        )
    )
    alerts_this_week = weekly_logs.scalar()
    
    return {
        "total_users": total_users,
        "total_enrolled": total_enrolled,
        "total_exams": total_exams,
        "completed_sessions": completed_sessions,
        "average_score": round(average_score, 2),
        "total_anomalies": total_anomalies,
        "sessions_this_week": sessions_this_week,
        "alerts_this_week": alerts_this_week
    }


@router.post("/test-impostor")
async def test_impostor(
    data: ImpostorTestRequest,
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Tester un imposteur pour calculer le FAR (False Acceptance Rate)
    
    L'admin fournit des données biométriques (visage/voix) et on vérifie
    si le système les accepte à la place de l'utilisateur cible.
    
    - Si accepté = IMPOSTOR_ATTEMPT (faux positif - mauvais pour la sécurité)
    - Si rejeté = IMPOSTOR_REJECTED (vrai négatif - bon pour la sécurité)
    """
    # Vérifier que l'utilisateur cible existe et est enrôlé
    result = await db.execute(select(User).where(User.id == data.target_user_id))
    target_user = result.scalar_one_or_none()
    
    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Utilisateur cible non trouvé"
        )
    
    if not target_user.is_enrolled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="L'utilisateur cible n'est pas enrôlé"
        )
    
    if not data.face_image_base64 and not data.voice_audio_base64:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Fournissez au moins une donnée biométrique (visage ou voix)"
        )
    
    # Effectuer la vérification biométrique avec les données de l'imposteur (anti-spoofing inclus)
    is_accepted, face_score, voice_score, combined_score, message, antispoof = await biometric_service.verify_user(
        db,
        data.target_user_id,
        data.face_image_base64,
        data.voice_audio_base64
    )
    
    # Enregistrer le résultat du test
    if is_accepted:
        # MAUVAIS : L'imposteur a été accepté (faux positif)
        log_type = LogType.IMPOSTOR_ATTEMPT
        result_message = "⚠️ ALERTE: L'imposteur a été ACCEPTÉ! (Faux positif - FAR)"
    else:
        # BON : L'imposteur a été rejeté (vrai négatif)
        log_type = LogType.IMPOSTOR_REJECTED
        result_message = "✅ L'imposteur a été correctement REJETÉ (Vrai négatif)"
        
        # Si c'est du spoofing, ajouter l'info
        if antispoof.get("enabled") and (not antispoof.get("face_passed") or not antispoof.get("voice_passed")):
            result_message += " (Spoofing détecté)"
    
    # Logger le test
    log = SecurityLog(
        user_id=data.target_user_id,
        log_type=log_type,
        message=f"Test imposteur sur {target_user.prenom} {target_user.nom}: {'accepté' if is_accepted else 'rejeté'}",
        face_score=face_score,
        voice_score=voice_score,
        combined_score=combined_score
    )
    db.add(log)
    await db.commit()
    
    return {
        "success": True,
        "impostor_accepted": is_accepted,
        "result": result_message,
        "target_user": f"{target_user.prenom} {target_user.nom}",
        "scores": {
            "face": round(face_score, 4),
            "voice": round(voice_score, 4),
            "combined": round(combined_score, 4)
        },
        "thresholds": {
            "face_min": 0.5,
            "voice_min": 0.55,
            "combined_min": 0.65
        },
        "log_type": log_type.value
    }


@router.get("/enrolled-users")
async def get_enrolled_users(
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Récupérer la liste des utilisateurs enrôlés (pour les tests d'imposteurs)"""
    result = await db.execute(
        select(User).where(User.is_enrolled == True)
    )
    users = result.scalars().all()
    
    return [
        {
            "id": user.id,
            "nom": user.nom,
            "prenom": user.prenom,
            "email": user.email
        }
        for user in users
    ]


@router.get("/biometric-metrics")
async def get_biometric_metrics(
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Obtenir les métriques biométriques FAR/FRR
    
    FAR (False Acceptance Rate) = Imposteurs acceptés / Total tentatives imposteurs
    FRR (False Rejection Rate) = Légitimes rejetés / Total tentatives légitimes
    """
    # Statistiques faciales
    face_success = await db.execute(
        select(func.count(SecurityLog.id)).where(
            SecurityLog.log_type == LogType.FACE_CHECK_SUCCESS
        )
    )
    face_success_count = face_success.scalar() or 0
    
    face_failed = await db.execute(
        select(func.count(SecurityLog.id)).where(
            SecurityLog.log_type == LogType.FACE_CHECK_FAILED
        )
    )
    face_failed_count = face_failed.scalar() or 0
    
    # Statistiques vocales
    voice_success = await db.execute(
        select(func.count(SecurityLog.id)).where(
            SecurityLog.log_type == LogType.VOICE_CHECK_SUCCESS
        )
    )
    voice_success_count = voice_success.scalar() or 0
    
    voice_failed = await db.execute(
        select(func.count(SecurityLog.id)).where(
            SecurityLog.log_type == LogType.VOICE_CHECK_FAILED
        )
    )
    voice_failed_count = voice_failed.scalar() or 0
    
    # Login stats (pour FAR/FRR globaux)
    login_success = await db.execute(
        select(func.count(SecurityLog.id)).where(
            SecurityLog.log_type == LogType.LOGIN_SUCCESS
        )
    )
    login_success_count = login_success.scalar() or 0
    
    login_failed = await db.execute(
        select(func.count(SecurityLog.id)).where(
            SecurityLog.log_type == LogType.LOGIN_FAILED
        )
    )
    login_failed_count = login_failed.scalar() or 0
    
    # Scores moyens
    avg_face_success = await db.execute(
        select(func.avg(SecurityLog.face_score)).where(
            SecurityLog.log_type == LogType.FACE_CHECK_SUCCESS,
            SecurityLog.face_score.isnot(None)
        )
    )
    avg_face_success_score = avg_face_success.scalar() or 0
    
    avg_face_failed = await db.execute(
        select(func.avg(SecurityLog.face_score)).where(
            SecurityLog.log_type == LogType.FACE_CHECK_FAILED,
            SecurityLog.face_score.isnot(None)
        )
    )
    avg_face_failed_score = avg_face_failed.scalar() or 0
    
    avg_voice_success = await db.execute(
        select(func.avg(SecurityLog.voice_score)).where(
            SecurityLog.log_type == LogType.VOICE_CHECK_SUCCESS,
            SecurityLog.voice_score.isnot(None)
        )
    )
    avg_voice_success_score = avg_voice_success.scalar() or 0
    
    avg_voice_failed = await db.execute(
        select(func.avg(SecurityLog.voice_score)).where(
            SecurityLog.log_type == LogType.VOICE_CHECK_FAILED,
            SecurityLog.voice_score.isnot(None)
        )
    )
    avg_voice_failed_score = avg_voice_failed.scalar() or 0
    
    # Statistiques FAR (tentatives d'imposteurs)
    impostor_attempts = await db.execute(
        select(func.count(SecurityLog.id)).where(
            SecurityLog.log_type == LogType.IMPOSTOR_ATTEMPT
        )
    )
    impostor_attempts_count = impostor_attempts.scalar() or 0
    
    impostor_rejected = await db.execute(
        select(func.count(SecurityLog.id)).where(
            SecurityLog.log_type == LogType.IMPOSTOR_REJECTED
        )
    )
    impostor_rejected_count = impostor_rejected.scalar() or 0
    
    # Calcul des taux
    total_face = face_success_count + face_failed_count
    total_voice = voice_success_count + voice_failed_count
    total_login = login_success_count + login_failed_count
    total_impostor = impostor_attempts_count + impostor_rejected_count
    
    # FRR estimé = échecs / total (approximation basée sur l'hypothèse que 
    # la majorité des tentatives sont légitimes)
    face_frr = (face_failed_count / total_face * 100) if total_face > 0 else 0
    voice_frr = (voice_failed_count / total_voice * 100) if total_voice > 0 else 0
    login_frr = (login_failed_count / total_login * 100) if total_login > 0 else 0
    
    # FAR = imposteurs acceptés / total tentatives imposteurs
    # IMPOSTOR_ATTEMPT = imposteur accepté (faux positif - mauvais)
    # IMPOSTOR_REJECTED = imposteur rejeté (vrai négatif - bon)
    far = (impostor_attempts_count / total_impostor * 100) if total_impostor > 0 else 0
    
    # Taux de réussite
    face_success_rate = (face_success_count / total_face * 100) if total_face > 0 else 0
    voice_success_rate = (voice_success_count / total_voice * 100) if total_voice > 0 else 0
    
    return {
        "face": {
            "total_checks": total_face,
            "success": face_success_count,
            "failed": face_failed_count,
            "success_rate": round(face_success_rate, 2),
            "frr_estimate": round(face_frr, 2),
            "avg_success_score": round(avg_face_success_score, 3),
            "avg_failed_score": round(avg_face_failed_score, 3)
        },
        "voice": {
            "total_checks": total_voice,
            "success": voice_success_count,
            "failed": voice_failed_count,
            "success_rate": round(voice_success_rate, 2),
            "frr_estimate": round(voice_frr, 2),
            "avg_success_score": round(avg_voice_success_score, 3),
            "avg_failed_score": round(avg_voice_failed_score, 3)
        },
        "login": {
            "total_attempts": total_login,
            "success": login_success_count,
            "failed": login_failed_count,
            "frr_estimate": round(login_frr, 2)
        },
        "far": {
            "total_impostor_tests": total_impostor,
            "impostor_accepted": impostor_attempts_count,
            "impostor_rejected": impostor_rejected_count,
            "far_rate": round(far, 2)
        },
        "thresholds": {
            "face": 0.5,
            "voice": 0.55,
            "multimodal": 0.65
        },
        "explanation": {
            "frr": "False Rejection Rate - % d'utilisateurs légitimes rejetés",
            "far": "False Acceptance Rate - % d'imposteurs acceptés (nécessite tests avec imposteurs)",
            "note": "Les valeurs FRR sont estimées. Pour FAR précis, des tests avec imposteurs connus sont nécessaires."
        }
    }


@router.get("/antispoof-stats")
async def get_antispoof_statistics(
    current_admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Statistiques des détections anti-spoofing (fraude biométrique)"""
    from app.config import settings
    
    # Compter les tentatives de spoofing détectées
    spoofing_result = await db.execute(
        select(func.count(SecurityLog.id)).where(
            SecurityLog.log_type == LogType.SPOOFING_DETECTED
        )
    )
    spoofing_count = spoofing_result.scalar() or 0
    
    # Détails des dernières tentatives de spoofing
    recent_spoofing = await db.execute(
        select(SecurityLog, User).join(
            User, SecurityLog.user_id == User.id, isouter=True
        ).where(
            SecurityLog.log_type == LogType.SPOOFING_DETECTED
        ).order_by(SecurityLog.created_at.desc()).limit(20)
    )
    
    spoofing_logs = []
    for log, user in recent_spoofing:
        spoofing_logs.append({
            "id": log.id,
            "date": log.created_at.isoformat() if log.created_at else None,
            "user": f"{user.prenom} {user.nom}" if user else "Inconnu",
            "user_id": log.user_id,
            "message": log.message,
            "face_score": log.face_score,
            "voice_score": log.voice_score,
            "exam_session_id": log.exam_session_id
        })
    
    return {
        "enabled": settings.ANTISPOOF_ENABLED,
        "total_spoofing_detected": spoofing_count,
        "recent_spoofing_attempts": spoofing_logs,
        "config": {
            "face_texture_threshold": settings.ANTISPOOF_FACE_TEXTURE_THRESHOLD,
            "face_edge_threshold": settings.ANTISPOOF_FACE_EDGE_THRESHOLD,
            "face_color_threshold": settings.ANTISPOOF_FACE_COLOR_THRESHOLD,
            "face_reflection_threshold": settings.ANTISPOOF_FACE_REFLECTION_THRESHOLD,
            "voice_quality_threshold": settings.ANTISPOOF_VOICE_QUALITY_THRESHOLD,
            "voice_replay_threshold": settings.ANTISPOOF_VOICE_REPLAY_THRESHOLD,
            "voice_spectral_threshold": settings.ANTISPOOF_VOICE_SPECTRAL_THRESHOLD,
            "voice_synthetic_threshold": settings.ANTISPOOF_VOICE_SYNTHETIC_THRESHOLD,
            "min_face_liveness_score": settings.ANTISPOOF_MIN_FACE_SCORE,
            "min_voice_liveness_score": settings.ANTISPOOF_MIN_VOICE_SCORE
        },
        "detection_methods": {
            "face": [
                "Analyse de texture LBP (Local Binary Patterns)",
                "Détection de bords (photos vs visages réels)",
                "Distribution des couleurs (photos imprimées)",
                "Détection de reflets d'écran",
                "Détection de clignement"
            ],
            "voice": [
                "Analyse de qualité audio",
                "Détection d'artefacts de replay",
                "Analyse spectrale (flatness)",
                "Détection de voix synthétique"
            ]
        }
    }
