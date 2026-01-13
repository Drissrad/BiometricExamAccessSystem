"""
Routes de gestion des examens
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
from datetime import datetime
import json

from app.database import get_db
from app.schemas.exam import (
    ExamCreate, ExamUpdate, ExamResponse, ExamWithQuestions,
    ExamSessionResponse, AnswerSubmit
)
from app.models.exam import Exam, ExamSession, ExamStatus, SessionStatus
from app.models.user import User, UserRole
from app.models.security_log import SecurityLog, LogType
from app.routers.auth import get_current_user, get_current_admin

router = APIRouter(prefix="/exams", tags=["Examens"])


# ==================== ADMIN ROUTES ====================

@router.post("/", response_model=ExamResponse)
async def create_exam(
    exam_data: ExamCreate,
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Créer un nouvel examen (Admin)"""
    exam = Exam(
        title=exam_data.title,
        description=exam_data.description,
        duration_minutes=exam_data.duration_minutes,
        questions_json=json.dumps([q.model_dump() for q in exam_data.questions]),
        face_check_enabled=exam_data.face_check_enabled,
        voice_check_enabled=exam_data.voice_check_enabled,
        face_check_interval=exam_data.face_check_interval,
        voice_check_interval=exam_data.voice_check_interval,
        start_time=exam_data.start_time,
        end_time=exam_data.end_time,
        created_by=current_user.id
    )
    
    db.add(exam)
    await db.commit()
    await db.refresh(exam)
    
    return exam


@router.get("/", response_model=List[ExamResponse])
async def list_exams(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Lister tous les examens"""
    if current_user.role == UserRole.ADMIN:
        # Admin voit tous les examens
        result = await db.execute(select(Exam))
    else:
        # Candidat voit seulement les examens publiés
        result = await db.execute(
            select(Exam).where(Exam.status.in_([ExamStatus.PUBLISHED, ExamStatus.ACTIVE]))
        )
    
    return result.scalars().all()


@router.get("/{exam_id}", response_model=ExamWithQuestions)
async def get_exam(
    exam_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Récupérer un examen avec ses questions"""
    result = await db.execute(select(Exam).where(Exam.id == exam_id))
    exam = result.scalar_one_or_none()
    
    if not exam:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Examen non trouvé"
        )
    
    # Candidat ne peut voir que les examens publiés
    if current_user.role == UserRole.CANDIDAT and exam.status not in [ExamStatus.PUBLISHED, ExamStatus.ACTIVE]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cet examen n'est pas accessible"
        )
    
    # Construire la réponse
    questions = json.loads(exam.questions_json) if exam.questions_json else []
    
    return ExamWithQuestions(
        id=exam.id,
        title=exam.title,
        description=exam.description,
        duration_minutes=exam.duration_minutes,
        status=exam.status,
        face_check_enabled=exam.face_check_enabled,
        voice_check_enabled=exam.voice_check_enabled,
        start_time=exam.start_time,
        end_time=exam.end_time,
        created_at=exam.created_at,
        questions=questions
    )


@router.put("/{exam_id}", response_model=ExamResponse)
async def update_exam(
    exam_id: int,
    exam_data: ExamUpdate,
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Mettre à jour un examen (Admin)"""
    result = await db.execute(select(Exam).where(Exam.id == exam_id))
    exam = result.scalar_one_or_none()
    
    if not exam:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Examen non trouvé"
        )
    
    # Mettre à jour les champs
    if exam_data.title is not None:
        exam.title = exam_data.title
    if exam_data.description is not None:
        exam.description = exam_data.description
    if exam_data.duration_minutes is not None:
        exam.duration_minutes = exam_data.duration_minutes
    if exam_data.status is not None:
        exam.status = exam_data.status
    if exam_data.questions is not None:
        exam.questions_json = json.dumps([q.model_dump() for q in exam_data.questions])
    
    await db.commit()
    await db.refresh(exam)
    
    return exam


@router.delete("/{exam_id}")
async def delete_exam(
    exam_id: int,
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Supprimer un examen (Admin)"""
    result = await db.execute(select(Exam).where(Exam.id == exam_id))
    exam = result.scalar_one_or_none()
    
    if not exam:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Examen non trouvé"
        )
    
    await db.delete(exam)
    await db.commit()
    
    return {"message": "Examen supprimé"}


# ==================== CANDIDAT ROUTES ====================

@router.post("/{exam_id}/start", response_model=ExamSessionResponse)
async def start_exam(
    exam_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Démarrer un examen (Candidat)"""
    # Vérifier que l'utilisateur est enrôlé
    if not current_user.is_enrolled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Veuillez d'abord effectuer l'enrôlement biométrique"
        )
    
    # Vérifier que l'examen existe et est accessible
    result = await db.execute(select(Exam).where(Exam.id == exam_id))
    exam = result.scalar_one_or_none()
    
    if not exam:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Examen non trouvé"
        )
    
    if exam.status not in [ExamStatus.PUBLISHED, ExamStatus.ACTIVE]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cet examen n'est pas disponible"
        )
    
    # Vérifier s'il y a déjà une session IN_PROGRESS
    result_in_progress = await db.execute(
        select(ExamSession).where(
            ExamSession.user_id == current_user.id,
            ExamSession.exam_id == exam_id,
            ExamSession.status == SessionStatus.IN_PROGRESS
        )
    )
    
    if result_in_progress.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Vous avez déjà une session en cours pour cet examen"
        )
    
    # Vérifier s'il y a une session PENDING (candidat assigné par admin)
    result_pending = await db.execute(
        select(ExamSession).where(
            ExamSession.user_id == current_user.id,
            ExamSession.exam_id == exam_id,
            ExamSession.status == SessionStatus.PENDING
        )
    )
    existing_pending = result_pending.scalar_one_or_none()
    
    if existing_pending:
        # Démarrer la session existante
        session = existing_pending
        session.status = SessionStatus.IN_PROGRESS
        session.started_at = datetime.utcnow()
    else:
        # Créer une nouvelle session
        session = ExamSession(
            user_id=current_user.id,
            exam_id=exam_id,
            status=SessionStatus.IN_PROGRESS,
            started_at=datetime.utcnow()
        )
        db.add(session)
    
    # Logger le démarrage
    log = SecurityLog(
        user_id=current_user.id,
        log_type=LogType.EXAM_STARTED,
        message=f"Début de l'examen: {exam.title}"
    )
    db.add(log)
    
    await db.commit()
    await db.refresh(session)
    
    return session


@router.post("/{exam_id}/submit", response_model=ExamSessionResponse)
async def submit_exam(
    exam_id: int,
    answers: List[AnswerSubmit],
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Soumettre les réponses d'un examen"""
    # Récupérer la session en cours
    result = await db.execute(
        select(ExamSession).where(
            ExamSession.user_id == current_user.id,
            ExamSession.exam_id == exam_id,
            ExamSession.status == SessionStatus.IN_PROGRESS
        )
    )
    session = result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Aucune session d'examen en cours"
        )
    
    # Récupérer l'examen pour calculer le score
    exam_result = await db.execute(select(Exam).where(Exam.id == exam_id))
    exam = exam_result.scalar_one_or_none()
    
    questions = json.loads(exam.questions_json) if exam.questions_json else []
    
    # Calculer le score
    total_points = 0
    earned_points = 0
    
    answers_dict = {a.question_id: a.answer for a in answers}
    
    for q in questions:
        total_points += q.get("points", 1.0)
        if answers_dict.get(q["id"]) == q.get("correct_answer"):
            earned_points += q.get("points", 1.0)
    
    score = (earned_points / total_points * 100) if total_points > 0 else 0
    
    # Mettre à jour la session
    session.answers_json = json.dumps([a.model_dump() for a in answers])
    session.score = score
    session.status = SessionStatus.COMPLETED
    session.completed_at = datetime.utcnow()
    
    # Logger la fin
    log = SecurityLog(
        user_id=current_user.id,
        exam_session_id=session.id,
        log_type=LogType.EXAM_COMPLETED,
        message=f"Examen terminé - Score: {score:.1f}%"
    )
    db.add(log)
    
    await db.commit()
    await db.refresh(session)
    
    return session


@router.get("/sessions/my", response_model=List[ExamSessionResponse])
async def get_my_sessions(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Récupérer mes sessions d'examen"""
    result = await db.execute(
        select(ExamSession).where(ExamSession.user_id == current_user.id)
    )
    
    return result.scalars().all()


@router.get("/sessions/{session_id}", response_model=ExamSessionResponse)
async def get_session(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Récupérer une session spécifique"""
    result = await db.execute(
        select(ExamSession).where(ExamSession.id == session_id)
    )
    session = result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session non trouvée"
        )
    
    # Vérifier les permissions
    if current_user.role != UserRole.ADMIN and session.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Accès non autorisé"
        )
    
    return session
