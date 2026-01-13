"""
Script de démarrage de l'application
"""
import uvicorn
import asyncio
from app.database import init_db, async_session_maker
from app.services.auth_service import create_user, get_user_by_email
from app.models.user import UserRole

# Importer tous les modèles pour que SQLAlchemy puisse résoudre les relations
from app.models import User, BiometricData, Exam, ExamSession, SecurityLog


async def create_admin_user():
    """Créer un utilisateur admin par défaut"""
    async with async_session_maker() as db:
        # Vérifier si l'admin existe déjà
        existing = await get_user_by_email(db, "admin@example.com")
        if not existing:
            await create_user(
                db,
                email="admin@example.com",
                password="admin123",
                nom="Admin",
                prenom="Super",
                role=UserRole.ADMIN
            )
            print("✅ Utilisateur admin créé: admin@example.com / admin123")
        else:
            print("ℹ️ Utilisateur admin existe déjà")


async def main():
    """Initialisation et démarrage"""
    print("🚀 Démarrage de l'application Biométrie Examen...")
    
    # Initialiser la base de données
    await init_db()
    print("✅ Base de données initialisée")
    
    # Créer l'admin par défaut
    await create_admin_user()


if __name__ == "__main__":
    # Initialisation
    asyncio.run(main())
    
    # Démarrer le serveur
    print("\n🌐 Serveur démarré sur http://localhost:8000")
    print("📚 Documentation API: http://localhost:8000/docs")
    print("🎨 Interface: http://localhost:8000/static/login.html\n")
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
