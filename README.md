# 🔐 Système d'Accès Biométrique aux Examens en Ligne

## 📋 Description

Ce projet est un système de sécurisation des examens en ligne utilisant la biométrie multimodale (visage + voix). Il permet de :

- ✅ Authentifier les candidats par reconnaissance faciale et vocale
- ✅ Surveiller en continu pendant les examens
- ✅ Détecter les anomalies et tentatives de fraude
- ✅ Gérer les examens et les utilisateurs

## 🏗️ Architecture

```
biometrie-exam/
├── app/
│   ├── __init__.py
│   ├── main.py              # Application FastAPI
│   ├── config.py            # Configuration
│   ├── database.py          # Base de données SQLite
│   ├── models/              # Modèles SQLAlchemy
│   │   ├── user.py
│   │   ├── biometric.py
│   │   ├── exam.py
│   │   └── security_log.py
│   ├── schemas/             # Schémas Pydantic
│   │   ├── user.py
│   │   ├── exam.py
│   │   └── biometric.py
│   ├── services/            # Services métier
│   │   ├── auth_service.py
│   │   ├── face_service.py
│   │   ├── voice_service.py
│   │   └── biometric_service.py
│   └── routers/             # Routes API
│       ├── auth.py
│       ├── exams.py
│       ├── surveillance.py
│       └── admin.py
├── static/                  # Frontend
│   ├── css/style.css
│   ├── js/
│   │   ├── api.js
│   │   └── biometric.js
│   ├── login.html
│   ├── dashboard.html
│   ├── enroll.html
│   ├── exams.html
│   ├── exam.html
│   └── admin.html
├── requirements.txt
├── run.py
└── README.md
```

## 🚀 Installation

### Prérequis

- Python 3.9+
- pip
- Webcam et microphone

### Étapes d'installation

1. **Cloner ou créer le projet**

2. **Créer un environnement virtuel**
```bash
python -m venv env
```

3. **Activer l'environnement**
```bash
# Windows
.\env\Scripts\activate

# Linux/Mac
source env/bin/activate
```

4. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

> ⚠️ **Note**: L'installation de `dlib` et `face-recognition` peut nécessiter CMake et un compilateur C++.

5. **Configurer l'environnement**
```bash
cp .env.example .env
# Modifiez le fichier .env selon vos besoins
```

6. **Lancer l'application**
```bash
python run.py
```

## 🌐 Accès

- **Application**: http://localhost:8000/static/login.html
- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Compte Admin par défaut
- Email: `admin@example.com`
- Mot de passe: `admin123`

## 📖 Fonctionnalités

### 👨‍💼 Administrateur

- Créer et gérer les comptes étudiants
- Créer et configurer les examens
- Consulter les journaux de sécurité
- Voir les statistiques et résultats

### 👩‍🎓 Candidat

- S'inscrire et se connecter
- Effectuer l'enrôlement biométrique (visage + voix)
- Passer les examens avec surveillance continue
- Consulter ses résultats

## 🔒 Sécurité Biométrique

### Enrôlement
1. Capture du visage via webcam
2. Enregistrement vocal (lecture d'une phrase)
3. Extraction et stockage des descripteurs biométriques

### Authentification
1. Vérification du visage
2. Vérification de la voix
3. Fusion multimodale: `Score = 0.6 × Visage + 0.4 × Voix`

### Surveillance Continue
- Vérifications faciales périodiques (toutes les 30s)
- Défis vocaux aléatoires (toutes les 2min)
- Détection d'absence
- Détection de changement de personne

## 🛠️ Technologies

- **Backend**: Python, FastAPI, SQLAlchemy
- **Base de données**: SQLite (async)
- **Reconnaissance faciale**: face_recognition, dlib, OpenCV
- **Reconnaissance vocale**: librosa, scipy
- **Frontend**: HTML5, CSS3, JavaScript vanilla
- **Authentification**: JWT

## ⚙️ Configuration

Modifiez le fichier `.env` pour personnaliser:

```env
# Seuils de reconnaissance (0-1)
FACE_RECOGNITION_THRESHOLD=0.6
VOICE_RECOGNITION_THRESHOLD=0.7
MULTIMODAL_THRESHOLD=0.65

# Poids de fusion
FACE_WEIGHT=0.6
VOICE_WEIGHT=0.4

# Intervalles de surveillance (secondes)
FACE_CHECK_INTERVAL_SECONDS=30
VOICE_CHALLENGE_INTERVAL_SECONDS=120
```

## 📝 API Endpoints

### Authentification
- `POST /api/auth/register` - Inscription
- `POST /api/auth/token` - Connexion
- `POST /api/auth/enroll` - Enrôlement biométrique
- `GET /api/auth/me` - Profil utilisateur

### Examens
- `GET /api/exams/` - Liste des examens
- `POST /api/exams/{id}/start` - Démarrer un examen
- `POST /api/exams/{id}/submit` - Soumettre les réponses

### Surveillance
- `POST /api/surveillance/face-check/{session_id}` - Vérification faciale
- `GET /api/surveillance/voice-challenge/{session_id}` - Obtenir un défi vocal
- `POST /api/surveillance/voice-check/{session_id}` - Soumettre le défi vocal

### Administration
- `GET /api/admin/users` - Liste des utilisateurs
- `GET /api/admin/security-logs` - Journaux de sécurité
- `GET /api/admin/statistics` - Statistiques

## 📄 Licence

Ce projet est développé à des fins éducatives.

## 👥 Auteur

Projet de fin d'études - Système d'accès biométrique aux examens en ligne
