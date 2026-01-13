/**
 * API Client pour le système biométrique
 */

const API_BASE_URL = '/api';

class ApiClient {
    constructor() {
        this.token = localStorage.getItem('access_token');
    }

    /**
     * Définir le token d'authentification
     */
    setToken(token) {
        this.token = token;
        localStorage.setItem('access_token', token);
    }

    /**
     * Supprimer le token
     */
    clearToken() {
        this.token = null;
        localStorage.removeItem('access_token');
    }

    /**
     * Headers par défaut
     */
    getHeaders() {
        const headers = {
            'Content-Type': 'application/json',
        };
        if (this.token) {
            headers['Authorization'] = `Bearer ${this.token}`;
        }
        return headers;
    }

    /**
     * Requête générique
     */
    async request(endpoint, options = {}) {
        const url = `${API_BASE_URL}${endpoint}`;
        const config = {
            headers: this.getHeaders(),
            ...options,
        };

        try {
            const response = await fetch(url, config);
            
            if (response.status === 401) {
                this.clearToken();
                window.location.href = '/static/login.html';
                throw new Error('Non authentifié');
            }

            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.detail || 'Une erreur est survenue');
            }

            return data;
        } catch (error) {
            console.error('API Error:', error);
            throw error;
        }
    }

    // ==================== AUTH ====================

    /**
     * Inscription
     */
    async register(userData) {
        return this.request('/auth/register', {
            method: 'POST',
            body: JSON.stringify(userData),
        });
    }

    /**
     * Connexion simple
     */
    async login(email, password) {
        const formData = new URLSearchParams();
        formData.append('username', email);
        formData.append('password', password);

        const response = await fetch(`${API_BASE_URL}/auth/token`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: formData,
        });

        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Échec de la connexion');
        }

        this.setToken(data.access_token);
        return data;
    }

    /**
     * Connexion avec biométrie
     */
    async loginBiometric(email, password, faceImage, voiceAudio) {
        return this.request('/auth/login-biometric', {
            method: 'POST',
            body: JSON.stringify({
                email,
                password,
                face_image_base64: faceImage,
                voice_audio_base64: voiceAudio,
            }),
        });
    }

    /**
     * Enrôlement biométrique
     */
    async enroll(faceImage, voiceAudio) {
        return this.request('/auth/enroll', {
            method: 'POST',
            body: JSON.stringify({
                face_image_base64: faceImage,
                voice_audio_base64: voiceAudio,
            }),
        });
    }

    /**
     * Vérification biométrique
     */
    async verifyBiometric(faceImage, voiceAudio) {
        return this.request('/auth/verify-biometric', {
            method: 'POST',
            body: JSON.stringify({
                face_image_base64: faceImage,
                voice_audio_base64: voiceAudio,
            }),
        });
    }

    /**
     * Obtenir l'utilisateur courant
     */
    async getCurrentUser() {
        return this.request('/auth/me');
    }

    // ==================== EXAMS ====================

    /**
     * Lister les examens
     */
    async getExams() {
        return this.request('/exams/');
    }

    /**
     * Obtenir un examen
     */
    async getExam(examId) {
        return this.request(`/exams/${examId}`);
    }

    /**
     * Créer un examen (admin)
     */
    async createExam(examData) {
        return this.request('/exams/', {
            method: 'POST',
            body: JSON.stringify(examData),
        });
    }

    /**
     * Mettre à jour un examen
     */
    async updateExam(examId, examData) {
        return this.request(`/exams/${examId}`, {
            method: 'PUT',
            body: JSON.stringify(examData),
        });
    }

    /**
     * Publier un examen
     */
    async publishExam(examId) {
        return this.updateExam(examId, { status: 'published' });
    }

    /**
     * Démarrer un examen
     */
    async startExam(examId) {
        return this.request(`/exams/${examId}/start`, {
            method: 'POST',
        });
    }

    /**
     * Soumettre les réponses
     */
    async submitExam(examId, answers) {
        return this.request(`/exams/${examId}/submit`, {
            method: 'POST',
            body: JSON.stringify(answers),
        });
    }

    /**
     * Obtenir mes sessions
     */
    async getMySessions() {
        return this.request('/exams/sessions/my');
    }

    // ==================== SURVEILLANCE ====================

    /**
     * Vérification faciale pendant l'examen
     */
    async checkFace(sessionId, imageBase64) {
        return this.request(`/surveillance/face-check/${sessionId}`, {
            method: 'POST',
            body: JSON.stringify({
                image_base64: imageBase64,
            }),
        });
    }

    /**
     * Obtenir un défi vocal
     */
    async getVoiceChallenge(sessionId) {
        return this.request(`/surveillance/voice-challenge/${sessionId}`);
    }

    /**
     * Soumettre le défi vocal
     */
    async submitVoiceChallenge(sessionId, challengeId, audioBase64) {
        return this.request(`/surveillance/voice-check/${sessionId}`, {
            method: 'POST',
            body: JSON.stringify({
                challenge_id: challengeId,
                audio_base64: audioBase64,
            }),
        });
    }

    /**
     * Signaler une absence
     */
    async reportAbsence(sessionId) {
        return this.request(`/surveillance/absence/${sessionId}`, {
            method: 'POST',
        });
    }

    /**
     * Obtenir le statut de surveillance
     */
    async getSurveillanceStatus(sessionId) {
        return this.request(`/surveillance/status/${sessionId}`);
    }

    // ==================== ADMIN ====================

    /**
     * Lister les utilisateurs
     */
    async getUsers(role = null) {
        const query = role ? `?role=${role}` : '';
        return this.request(`/admin/users${query}`);
    }

    /**
     * Assigner un candidat à un examen
     */
    async assignCandidateToExam(userId, examId) {
        return this.request('/admin/assign-candidate', {
            method: 'POST',
            body: JSON.stringify({ user_id: userId, exam_id: examId }),
        });
    }

    /**
     * Obtenir les sessions d'un examen
     */
    async getExamSessions(examId) {
        return this.request(`/admin/exam-sessions/${examId}`);
    }

    /**
     * Obtenir les logs de sécurité
     */
    async getSecurityLogs(userId = null, logType = null, limit = 100) {
        const params = new URLSearchParams();
        if (userId) params.append('user_id', userId);
        if (logType) params.append('log_type', logType);
        params.append('limit', limit);
        
        return this.request(`/admin/security-logs?${params}`);
    }

    /**
     * Obtenir les statistiques
     */
    async getStatistics() {
        return this.request('/admin/statistics');
    }

    /**
     * Créer un candidat avec données biométriques (admin)
     */
    async createCandidateWithBiometric(candidateData) {
        return this.request('/admin/create-candidate', {
            method: 'POST',
            body: JSON.stringify(candidateData),
        });
    }
}

// Instance globale
const api = new ApiClient();
