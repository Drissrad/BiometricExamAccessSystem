/**
 * Gestionnaire de la caméra et de la reconnaissance faciale
 */

class CameraManager {
    constructor(videoElement, canvasElement = null) {
        this.video = videoElement;
        this.canvas = canvasElement || document.createElement('canvas');
        this.stream = null;
        this.isActive = false;
    }

    /**
     * Démarrer la caméra
     */
    async start() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                },
                audio: false
            });

            this.video.srcObject = this.stream;
            await this.video.play();
            this.isActive = true;

            // Configurer le canvas
            this.canvas.width = this.video.videoWidth;
            this.canvas.height = this.video.videoHeight;

            return true;
        } catch (error) {
            console.error('Erreur d\'accès à la caméra:', error);
            throw new Error('Impossible d\'accéder à la caméra. Veuillez vérifier les permissions.');
        }
    }

    /**
     * Arrêter la caméra
     */
    stop() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        this.video.srcObject = null;
        this.isActive = false;
    }

    /**
     * Capturer une image
     */
    captureImage() {
        if (!this.isActive) {
            throw new Error('La caméra n\'est pas active');
        }

        const ctx = this.canvas.getContext('2d');
        ctx.drawImage(this.video, 0, 0);
        
        // Retourner en base64
        return this.canvas.toDataURL('image/jpeg', 0.8);
    }

    /**
     * Vérifier si un visage est visible (approximation)
     */
    checkFacePresence() {
        // Cette méthode pourrait être améliorée avec une détection côté client
        // Pour l'instant, on vérifie juste si la caméra fonctionne
        return this.isActive;
    }
}

/**
 * Gestionnaire du microphone et de l'enregistrement vocal
 * Enregistre directement en WAV pour éviter les problèmes de conversion
 */

class AudioRecorder {
    constructor() {
        this.audioContext = null;
        this.mediaStreamSource = null;
        this.processor = null;
        this.stream = null;
        this.audioChunks = [];
        this.isRecording = false;
        this.sampleRate = 16000;
    }

    /**
     * Démarrer l'enregistrement
     */
    async start() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    sampleRate: this.sampleRate
                }
            });

            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: this.sampleRate
            });
            
            this.mediaStreamSource = this.audioContext.createMediaStreamSource(this.stream);
            this.processor = this.audioContext.createScriptProcessor(4096, 1, 1);
            
            this.audioChunks = [];
            
            this.processor.onaudioprocess = (e) => {
                if (this.isRecording) {
                    const inputData = e.inputBuffer.getChannelData(0);
                    // Copier les données car le buffer est réutilisé
                    this.audioChunks.push(new Float32Array(inputData));
                }
            };
            
            this.mediaStreamSource.connect(this.processor);
            this.processor.connect(this.audioContext.destination);
            
            this.isRecording = true;
            return true;
        } catch (error) {
            console.error('Erreur d\'accès au microphone:', error);
            throw new Error('Impossible d\'accéder au microphone. Veuillez vérifier les permissions.');
        }
    }

    /**
     * Arrêter l'enregistrement et obtenir l'audio en base64 WAV
     */
    async stop() {
        return new Promise((resolve, reject) => {
            if (!this.isRecording) {
                reject(new Error('Aucun enregistrement en cours'));
                return;
            }

            this.isRecording = false;
            
            // Déconnecter les noeuds audio
            if (this.processor) {
                this.processor.disconnect();
            }
            if (this.mediaStreamSource) {
                this.mediaStreamSource.disconnect();
            }
            
            // Arrêter le stream
            if (this.stream) {
                this.stream.getTracks().forEach(track => track.stop());
            }
            
            try {
                // Combiner tous les chunks audio
                const totalLength = this.audioChunks.reduce((acc, chunk) => acc + chunk.length, 0);
                const audioData = new Float32Array(totalLength);
                let offset = 0;
                for (const chunk of this.audioChunks) {
                    audioData.set(chunk, offset);
                    offset += chunk.length;
                }
                
                // Convertir en WAV
                const wavBlob = this.createWavBlob(audioData, this.audioContext.sampleRate);
                
                // Convertir en base64
                const reader = new FileReader();
                reader.onloadend = () => resolve(reader.result);
                reader.onerror = () => reject(new Error('Erreur de conversion audio'));
                reader.readAsDataURL(wavBlob);
                
            } catch (error) {
                reject(error);
            }
        });
    }
    
    /**
     * Créer un Blob WAV à partir des données audio Float32
     */
    createWavBlob(audioData, sampleRate) {
        const numChannels = 1;
        const bitsPerSample = 16;
        const bytesPerSample = bitsPerSample / 8;
        const blockAlign = numChannels * bytesPerSample;
        
        // Convertir Float32 en Int16
        const samples = new Int16Array(audioData.length);
        for (let i = 0; i < audioData.length; i++) {
            const s = Math.max(-1, Math.min(1, audioData[i]));
            samples[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        
        const dataSize = samples.length * bytesPerSample;
        const buffer = new ArrayBuffer(44 + dataSize);
        const view = new DataView(buffer);
        
        // WAV Header
        this.writeString(view, 0, 'RIFF');
        view.setUint32(4, 36 + dataSize, true);
        this.writeString(view, 8, 'WAVE');
        
        // Format chunk
        this.writeString(view, 12, 'fmt ');
        view.setUint32(16, 16, true); // Chunk size
        view.setUint16(20, 1, true); // Audio format (PCM)
        view.setUint16(22, numChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * blockAlign, true); // Byte rate
        view.setUint16(32, blockAlign, true);
        view.setUint16(34, bitsPerSample, true);
        
        // Data chunk
        this.writeString(view, 36, 'data');
        view.setUint32(40, dataSize, true);
        
        // Write audio data
        const dataView = new Int16Array(buffer, 44);
        dataView.set(samples);
        
        return new Blob([buffer], { type: 'audio/wav' });
    }
    
    writeString(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }

    /**
     * Convertir un Blob en base64
     */
    blobToBase64(blob) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onloadend = () => resolve(reader.result);
            reader.onerror = reject;
            reader.readAsDataURL(blob);
        });
    }

    /**
     * Annuler l'enregistrement
     */
    cancel() {
        this.isRecording = false;
        if (this.processor) {
            this.processor.disconnect();
        }
        if (this.mediaStreamSource) {
            this.mediaStreamSource.disconnect();
        }
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
        }
        this.audioChunks = [];
    }
}

/**
 * Gestionnaire de surveillance pendant l'examen
 */

class SurveillanceManager {
    constructor(sessionId, videoElement) {
        this.sessionId = sessionId;
        this.camera = new CameraManager(videoElement);
        this.audioRecorder = new AudioRecorder();
        
        this.faceCheckInterval = null;
        this.voiceChallengeInterval = null;
        this.presenceCheckInterval = null;  // Vérification de présence locale
        this.isActive = false;
        this.isCheckingFace = false;  // Pour éviter les requêtes simultanées
        
        this.onFaceCheckResult = null;
        this.onVoiceChallengeRequired = null;
        this.onAbsenceDetected = null;
        this.onPresenceStatus = null;  // Nouveau callback pour le statut de présence
        this.onError = null;
    }

    /**
     * Démarrer la surveillance
     */
    async start(faceCheckIntervalMs = 5000, voiceChallengeIntervalMs = 120000) {
        try {
            await this.camera.start();
            this.isActive = true;

            // Vérification d'identité périodique (envoi au serveur)
            this.faceCheckInterval = setInterval(async () => {
                await this.performFaceCheck();
            }, faceCheckIntervalMs);

            // Défi vocal périodique
            this.voiceChallengeInterval = setInterval(async () => {
                await this.requestVoiceChallenge();
            }, voiceChallengeIntervalMs);

            // Première vérification immédiate
            await this.performFaceCheck();

            return true;
        } catch (error) {
            this.handleError(error);
            return false;
        }
    }

    /**
     * Arrêter la surveillance
     */
    stop() {
        this.isActive = false;
        if (this.faceCheckInterval) {
            clearInterval(this.faceCheckInterval);
            this.faceCheckInterval = null;
        }
        if (this.voiceChallengeInterval) {
            clearInterval(this.voiceChallengeInterval);
            this.voiceChallengeInterval = null;
        }
        if (this.presenceCheckInterval) {
            clearInterval(this.presenceCheckInterval);
            this.presenceCheckInterval = null;
        }
        this.camera.stop();
    }

    /**
     * Effectuer une vérification faciale
     */
    async performFaceCheck() {
        if (!this.isActive || this.isCheckingFace) return;
        
        this.isCheckingFace = true;
        
        try {
            const imageBase64 = this.camera.captureImage();
            const result = await api.checkFace(this.sessionId, imageBase64);
            
            if (this.onFaceCheckResult) {
                this.onFaceCheckResult(result);
            }
            
            // Si disqualifié, arrêter la surveillance
            if (result.disqualified) {
                this.stop();
            }

            return result;
        } catch (error) {
            console.error('Erreur vérification faciale:', error);
            // Signaler l'erreur mais ne pas arrêter l'examen
            if (this.onError) {
                this.onError(error);
            }
        } finally {
            this.isCheckingFace = false;
        }
    }

    /**
     * Demander un défi vocal
     */
    async requestVoiceChallenge() {
        try {
            const challenge = await api.getVoiceChallenge(this.sessionId);
            
            if (this.onVoiceChallengeRequired) {
                this.onVoiceChallengeRequired(challenge);
            }

            return challenge;
        } catch (error) {
            this.handleError(error);
            throw error;
        }
    }

    /**
     * Soumettre la réponse au défi vocal
     */
    async submitVoiceResponse(challengeId) {
        try {
            // Enregistrer 5 secondes d'audio
            await this.audioRecorder.start();
            
            await new Promise(resolve => setTimeout(resolve, 5000));
            
            const audioBase64 = await this.audioRecorder.stop();
            const result = await api.submitVoiceChallenge(this.sessionId, challengeId, audioBase64);
            
            return result;
        } catch (error) {
            this.audioRecorder.cancel();
            throw error;
        }
    }

    /**
     * Gérer les erreurs
     */
    handleError(error) {
        console.error('Surveillance Error:', error);
        if (this.onError) {
            this.onError(error);
        }
    }
}

// Export pour utilisation globale
window.CameraManager = CameraManager;
window.AudioRecorder = AudioRecorder;
window.SurveillanceManager = SurveillanceManager;
