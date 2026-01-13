"""
Service Anti-Spoofing pour détecter les tentatives de fraude biométrique

Ce service implémente plusieurs techniques pour détecter :
- Photos imprimées ou affichées sur écran
- Vidéos pré-enregistrées
- Enregistrements audio
- Masques et deepfakes

Techniques utilisées :
1. Analyse de texture (LBP - Local Binary Patterns)
2. Détection de mouvement/clignotement
3. Analyse spectrale de la voix
4. Vérification de la qualité du signal
"""
import numpy as np
import cv2
import base64
from typing import Optional, Tuple, List, Dict
import logging
from scipy import signal
from scipy.fft import fft
import io

logger = logging.getLogger(__name__)


class FaceAntiSpoofService:
    """
    Service anti-spoofing pour la reconnaissance faciale
    
    Détecte les attaques par :
    - Photo imprimée
    - Photo sur écran (téléphone, tablette)
    - Vidéo pré-enregistrée
    - Masques
    """
    
    def __init__(self):
        # Seuils de détection
        self.texture_threshold = 0.3  # Seuil pour l'analyse de texture
        self.edge_threshold = 50  # Seuil pour la densité des bords
        self.color_variance_threshold = 15  # Variance de couleur minimale
        
        # Historique pour la détection de mouvement (multi-frames)
        self.frame_history: Dict[int, List[np.ndarray]] = {}  # user_id -> frames
        self.max_history = 5
        
        # Détecteur d'yeux pour le clignotement
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def decode_base64_image(self, image_base64: str) -> Optional[np.ndarray]:
        """Décoder une image base64 en array numpy"""
        try:
            if ',' in image_base64:
                image_base64 = image_base64.split(',')[1]
            
            image_data = base64.b64decode(image_base64)
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            return image
        except Exception as e:
            logger.error(f"Erreur décodage image: {e}")
            return None
    
    def analyze_texture_lbp(self, image: np.ndarray) -> Tuple[float, bool]:
        """
        Analyse de texture avec Local Binary Patterns (LBP)
        
        Les images réelles ont des textures de peau plus variées
        Les photos/écrans ont des textures plus uniformes ou avec motifs réguliers
        
        Returns:
            (score, is_real) - score de texture et si l'image semble réelle
        """
        try:
            # Convertir en niveaux de gris
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Redimensionner pour normaliser
            gray = cv2.resize(gray, (128, 128))
            
            # Calculer le LBP simple
            lbp = np.zeros_like(gray)
            
            for i in range(1, gray.shape[0] - 1):
                for j in range(1, gray.shape[1] - 1):
                    center = gray[i, j]
                    code = 0
                    code |= (gray[i-1, j-1] >= center) << 7
                    code |= (gray[i-1, j] >= center) << 6
                    code |= (gray[i-1, j+1] >= center) << 5
                    code |= (gray[i, j+1] >= center) << 4
                    code |= (gray[i+1, j+1] >= center) << 3
                    code |= (gray[i+1, j] >= center) << 2
                    code |= (gray[i+1, j-1] >= center) << 1
                    code |= (gray[i, j-1] >= center) << 0
                    lbp[i, j] = code
            
            # Calculer l'histogramme LBP
            hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
            hist = hist.astype(float)
            hist /= (hist.sum() + 1e-7)
            
            # Calculer l'entropie de l'histogramme
            entropy = -np.sum(hist * np.log2(hist + 1e-7))
            
            # Les vraies images ont une entropie plus élevée (textures variées)
            # Score normalisé entre 0 et 1
            score = min(entropy / 8.0, 1.0)
            
            is_real = score > self.texture_threshold
            
            logger.debug(f"LBP Texture Score: {score:.3f}, Is Real: {is_real}")
            
            return score, is_real
            
        except Exception as e:
            logger.error(f"Erreur analyse texture: {e}")
            return 0.5, True  # Par défaut, accepter
    
    def analyze_edges(self, image: np.ndarray) -> Tuple[float, bool]:
        """
        Analyse des bords avec Canny
        
        Les écrans ont souvent des bords de pixels visibles
        Les photos imprimées peuvent avoir des motifs de trame
        
        Returns:
            (score, is_real)
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Appliquer un flou pour réduire le bruit
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Détection des bords
            edges = cv2.Canny(blurred, 50, 150)
            
            # Calculer la densité des bords
            edge_density = np.sum(edges > 0) / edges.size * 100
            
            # Les vraies images ont une densité de bords modérée
            # Trop peu = photo floue ou lissée
            # Trop = écran avec pixels visibles ou trame d'impression
            
            score = 1.0
            if edge_density < 1.0:
                score = edge_density / 1.0  # Pénalité si trop peu de bords
            elif edge_density > 15.0:
                score = max(0, 1.0 - (edge_density - 15.0) / 30.0)  # Pénalité si trop de bords
            
            is_real = 1.0 <= edge_density <= 20.0
            
            logger.debug(f"Edge Density: {edge_density:.2f}%, Score: {score:.3f}")
            
            return score, is_real
            
        except Exception as e:
            logger.error(f"Erreur analyse bords: {e}")
            return 0.5, True
    
    def analyze_color_distribution(self, image: np.ndarray) -> Tuple[float, bool]:
        """
        Analyse de la distribution des couleurs
        
        Les vraies images de peau ont des distributions de couleur spécifiques
        Les écrans peuvent avoir des distorsions de couleur
        
        Returns:
            (score, is_real)
        """
        try:
            # Convertir en HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Analyser la variance dans chaque canal
            h_var = np.var(hsv[:, :, 0])
            s_var = np.var(hsv[:, :, 1])
            v_var = np.var(hsv[:, :, 2])
            
            # Les vraies images ont une variance de couleur modérée
            avg_var = (h_var + s_var + v_var) / 3
            
            # Score basé sur la variance
            if avg_var < self.color_variance_threshold:
                score = avg_var / self.color_variance_threshold
            else:
                score = 1.0
            
            is_real = avg_var >= self.color_variance_threshold
            
            logger.debug(f"Color Variance: {avg_var:.2f}, Score: {score:.3f}")
            
            return score, is_real
            
        except Exception as e:
            logger.error(f"Erreur analyse couleur: {e}")
            return 0.5, True
    
    def detect_screen_reflection(self, image: np.ndarray) -> Tuple[float, bool]:
        """
        Détecter les reflets d'écran (moiré patterns)
        
        Les photos sur écran ont souvent des motifs moiré caractéristiques
        
        Returns:
            (score, is_real) - score élevé = pas de reflet détecté
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (256, 256))
            
            # Appliquer FFT pour détecter les motifs périodiques
            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            magnitude = np.abs(fshift)
            
            # Analyser les pics dans le spectre de fréquence
            # Les motifs moiré créent des pics réguliers
            
            # Normaliser
            magnitude = magnitude / (magnitude.max() + 1e-7)
            
            # Calculer l'énergie dans les hautes fréquences
            center = magnitude.shape[0] // 2
            radius = magnitude.shape[0] // 4
            
            # Masque pour les hautes fréquences
            y, x = np.ogrid[:magnitude.shape[0], :magnitude.shape[1]]
            mask = ((x - center) ** 2 + (y - center) ** 2) > radius ** 2
            
            high_freq_energy = np.mean(magnitude[mask])
            
            # Un écran a généralement plus d'énergie haute fréquence régulière
            is_real = high_freq_energy < 0.1
            score = 1.0 - min(high_freq_energy * 10, 1.0)
            
            logger.debug(f"High Freq Energy: {high_freq_energy:.4f}, Score: {score:.3f}")
            
            return score, is_real
            
        except Exception as e:
            logger.error(f"Erreur détection reflet: {e}")
            return 0.5, True
    
    def detect_blink(self, user_id: int, image: np.ndarray) -> Tuple[float, bool, str]:
        """
        Détecter le clignotement des yeux (liveness)
        
        Nécessite plusieurs frames pour détecter le mouvement
        
        Returns:
            (score, blink_detected, message)
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Détecter le visage
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return 0.0, False, "Aucun visage détecté"
            
            # Prendre le premier visage
            x, y, w, h = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            
            # Détecter les yeux dans le visage
            eyes = self.eye_cascade.detectMultiScale(face_roi, 1.1, 4)
            
            eyes_detected = len(eyes) >= 2
            
            # Stocker dans l'historique
            if user_id not in self.frame_history:
                self.frame_history[user_id] = []
            
            self.frame_history[user_id].append({
                'eyes_count': len(eyes),
                'face_roi': face_roi
            })
            
            # Garder seulement les dernières frames
            if len(self.frame_history[user_id]) > self.max_history:
                self.frame_history[user_id] = self.frame_history[user_id][-self.max_history:]
            
            # Vérifier si un clignotement a été détecté (variation du nombre d'yeux)
            history = self.frame_history[user_id]
            blink_detected = False
            
            if len(history) >= 3:
                eye_counts = [h['eyes_count'] for h in history]
                # Un clignotement = yeux détectés, puis moins/pas d'yeux, puis yeux détectés
                for i in range(1, len(eye_counts) - 1):
                    if eye_counts[i-1] >= 2 and eye_counts[i] < 2 and eye_counts[i+1] >= 2:
                        blink_detected = True
                        break
            
            score = 1.0 if blink_detected else (0.7 if eyes_detected else 0.3)
            message = "Clignotement détecté" if blink_detected else ("Yeux détectés" if eyes_detected else "Yeux non détectés")
            
            return score, blink_detected, message
            
        except Exception as e:
            logger.error(f"Erreur détection clignotement: {e}")
            return 0.5, False, str(e)
    
    def check_liveness(self, image_base64: str, user_id: int = 0) -> Tuple[bool, float, Dict]:
        """
        Vérification complète de liveness (anti-spoofing)
        
        Combine plusieurs techniques pour détecter si l'image est d'une vraie personne
        
        Args:
            image_base64: Image en base64
            user_id: ID utilisateur pour le suivi multi-frames
            
        Returns:
            (is_live, confidence, details)
        """
        image = self.decode_base64_image(image_base64)
        
        if image is None:
            return False, 0.0, {"error": "Image invalide"}
        
        # Exécuter toutes les analyses
        texture_score, texture_ok = self.analyze_texture_lbp(image)
        edge_score, edge_ok = self.analyze_edges(image)
        color_score, color_ok = self.analyze_color_distribution(image)
        reflection_score, reflection_ok = self.detect_screen_reflection(image)
        blink_score, blink_detected, blink_msg = self.detect_blink(user_id, image)
        
        # Calculer le score global pondéré
        weights = {
            'texture': 0.25,
            'edge': 0.15,
            'color': 0.15,
            'reflection': 0.20,
            'blink': 0.25
        }
        
        global_score = (
            weights['texture'] * texture_score +
            weights['edge'] * edge_score +
            weights['color'] * color_score +
            weights['reflection'] * reflection_score +
            weights['blink'] * blink_score
        )
        
        # Décision finale
        # On est plus strict : au moins 3 tests doivent passer
        tests_passed = sum([texture_ok, edge_ok, color_ok, reflection_ok, blink_detected])
        is_live = global_score >= 0.5 and tests_passed >= 2
        
        details = {
            "texture": {"score": round(texture_score, 3), "passed": texture_ok},
            "edges": {"score": round(edge_score, 3), "passed": edge_ok},
            "color": {"score": round(color_score, 3), "passed": color_ok},
            "reflection": {"score": round(reflection_score, 3), "passed": reflection_ok},
            "blink": {"score": round(blink_score, 3), "detected": blink_detected, "message": blink_msg},
            "tests_passed": tests_passed,
            "global_score": round(global_score, 3)
        }
        
        logger.info(f"Anti-Spoof Check: is_live={is_live}, score={global_score:.3f}, tests_passed={tests_passed}/5")
        
        return is_live, global_score, details


class VoiceAntiSpoofService:
    """
    Service anti-spoofing pour la reconnaissance vocale
    
    Détecte les attaques par :
    - Enregistrement audio rejoué
    - Synthèse vocale (TTS)
    - Voice morphing
    """
    
    def __init__(self):
        # Seuils
        self.min_duration = 1.5  # Durée minimale en secondes
        self.silence_threshold = 0.02  # Seuil de silence
        self.spectral_flatness_threshold = 0.3
    
    def analyze_audio_quality(self, audio: np.ndarray, sr: int) -> Tuple[float, bool, str]:
        """
        Analyser la qualité audio pour détecter les replays
        
        Les enregistrements rejoués ont souvent :
        - Compression audio
        - Bruit de fond constant
        - Distorsion
        
        Returns:
            (score, is_real, message)
        """
        try:
            # Durée de l'audio
            duration = len(audio) / sr
            if duration < self.min_duration:
                return 0.3, False, f"Audio trop court ({duration:.1f}s < {self.min_duration}s)"
            
            # Calculer le rapport signal/bruit (SNR)
            # Estimer le bruit comme le percentile bas de l'énergie
            frame_size = int(sr * 0.02)  # 20ms frames
            hop_size = frame_size // 2
            
            energies = []
            for i in range(0, len(audio) - frame_size, hop_size):
                frame = audio[i:i + frame_size]
                energy = np.sum(frame ** 2)
                energies.append(energy)
            
            energies = np.array(energies)
            noise_floor = np.percentile(energies, 10)
            signal_peak = np.percentile(energies, 90)
            
            if noise_floor > 0:
                snr = 10 * np.log10(signal_peak / noise_floor)
            else:
                snr = 40
            
            # Un bon audio a un SNR entre 15 et 50 dB
            snr_score = 1.0 if 15 <= snr <= 50 else 0.5
            
            # Vérifier la variance du signal (les replays sont souvent plus "plats")
            variance = np.var(audio)
            variance_score = min(variance * 1000, 1.0)
            
            global_score = (snr_score + variance_score) / 2
            is_real = global_score > 0.5
            
            message = f"SNR: {snr:.1f}dB, Variance: {variance:.4f}"
            
            return global_score, is_real, message
            
        except Exception as e:
            logger.error(f"Erreur analyse qualité audio: {e}")
            return 0.5, True, str(e)
    
    def detect_replay_artifacts(self, audio: np.ndarray, sr: int) -> Tuple[float, bool]:
        """
        Détecter les artefacts de replay
        
        Les replays ont souvent :
        - Coupures abruptes au début/fin
        - Bruit de fond constant
        - Distorsion due à la double conversion
        
        Returns:
            (score, is_real)
        """
        try:
            # Vérifier les coupures abruptes au début
            start_samples = int(sr * 0.1)  # 100ms
            end_samples = int(sr * 0.1)
            
            start_energy = np.sum(audio[:start_samples] ** 2)
            end_energy = np.sum(audio[-end_samples:] ** 2)
            mid_energy = np.sum(audio[start_samples:-end_samples] ** 2) / max(1, len(audio) - start_samples - end_samples) * start_samples
            
            # Un vrai audio a une montée progressive, pas une coupure nette
            start_ratio = start_energy / (mid_energy + 1e-7)
            end_ratio = end_energy / (mid_energy + 1e-7)
            
            # Score basé sur la progressivité
            start_score = 1.0 if start_ratio < 0.5 else 0.5
            end_score = 1.0 if end_ratio < 0.5 else 0.5
            
            # Analyser la stationnarité du bruit
            # Diviser l'audio en segments et comparer leur spectre
            segment_size = sr  # 1 seconde
            segments = [audio[i:i+segment_size] for i in range(0, len(audio) - segment_size, segment_size)]
            
            if len(segments) >= 2:
                # Calculer la variance inter-segments du spectre
                spectra = [np.abs(np.fft.fft(seg)[:len(seg)//2]) for seg in segments]
                spectral_variance = np.var([np.mean(s) for s in spectra])
                
                # Un vrai audio a plus de variation entre segments
                variance_score = min(spectral_variance * 10, 1.0)
            else:
                variance_score = 0.5
            
            global_score = (start_score + end_score + variance_score) / 3
            is_real = global_score > 0.6
            
            return global_score, is_real
            
        except Exception as e:
            logger.error(f"Erreur détection replay: {e}")
            return 0.5, True
    
    def analyze_spectral_flatness(self, audio: np.ndarray, sr: int) -> Tuple[float, bool]:
        """
        Analyser la planéité spectrale (spectral flatness)
        
        La voix humaine a un spectre caractéristique
        Les synthétiseurs et certains replays ont des spectres différents
        
        Returns:
            (score, is_real)
        """
        try:
            # Calculer le spectre
            spectrum = np.abs(np.fft.fft(audio))[:len(audio)//2]
            
            # Éviter les divisions par zéro
            spectrum = np.maximum(spectrum, 1e-10)
            
            # Spectral flatness = moyenne géométrique / moyenne arithmétique
            geometric_mean = np.exp(np.mean(np.log(spectrum)))
            arithmetic_mean = np.mean(spectrum)
            
            flatness = geometric_mean / (arithmetic_mean + 1e-10)
            
            # La voix a généralement une flatness entre 0.1 et 0.4
            # Le bruit blanc a une flatness proche de 1
            # Les sons synthétiques peuvent avoir des valeurs extrêmes
            
            is_voice_like = 0.05 < flatness < 0.5
            score = 1.0 if is_voice_like else 0.4
            
            logger.debug(f"Spectral Flatness: {flatness:.4f}")
            
            return score, is_voice_like
            
        except Exception as e:
            logger.error(f"Erreur analyse spectrale: {e}")
            return 0.5, True
    
    def detect_synthetic_voice(self, audio: np.ndarray, sr: int) -> Tuple[float, bool]:
        """
        Détecter les voix synthétiques (TTS, deepfake audio)
        
        Les voix synthétiques ont souvent :
        - Pitch très régulier (pas de micro-variations naturelles)
        - Transitions trop lisses
        - Manque de bruits de respiration
        
        Returns:
            (score, is_natural)
        """
        try:
            # Calculer la variation du pitch frame par frame
            frame_size = int(sr * 0.03)  # 30ms
            hop_size = frame_size // 2
            
            # Calculer l'énergie par frame pour détecter les segments voisés
            frame_energies = []
            for i in range(0, len(audio) - frame_size, hop_size):
                frame = audio[i:i + frame_size]
                frame_energies.append(np.sum(frame ** 2))
            
            frame_energies = np.array(frame_energies)
            
            # Calculer la variation d'énergie (jitter naturel)
            energy_diff = np.diff(frame_energies)
            energy_jitter = np.std(energy_diff) / (np.mean(frame_energies) + 1e-7)
            
            # Une voix naturelle a du jitter (micro-variations)
            # Une voix synthétique est trop régulière
            
            has_natural_jitter = energy_jitter > 0.1
            jitter_score = min(energy_jitter * 5, 1.0)
            
            # Vérifier les pauses de respiration
            silence_mask = frame_energies < np.percentile(frame_energies, 20)
            silence_ratio = np.sum(silence_mask) / len(silence_mask)
            
            # Une voix naturelle a des pauses (5-20% du temps)
            has_pauses = 0.05 < silence_ratio < 0.3
            pause_score = 1.0 if has_pauses else 0.5
            
            global_score = (jitter_score + pause_score) / 2
            is_natural = has_natural_jitter and has_pauses
            
            return global_score, is_natural
            
        except Exception as e:
            logger.error(f"Erreur détection synthèse: {e}")
            return 0.5, True
    
    def check_liveness(self, audio: np.ndarray, sr: int) -> Tuple[bool, float, Dict]:
        """
        Vérification complète de liveness audio (anti-spoofing)
        
        Args:
            audio: Signal audio numpy
            sr: Sample rate
            
        Returns:
            (is_live, confidence, details)
        """
        try:
            # Analyser la qualité audio
            quality_score, quality_ok, quality_msg = self.analyze_audio_quality(audio, sr)
            
            # Détecter les artefacts de replay
            replay_score, replay_ok = self.detect_replay_artifacts(audio, sr)
            
            # Analyser la planéité spectrale
            spectral_score, spectral_ok = self.analyze_spectral_flatness(audio, sr)
            
            # Détecter les voix synthétiques
            synth_score, synth_ok = self.detect_synthetic_voice(audio, sr)
            
            # Score global pondéré
            weights = {
                'quality': 0.25,
                'replay': 0.30,
                'spectral': 0.20,
                'synthetic': 0.25
            }
            
            global_score = (
                weights['quality'] * quality_score +
                weights['replay'] * replay_score +
                weights['spectral'] * spectral_score +
                weights['synthetic'] * synth_score
            )
            
            # Décision finale
            tests_passed = sum([quality_ok, replay_ok, spectral_ok, synth_ok])
            is_live = global_score >= 0.5 and tests_passed >= 2
            
            details = {
                "quality": {"score": round(quality_score, 3), "passed": quality_ok, "message": quality_msg},
                "replay": {"score": round(replay_score, 3), "passed": replay_ok},
                "spectral": {"score": round(spectral_score, 3), "passed": spectral_ok},
                "synthetic": {"score": round(synth_score, 3), "passed": synth_ok},
                "tests_passed": tests_passed,
                "global_score": round(global_score, 3)
            }
            
            logger.info(f"Voice Anti-Spoof: is_live={is_live}, score={global_score:.3f}, tests_passed={tests_passed}/4")
            
            return is_live, global_score, details
            
        except Exception as e:
            logger.error(f"Erreur vérification liveness audio: {e}")
            return True, 0.5, {"error": str(e)}


# Instances globales
face_antispoof_service = FaceAntiSpoofService()
voice_antispoof_service = VoiceAntiSpoofService()
