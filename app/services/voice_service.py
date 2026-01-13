"""
Service de reconnaissance vocale
"""
import numpy as np
import base64
import io
import librosa
import soundfile as sf
from scipy.spatial.distance import cosine
from typing import Optional, Tuple, List
import logging
import random
import string
from datetime import datetime, timedelta
import tempfile
import os
import subprocess
import shutil
from app.config import settings

logger = logging.getLogger(__name__)


def find_ffmpeg() -> Optional[str]:
    """Trouver le chemin de ffmpeg sur le système"""
    # D'abord vérifier dans le PATH
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        return ffmpeg_path
    
    # Emplacements courants sur Windows
    common_paths = [
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\tools\ffmpeg\bin\ffmpeg.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-7.1-full_build\bin\ffmpeg.exe"),
    ]
    
    # Chercher dans WinGet packages
    winget_path = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages")
    if os.path.exists(winget_path):
        for folder in os.listdir(winget_path):
            if 'ffmpeg' in folder.lower():
                potential = os.path.join(winget_path, folder)
                for root, dirs, files in os.walk(potential):
                    if 'ffmpeg.exe' in files:
                        return os.path.join(root, 'ffmpeg.exe')
    
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    return None


# Trouver ffmpeg au démarrage
FFMPEG_PATH = find_ffmpeg()
if FFMPEG_PATH:
    logger.info(f"FFmpeg trouvé: {FFMPEG_PATH}")
else:
    logger.warning("FFmpeg non trouvé - la conversion audio sera limitée")


# Texte fixe pour la vérification vocale pendant l'examen
VOICE_CHALLENGES = [
    "Bonjour, je confirme mon identité pour passer cet examen."
]


class VoiceRecognitionService:
    """Service pour la reconnaissance vocale"""
    
    def __init__(self, threshold: float = 0.85):
        """
        Initialiser le service
        Args:
            threshold: Seuil de similarité (0 à 1). Plus grand = plus similaire requis.
                      0.85 est plus strict pour éviter les faux positifs
        """
        self.threshold = threshold
        self.active_challenges = {}  # Stockage des défis en cours
    
    def _convert_audio_to_wav(self, audio_data: bytes, original_format: str = None) -> Optional[bytes]:
        """
        Convertir n'importe quel format audio en WAV en utilisant ffmpeg
        """
        if not FFMPEG_PATH:
            logger.warning("FFmpeg non disponible pour la conversion")
            return None
            
        try:
            # Créer des fichiers temporaires avec extension appropriée
            suffix = f'.{original_format}' if original_format else '.tmp'
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as input_file:
                input_file.write(audio_data)
                input_path = input_file.name
            
            output_path = input_path + '.wav'
            
            try:
                # Utiliser ffmpeg pour convertir
                logger.info(f"Conversion audio avec ffmpeg: {input_path} -> {output_path}")
                result = subprocess.run([
                    FFMPEG_PATH, '-y', '-i', input_path,
                    '-acodec', 'pcm_s16le',
                    '-ar', '16000',
                    '-ac', '1',
                    output_path
                ], capture_output=True, timeout=30)
                
                if result.returncode == 0 and os.path.exists(output_path):
                    with open(output_path, 'rb') as f:
                        wav_data = f.read()
                    logger.info(f"Conversion réussie: {len(wav_data)} bytes")
                    return wav_data
                else:
                    logger.error(f"ffmpeg error: {result.stderr.decode()}")
                    return None
            finally:
                # Nettoyer les fichiers temporaires
                if os.path.exists(input_path):
                    os.unlink(input_path)
                if os.path.exists(output_path):
                    os.unlink(output_path)
                    
        except FileNotFoundError:
            logger.warning("ffmpeg non trouvé, tentative sans conversion")
            return None
        except Exception as e:
            logger.error(f"Erreur de conversion: {e}")
            return None
    
    def decode_base64_audio(self, audio_base64: str) -> Optional[Tuple[np.ndarray, int]]:
        """
        Décoder un audio base64 en array numpy
        Supporte plusieurs formats (WAV, WebM, MP3, AAC, etc.)
        Returns:
            Tuple (audio_data, sample_rate) ou None
        """
        try:
            # Retirer le préfixe data:audio si présent
            original_format = None
            if ',' in audio_base64:
                header = audio_base64.split(',')[0].lower()
                logger.info(f"Header audio détecté: {header}")
                if 'webm' in header:
                    original_format = 'webm'
                elif 'mp3' in header or 'mpeg' in header:
                    original_format = 'mp3'
                elif 'wav' in header:
                    original_format = 'wav'
                elif 'ogg' in header:
                    original_format = 'ogg'
                elif 'aac' in header or 'm4a' in header or 'mp4' in header:
                    original_format = 'aac'
                elif 'x-m4a' in header:
                    original_format = 'm4a'
                audio_base64 = audio_base64.split(',')[1]
            
            # Décoder base64
            audio_data = base64.b64decode(audio_base64)
            logger.info(f"Audio décodé depuis base64: {len(audio_data)} bytes, format: {original_format}")
            
            # Si format non-WAV, convertir d'abord avec ffmpeg
            if original_format and original_format != 'wav':
                logger.info(f"Format {original_format} détecté, conversion avec ffmpeg...")
                wav_data = self._convert_audio_to_wav(audio_data, original_format)
                if wav_data:
                    audio_buffer = io.BytesIO(wav_data)
                    y, sr = sf.read(audio_buffer)
                    logger.info(f"Audio converti et chargé: {len(y)} échantillons, {sr} Hz")
                else:
                    logger.error("Échec de la conversion ffmpeg")
                    return None
            else:
                # Essayer de lire directement avec soundfile
                audio_buffer = io.BytesIO(audio_data)
                try:
                    y, sr = sf.read(audio_buffer)
                except Exception as e:
                    logger.info(f"Format non reconnu par soundfile, tentative de conversion avec ffmpeg...")
                    
                    # Convertir avec ffmpeg
                    wav_data = self._convert_audio_to_wav(audio_data, original_format)
                    if wav_data is None:
                        # Essayer avec librosa directement (plus lent mais supporte plus de formats)
                        try:
                            with tempfile.NamedTemporaryFile(suffix=f'.{original_format or "tmp"}', delete=False) as tmp:
                                tmp.write(audio_data)
                                tmp_path = tmp.name
                            try:
                                y, sr = librosa.load(tmp_path, sr=16000)
                                logger.info(f"Audio chargé avec librosa: {len(y)} échantillons")
                            finally:
                                if os.path.exists(tmp_path):
                                    os.unlink(tmp_path)
                        except Exception as e2:
                            logger.error(f"Impossible de décoder l'audio: {e2}")
                            return None
                    else:
                        wav_buffer = io.BytesIO(wav_data)
                        y, sr = sf.read(wav_buffer)
            
            # Convertir en mono si stéréo
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)
            
            # S'assurer que l'audio n'est pas vide
            if len(y) == 0:
                logger.error("Audio vide reçu")
                return None
                
            logger.info(f"Audio décodé: {len(y)} échantillons, {sr} Hz")
            return y, sr
            
        except Exception as e:
            logger.error(f"Erreur de décodage audio: {e}")
            return None
    
    def extract_voice_features(self, audio: np.ndarray, sr: int) -> Optional[np.ndarray]:
        """
        Extraire les caractéristiques vocales améliorées pour discriminer les voix similaires
        Utilise MFCC + caractéristiques prosodiques + formants
        Args:
            audio: Signal audio
            sr: Fréquence d'échantillonnage
        Returns:
            Vecteur de caractéristiques moyennes
        """
        try:
            # Rééchantillonner à 16kHz si nécessaire
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000
            
            # Extraire les MFCC (Mel-Frequency Cepstral Coefficients)
            # Utiliser plus de coefficients pour une meilleure discrimination
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)  # 20 coefficients au lieu de 13
            
            # Calculer les deltas (variations temporelles)
            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
            
            # Statistiques sur les MFCC
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            mfcc_max = np.max(mfccs, axis=1)
            mfcc_min = np.min(mfccs, axis=1)
            
            # Statistiques sur les deltas
            delta_mean = np.mean(mfcc_delta, axis=1)
            delta_std = np.std(mfcc_delta, axis=1)
            delta2_mean = np.mean(mfcc_delta2, axis=1)
            delta2_std = np.std(mfcc_delta2, axis=1)
            
            # === Caractéristiques supplémentaires pour discriminer les voix familiales ===
            
            # Fréquence fondamentale (F0) - très discriminante entre individus
            f0, voiced_flag, voiced_probs = librosa.pyin(audio, fmin=50, fmax=400, sr=sr)
            f0_clean = f0[~np.isnan(f0)] if len(f0[~np.isnan(f0)]) > 0 else np.array([0])
            f0_mean = np.mean(f0_clean)
            f0_std = np.std(f0_clean)
            f0_range = np.max(f0_clean) - np.min(f0_clean) if len(f0_clean) > 1 else 0
            
            # Spectral features - caractéristiques du timbre vocal
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            
            # Statistiques spectrales
            sc_mean = np.mean(spectral_centroid)
            sc_std = np.std(spectral_centroid)
            sb_mean = np.mean(spectral_bandwidth)
            sb_std = np.std(spectral_bandwidth)
            sr_mean = np.mean(spectral_rolloff)
            contrast_mean = np.mean(spectral_contrast, axis=1)
            
            # Zero crossing rate - lié à la texture vocale
            zcr = librosa.feature.zero_crossing_rate(audio)
            zcr_mean = np.mean(zcr)
            zcr_std = np.std(zcr)
            
            # RMS energy pattern
            rms = librosa.feature.rms(y=audio)
            rms_mean = np.mean(rms)
            rms_std = np.std(rms)
            
            # Combiner en un vecteur de caractéristiques plus riche
            # Garder les valeurs brutes SANS normalisation globale
            # Chaque caractéristique garde sa propre échelle
            features = np.concatenate([
                mfcc_mean, mfcc_std, mfcc_max, mfcc_min,
                delta_mean, delta_std, delta2_mean, delta2_std,
                [f0_mean, f0_std, f0_range],  # Fréquence fondamentale brute - TRÈS discriminante
                [sc_mean, sc_std, sb_mean, sb_std],  # Caractéristiques spectrales brutes
                [sr_mean, zcr_mean * 1000, zcr_std * 1000],  # Zero crossing rate
                [rms_mean * 100, rms_std * 100],  # RMS
                contrast_mean  # Contraste spectral brut
            ])
            
            # PAS DE NORMALISATION - garder les valeurs brutes pour discrimination
            # La F0 (fréquence fondamentale) varie typiquement de 85-180Hz (homme) à 165-255Hz (femme)
            # Cette différence est cruciale pour distinguer les individus
            
            logger.info(f"Caractéristiques extraites: {len(features)} dimensions, F0={f0_mean:.1f}Hz")
            
            return features
            
        except Exception as e:
            logger.error(f"Erreur d'extraction des caractéristiques vocales: {e}")
            return None
    
    def encode_to_bytes(self, encoding: np.ndarray) -> bytes:
        """Convertir un encoding numpy en bytes pour stockage"""
        return encoding.tobytes()
    
    def decode_from_bytes(self, data: bytes) -> np.ndarray:
        """Reconvertir des bytes en encoding numpy"""
        return np.frombuffer(data, dtype=np.float64)
    
    def compare_voices(
        self,
        known_encoding: np.ndarray,
        unknown_encoding: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Comparer deux encodings vocaux avec méthode stricte mais calibrée
        Args:
            known_encoding: Encoding enregistré
            unknown_encoding: Encoding à vérifier
        Returns:
            Tuple (match: bool, score: float entre 0 et 1)
        """
        try:
            logger.info(f"=== COMPARAISON VOCALE ===")
            logger.info(f"Taille encoding stocké: {len(known_encoding)}")
            logger.info(f"Taille encoding fourni: {len(unknown_encoding)}")
            
            # Vérifier que les tailles correspondent
            if len(known_encoding) != len(unknown_encoding):
                logger.error(f"❌ TAILLES DIFFÉRENTES! Réenrôlement nécessaire.")
                return False, 0.0
            
            # Calculer les différences par dimension
            diff = known_encoding - unknown_encoding
            abs_diff = np.abs(diff)
            
            # 1. Distance euclidienne
            euclidean_dist = np.linalg.norm(diff)
            
            # 2. Différence relative moyenne (plus robuste aux échelles)
            # Calculer la différence relative par rapport à la valeur moyenne
            avg_magnitude = (np.abs(known_encoding) + np.abs(unknown_encoding)) / 2 + 1e-8
            relative_diff = abs_diff / avg_magnitude
            mean_relative_diff = np.mean(relative_diff)
            
            # 3. Corrélation de Pearson (insensible à l'échelle)
            correlation = np.corrcoef(known_encoding, unknown_encoding)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
            
            # 4. Similarité cosinus (insensible à la magnitude)
            cosine_sim = np.dot(known_encoding, unknown_encoding) / (
                np.linalg.norm(known_encoding) * np.linalg.norm(unknown_encoding) + 1e-8
            )
            
            logger.info(f"Distance euclidienne: {euclidean_dist:.2f}")
            logger.info(f"Différence relative moyenne: {mean_relative_diff:.4f}")
            logger.info(f"Corrélation: {correlation:.4f}")
            logger.info(f"Similarité cosinus: {cosine_sim:.4f}")
            
            # Convertir en scores
            # Différence relative: même personne < 0.3, différente > 0.6
            relative_score = max(0, 1 - (mean_relative_diff / 0.8))
            
            # Corrélation: même personne > 0.9, différente < 0.7
            corr_score = max(0, (correlation - 0.5) / 0.5) if correlation > 0.5 else 0
            
            # Cosinus: même personne > 0.95, différente < 0.85
            cosine_score = max(0, (cosine_sim - 0.7) / 0.3) if cosine_sim > 0.7 else 0
            
            logger.info(f"Score différence relative: {relative_score:.4f}")
            logger.info(f"Score corrélation: {corr_score:.4f}")
            logger.info(f"Score cosinus: {cosine_score:.4f}")
            
            # Score final pondéré
            combined_score = (
                0.35 * relative_score +
                0.35 * corr_score +
                0.30 * cosine_score
            )
            
            logger.info(f"Score vocal combiné: {combined_score:.4f}")
            logger.info(f"Seuil requis: {self.threshold}")
            
            # Vérifier si c'est un match
            is_match = combined_score >= self.threshold
            
            logger.info(f"Résultat: {'✅ MATCH' if is_match else '❌ PAS DE MATCH'}")
            
            return is_match, combined_score
            
            return is_match, combined_score
            
            return is_match, combined_score
            
        except Exception as e:
            logger.error(f"Erreur de comparaison vocale: {e}")
            return False, 0.0
    
    def enroll_voice(self, audio_base64: str) -> Tuple[Optional[bytes], float]:
        """
        Enrôler une voix à partir d'un audio base64
        Returns:
            Tuple (encoding en bytes ou None, qualité)
        """
        # Décoder l'audio
        result = self.decode_base64_audio(audio_base64)
        if result is None:
            return None, 0.0
        
        audio, sr = result
        
        # Vérifier la durée minimale (au moins 2 secondes)
        duration = len(audio) / sr
        if duration < 2.0:
            logger.warning(f"Audio trop court: {duration}s")
            return None, 0.0
        
        # Extraire les caractéristiques
        features = self.extract_voice_features(audio, sr)
        if features is None:
            return None, 0.0
        
        # Évaluer la qualité (basée sur l'énergie du signal)
        energy = np.mean(audio ** 2)
        quality = min(1.0, energy * 1000)  # Normaliser
        
        return self.encode_to_bytes(features), quality
    
    def verify_voice(
        self,
        stored_encoding_bytes: bytes,
        audio_base64: str
    ) -> Tuple[bool, float]:
        """
        Vérifier une voix par rapport à l'encoding stocké
        Args:
            stored_encoding_bytes: Encoding stocké en bytes
            audio_base64: Audio à vérifier en base64
        Returns:
            Tuple (match: bool, score: float)
        """
        # Décoder l'audio
        result = self.decode_base64_audio(audio_base64)
        if result is None:
            return False, 0.0
        
        audio, sr = result
        
        # Extraire les caractéristiques
        unknown_encoding = self.extract_voice_features(audio, sr)
        if unknown_encoding is None:
            return False, 0.0
        
        # Récupérer l'encoding stocké
        known_encoding = self.decode_from_bytes(stored_encoding_bytes)
        
        # Comparer
        return self.compare_voices(known_encoding, unknown_encoding)
    
    def generate_challenge(self, user_id: int) -> Tuple[str, str, datetime]:
        """
        Générer un défi vocal pour un utilisateur
        Returns:
            Tuple (challenge_id, text_to_read, expires_at)
        """
        # Générer un ID unique
        challenge_id = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
        
        # Sélectionner un texte aléatoire
        text = random.choice(VOICE_CHALLENGES)
        
        # Définir l'expiration (60 secondes)
        expires_at = datetime.utcnow() + timedelta(seconds=60)
        
        # Stocker le défi
        self.active_challenges[challenge_id] = {
            "user_id": user_id,
            "text": text,
            "expires_at": expires_at
        }
        
        return challenge_id, text, expires_at
    
    def validate_challenge(self, challenge_id: str, user_id: int) -> Optional[str]:
        """
        Valider un défi vocal
        Returns:
            Le texte attendu si valide, None sinon
        """
        challenge = self.active_challenges.get(challenge_id)
        
        if challenge is None:
            return None
        
        # Vérifier l'utilisateur
        if challenge["user_id"] != user_id:
            return None
        
        # Vérifier l'expiration
        if datetime.utcnow() > challenge["expires_at"]:
            del self.active_challenges[challenge_id]
            return None
        
        # Supprimer le défi après utilisation
        text = challenge["text"]
        del self.active_challenges[challenge_id]
        
        return text


# Instance globale du service avec le seuil de configuration
voice_service = VoiceRecognitionService(threshold=settings.VOICE_RECOGNITION_THRESHOLD)
