"""
Advanced Audio Analysis Module

Leverages librosa, torchaudio, Pydub, and enhanced Whisper capabilities
to extract comprehensive audio features and insights for educational content.
"""

import os
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from datetime import timedelta

# Core audio processing
try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    librosa = None

try:
    import torch
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    torch = None
    torchaudio = None

try:
    from pydub import AudioSegment
    from pydub.silence import split_on_silence
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    AudioSegment = None

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    whisper = None

# Additional analysis libraries
try:
    from scipy import signal, stats
    import matplotlib.pyplot as plt
    SCIPY_MATPLOTLIB_AVAILABLE = True
except ImportError:
    SCIPY_MATPLOTLIB_AVAILABLE = False

from .logging_utils import get_logger

logger = get_logger(__name__)

class AdvancedAudioAnalyzer:
    """Comprehensive audio analysis using multiple libraries."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.sample_rate = self.config.get('sample_rate', 16000)
        self.whisper_model_size = self.config.get('whisper_model', 'base')
        self.output_dir = self.config.get('output_dir', 'output/audio_analysis')

        # Initialize models
        self.whisper_model = None
        self._load_models()

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_models(self):
        """Load AI models for audio analysis."""
        if WHISPER_AVAILABLE:
            try:
                self.whisper_model = whisper.load_model(self.whisper_model_size)
                logger.info(f"Loaded Whisper model: {self.whisper_model_size}")
            except Exception as e:
                logger.warning(f"Failed to load Whisper model: {e}")

    def analyze_audio_file(self, audio_path: str) -> Dict[str, Any]:
        """
        Comprehensive analysis of an audio file.

        Returns:
            Dictionary containing all extracted features and analysis results
        """
        audio_path = Path(audio_path)
        logger.info(f"Starting comprehensive audio analysis: {audio_path.name}")

        results = {
            'file_path': str(audio_path),
            'file_name': audio_path.name,
            'analysis_timestamp': str(pd.Timestamp.now()),
            'available_libraries': {
                'librosa': LIBROSA_AVAILABLE,
                'torchaudio': TORCHAUDIO_AVAILABLE,
                'pydub': PYDUB_AVAILABLE,
                'whisper': WHISPER_AVAILABLE,
                'scipy_matplotlib': SCIPY_MATPLOTLIB_AVAILABLE
            }
        }

        try:
            # 1. Enhanced Transcription with Word-Level Timing
            if WHISPER_AVAILABLE and self.whisper_model:
                results['transcription'] = self._enhanced_transcription(audio_path)

            # 2. Audio Quality Assessment
            results['quality_metrics'] = self._assess_audio_quality(audio_path)

            # 3. Spectral Feature Analysis
            if LIBROSA_AVAILABLE:
                results['spectral_features'] = self._extract_spectral_features(audio_path)

            # 4. Silence and Speech Activity Detection
            results['speech_analysis'] = self._analyze_speech_activity(audio_path)

            # 5. Emotional and Engagement Analysis
            results['engagement_metrics'] = self._analyze_engagement(audio_path)

            # 6. Speaker Analysis (basic)
            results['speaker_analysis'] = self._analyze_speakers(audio_path)

            # 7. Audio Accessibility Assessment
            results['accessibility'] = self._assess_accessibility(audio_path)

            # 8. Content Structure Analysis
            results['structure_analysis'] = self._analyze_content_structure(audio_path)

            # 9. Audio Fingerprinting
            results['fingerprint'] = self._generate_audio_fingerprint(audio_path)

            logger.info(f"Audio analysis complete: {audio_path.name}")

        except Exception as e:
            logger.error(f"Error in audio analysis: {e}", exc_info=True)
            results['error'] = str(e)

        return results

    def _enhanced_transcription(self, audio_path: str) -> Dict[str, Any]:
        """Enhanced transcription with word-level timing and confidence."""
        try:
            result = self.whisper_model.transcribe(
                str(audio_path),
                word_timestamps=True,
                verbose=False
            )

            # Extract word-level details
            words = []
            if 'segments' in result:
                for segment in result['segments']:
                    if 'words' in segment:
                        for word in segment['words']:
                            words.append({
                                'word': word.get('word', '').strip(),
                                'start': word.get('start', 0),
                                'end': word.get('end', 0),
                                'confidence': word.get('probability', 0)
                            })

            return {
                'text': result.get('text', ''),
                'language': result.get('language', 'unknown'),
                'segments': result.get('segments', []),
                'words': words,
                'word_count': len(words),
                'avg_confidence': np.mean([w['confidence'] for w in words]) if words else 0,
                'low_confidence_words': [w for w in words if w['confidence'] < 0.8]
            }

        except Exception as e:
            logger.error(f"Enhanced transcription failed: {e}")
            return {'error': str(e)}

    def _assess_audio_quality(self, audio_path: str) -> Dict[str, Any]:
        """Assess audio quality using multiple metrics."""
        try:
            quality = {}

            if LIBROSA_AVAILABLE:
                # Load audio
                y, sr = librosa.load(str(audio_path), sr=self.sample_rate)

                # Signal-to-noise ratio estimation
                quality['snr_estimate'] = self._estimate_snr(y)

                # Dynamic range
                quality['dynamic_range'] = np.max(y) - np.min(y)

                # RMS energy
                quality['rms_energy'] = float(np.sqrt(np.mean(y**2)))

                # Zero crossing rate (measure of noisiness)
                quality['zero_crossing_rate'] = float(np.mean(librosa.feature.zero_crossing_rate(y)))

                # Spectral centroid (brightness)
                spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
                quality['spectral_centroid_mean'] = float(np.mean(spectral_centroids))

                # Spectral rolloff
                spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                quality['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))

            if PYDUB_AVAILABLE:
                # Load with pydub for additional metrics
                audio = AudioSegment.from_file(str(audio_path))
                quality['loudness_lufs'] = audio.dBFS
                quality['max_possible_loudness'] = audio.max_possible_amplitude

            return quality

        except Exception as e:
            logger.error(f"Audio quality assessment failed: {e}")
            return {'error': str(e)}

    def _extract_spectral_features(self, audio_path: str) -> Dict[str, Any]:
        """Extract comprehensive spectral features using librosa."""
        try:
            y, sr = librosa.load(str(audio_path), sr=self.sample_rate)

            features = {}

            # MFCCs (Mel-frequency cepstral coefficients)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features['mfccs'] = {
                'mean': mfccs.mean(axis=1).tolist(),
                'std': mfccs.std(axis=1).tolist()
            }

            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features['chroma'] = {
                'mean': chroma.mean(axis=1).tolist(),
                'std': chroma.std(axis=1).tolist()
            }

            # Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            features['spectral_contrast'] = {
                'mean': contrast.mean(axis=1).tolist(),
                'std': contrast.std(axis=1).tolist()
            }

            # Tonnetz (harmonic features)
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            features['tonnetz'] = {
                'mean': tonnetz.mean(axis=1).tolist(),
                'std': tonnetz.std(axis=1).tolist()
            }

            # Tempo and beat tracking
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = float(tempo)
            features['beat_count'] = len(beats)

            # Onset detection
            onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
            features['onset_count'] = len(onsets)
            features['onset_rate'] = len(onsets) / len(y) * sr  # onsets per second

            return features

        except Exception as e:
            logger.error(f"Spectral feature extraction failed: {e}")
            return {'error': str(e)}

    def _analyze_speech_activity(self, audio_path: str) -> Dict[str, Any]:
        """Analyze silence, speech activity, and pacing."""
        try:
            analysis = {}

            if PYDUB_AVAILABLE:
                # Load audio
                audio = AudioSegment.from_file(str(audio_path))

                # Split on silence
                silence_thresh = audio.dBFS - 16  # 16dB below average
                chunks = split_on_silence(
                    audio,
                    min_silence_len=500,  # 500ms
                    silence_thresh=silence_thresh,
                    keep_silence=100
                )

                analysis['speech_segments'] = len(chunks)
                analysis['silence_threshold_db'] = silence_thresh

                # Calculate speech vs silence ratio
                total_duration = len(audio)
                speech_duration = sum(len(chunk) for chunk in chunks)
                silence_duration = total_duration - speech_duration

                analysis['total_duration_ms'] = total_duration
                analysis['speech_duration_ms'] = speech_duration
                analysis['silence_duration_ms'] = silence_duration
                analysis['speech_ratio'] = speech_duration / total_duration if total_duration > 0 else 0

                # Pacing analysis
                if chunks:
                    chunk_lengths = [len(chunk) for chunk in chunks]
                    analysis['avg_speech_segment_ms'] = np.mean(chunk_lengths)
                    analysis['speech_segment_variance'] = np.var(chunk_lengths)
                    analysis['speaking_rate_consistency'] = 1 / (1 + np.std(chunk_lengths) / np.mean(chunk_lengths))

            if LIBROSA_AVAILABLE:
                # Voice activity detection using energy
                y, sr = librosa.load(str(audio_path), sr=self.sample_rate)

                # Frame-level energy
                frame_length = int(0.025 * sr)  # 25ms frames
                hop_length = int(0.010 * sr)    # 10ms hop

                # Short-time energy
                energy = []
                for i in range(0, len(y) - frame_length, hop_length):
                    frame = y[i:i+frame_length]
                    energy.append(np.sum(frame**2))

                energy = np.array(energy)
                energy_threshold = np.mean(energy) * 0.1  # 10% of mean energy

                voiced_frames = np.sum(energy > energy_threshold)
                total_frames = len(energy)

                analysis['voice_activity_ratio'] = voiced_frames / total_frames if total_frames > 0 else 0
                analysis['avg_frame_energy'] = float(np.mean(energy))
                analysis['energy_variance'] = float(np.var(energy))

            return analysis

        except Exception as e:
            logger.error(f"Speech activity analysis failed: {e}")
            return {'error': str(e)}

    def _analyze_engagement(self, audio_path: str) -> Dict[str, Any]:
        """Analyze audio features that correlate with engagement."""
        try:
            metrics = {}

            if LIBROSA_AVAILABLE:
                y, sr = librosa.load(str(audio_path), sr=self.sample_rate)

                # Pitch variation (engagement often correlates with pitch variation)
                pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
                pitch_values = []
                for t in range(pitches.shape[1]):
                    index = magnitudes[:, t].argmax()
                    pitch = pitches[index, t]
                    if pitch > 0:
                        pitch_values.append(pitch)

                if pitch_values:
                    metrics['pitch_mean'] = float(np.mean(pitch_values))
                    metrics['pitch_std'] = float(np.std(pitch_values))
                    metrics['pitch_range'] = float(np.max(pitch_values) - np.min(pitch_values))
                    metrics['pitch_variation_coefficient'] = metrics['pitch_std'] / metrics['pitch_mean']

                # Volume dynamics
                rms = librosa.feature.rms(y=y)
                metrics['volume_mean'] = float(np.mean(rms))
                metrics['volume_std'] = float(np.std(rms))
                metrics['volume_dynamics'] = metrics['volume_std'] / metrics['volume_mean']

                # Speaking rate estimation (using onset detection)
                onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
                duration = len(y) / sr
                metrics['onset_density'] = len(onsets) / duration  # onsets per second

                # Spectral flux (measure of spectral change)
                stft = librosa.stft(y)
                spectral_flux = np.sum(np.diff(np.abs(stft), axis=1)**2, axis=0)
                metrics['spectral_flux_mean'] = float(np.mean(spectral_flux))
                metrics['spectral_flux_std'] = float(np.std(spectral_flux))

            return metrics

        except Exception as e:
            logger.error(f"Engagement analysis failed: {e}")
            return {'error': str(e)}

    def _analyze_speakers(self, audio_path: str) -> Dict[str, Any]:
        """Basic speaker analysis and diarization preparation."""
        try:
            analysis = {}

            if LIBROSA_AVAILABLE:
                y, sr = librosa.load(str(audio_path), sr=self.sample_rate)

                # Fundamental frequency analysis (voice characteristics)
                pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

                # Spectral centroid (voice brightness)
                spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
                analysis['voice_brightness'] = float(np.mean(spectral_centroids))

                # MFCC-based voice characteristics
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                analysis['voice_signature'] = mfccs.mean(axis=1)[:5].tolist()  # First 5 MFCCs

                # Estimate speaker consistency (single vs multiple speakers)
                # Simple approach: analyze MFCC variance over time
                mfcc_variance = np.var(mfccs, axis=1)
                analysis['voice_consistency'] = float(np.mean(mfcc_variance))
                analysis['likely_single_speaker'] = analysis['voice_consistency'] < 100  # Threshold

            return analysis

        except Exception as e:
            logger.error(f"Speaker analysis failed: {e}")
            return {'error': str(e)}

    def _assess_accessibility(self, audio_path: str) -> Dict[str, Any]:
        """Assess audio accessibility features."""
        try:
            accessibility = {}

            if LIBROSA_AVAILABLE:
                y, sr = librosa.load(str(audio_path), sr=self.sample_rate)

                # Speaking rate (words per minute estimate)
                if hasattr(self, 'transcription') and 'words' in self.transcription:
                    duration_minutes = len(y) / sr / 60
                    word_count = len(self.transcription['words'])
                    accessibility['speaking_rate_wpm'] = word_count / duration_minutes

                    # Rate assessment
                    if accessibility['speaking_rate_wpm'] < 140:
                        accessibility['speaking_rate_assessment'] = 'slow'
                    elif accessibility['speaking_rate_wpm'] > 200:
                        accessibility['speaking_rate_assessment'] = 'fast'
                    else:
                        accessibility['speaking_rate_assessment'] = 'normal'

                # Audio clarity metrics
                spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
                accessibility['clarity_score'] = float(np.mean(spectral_centroids))

                # Volume consistency
                rms = librosa.feature.rms(y=y)
                accessibility['volume_consistency'] = 1 / (1 + np.std(rms) / np.mean(rms))

                # Frequency range (important for hearing accessibility)
                stft = librosa.stft(y)
                freqs = librosa.fft_frequencies(sr=sr)
                magnitude = np.abs(stft)

                # Energy in speech frequency range (300-3400 Hz)
                speech_freq_mask = (freqs >= 300) & (freqs <= 3400)
                speech_energy = np.sum(magnitude[speech_freq_mask, :])
                total_energy = np.sum(magnitude)
                accessibility['speech_frequency_ratio'] = speech_energy / total_energy

            return accessibility

        except Exception as e:
            logger.error(f"Accessibility assessment failed: {e}")
            return {'error': str(e)}

    def _analyze_content_structure(self, audio_path: str) -> Dict[str, Any]:
        """Analyze the structural elements of the audio content."""
        try:
            structure = {}

            if LIBROSA_AVAILABLE:
                y, sr = librosa.load(str(audio_path), sr=self.sample_rate)

                # Segment boundaries using onset detection
                onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
                structure['onset_times'] = onsets.tolist()
                structure['estimated_segments'] = len(onsets)

                # Tempo changes (indicates section changes)
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
                structure['overall_tempo'] = float(tempo)

                # Novelty curve (indicates structural changes)
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                novelty = np.sum(np.diff(chroma, axis=1)**2, axis=0)
                novelty_peaks = librosa.util.peak_pick(novelty, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10)
                structure['structure_change_times'] = (novelty_peaks * 512 / sr).tolist()  # Convert to seconds
                structure['estimated_structural_segments'] = len(novelty_peaks)

            return structure

        except Exception as e:
            logger.error(f"Content structure analysis failed: {e}")
            return {'error': str(e)}

    def _generate_audio_fingerprint(self, audio_path: str) -> Dict[str, Any]:
        """Generate audio fingerprint for duplicate detection."""
        try:
            fingerprint = {}

            if LIBROSA_AVAILABLE:
                y, sr = librosa.load(str(audio_path), sr=self.sample_rate)

                # Chromagram-based fingerprint
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                fingerprint['chroma_fingerprint'] = chroma.mean(axis=1).tolist()

                # MFCC-based fingerprint
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                fingerprint['mfcc_fingerprint'] = mfccs.mean(axis=1).tolist()

                # Spectral centroid fingerprint
                spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
                fingerprint['spectral_fingerprint'] = [float(np.mean(spectral_centroids)), float(np.std(spectral_centroids))]

                # Simple hash based on audio characteristics
                feature_vector = np.concatenate([
                    chroma.mean(axis=1),
                    mfccs.mean(axis=1),
                    [np.mean(spectral_centroids), np.std(spectral_centroids)]
                ])
                fingerprint['feature_hash'] = hash(tuple(np.round(feature_vector, 3)))

            return fingerprint

        except Exception as e:
            logger.error(f"Audio fingerprinting failed: {e}")
            return {'error': str(e)}

    def _estimate_snr(self, y: np.ndarray) -> float:
        """Estimate signal-to-noise ratio."""
        try:
            # Simple SNR estimation: ratio of signal power to noise power
            # Assume the quietest 10% represents noise
            sorted_energy = np.sort(y**2)
            noise_floor = np.mean(sorted_energy[:int(0.1 * len(sorted_energy))])
            signal_power = np.mean(y**2)

            if noise_floor > 0:
                snr = 10 * np.log10(signal_power / noise_floor)
            else:
                snr = float('inf')

            return float(snr)
        except:
            return 0.0

# Convenience function for batch processing
def analyze_audio_directory(input_dir: str, output_dir: str, config: Optional[Dict] = None) -> Dict[str, Any]:
    """Analyze all audio files in a directory."""
    analyzer = AdvancedAudioAnalyzer(config)
    results = {}

    audio_extensions = {'.wav', '.mp3', '.m4a', '.aac', '.flac', '.ogg'}

    for audio_file in Path(input_dir).rglob('*'):
        if audio_file.suffix.lower() in audio_extensions:
            try:
                analysis = analyzer.analyze_audio_file(str(audio_file))
                results[str(audio_file)] = analysis

                # Save individual analysis
                output_file = Path(output_dir) / f"{audio_file.stem}_analysis.json"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w') as f:
                    json.dump(analysis, f, indent=2, default=str)

            except Exception as e:
                logger.error(f"Failed to analyze {audio_file}: {e}")
                results[str(audio_file)] = {'error': str(e)}

    return results

# Fix missing import
try:
    import pandas as pd
except ImportError:
    import datetime as pd
    pd.Timestamp = datetime.datetime
