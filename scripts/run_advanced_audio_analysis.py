"""
Advanced Audio Analysis Runner

Runs comprehensive audio analysis on processed audio files using
librosa, torchaudio, Pydub, and enhanced Whisper capabilities.
"""

import logging
import datetime
import json
from pathlib import Path
from course_compiler.config import (
    AUDIO_PREPPED_DIR, VIDEO_PREPPED_DIR, ANALYSIS_DIR, LOGS_DIR
)
from course_compiler.advanced_audio_analysis import AdvancedAudioAnalyzer, analyze_audio_directory

# Set up logging
Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
logfile = Path(LOGS_DIR) / f"advanced_audio_analysis_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"
logging.basicConfig(
    filename=logfile,
    filemode="a",
    format="%(asctime)s %(levelname)s:%(message)s",
    level=logging.INFO
)

def main():
    print("=== Advanced Audio Analysis ===")
    logging.info("Starting advanced audio analysis...")

    # Create analysis output directory
    audio_analysis_dir = Path(ANALYSIS_DIR) / "audio_features"
    audio_analysis_dir.mkdir(parents=True, exist_ok=True)

    # Configuration for advanced analysis
    config = {
        'sample_rate': 16000,
        'whisper_model': 'base',  # Can be: tiny, base, small, medium, large
        'output_dir': str(audio_analysis_dir)
    }

    # Analyze audio from processed directories
    directories_to_analyze = [
        (AUDIO_PREPPED_DIR, "audio_prepped"),
        (VIDEO_PREPPED_DIR, "video_prepped")
    ]

    all_results = {}
    total_files = 0

    for directory, source_type in directories_to_analyze:
        dir_path = Path(directory)
        if not dir_path.exists():
            print(f"Directory not found: {directory}")
            logging.warning(f"Directory not found: {directory}")
            continue

        print(f"\nAnalyzing {source_type} files in: {directory}")
        logging.info(f"Analyzing {source_type} files in: {directory}")

        # Count audio files
        audio_extensions = {'.wav', '.mp3', '.m4a', '.aac', '.flac', '.ogg'}
        audio_files = [f for f in dir_path.rglob('*') if f.suffix.lower() in audio_extensions]

        if not audio_files:
            print(f"No audio files found in {directory}")
            logging.info(f"No audio files found in {directory}")
            continue

        print(f"Found {len(audio_files)} audio files")
        total_files += len(audio_files)

        # Run analysis
        try:
            results = analyze_audio_directory(
                input_dir=str(dir_path),
                output_dir=str(audio_analysis_dir / source_type),
                config=config
            )
            all_results[source_type] = results
            print(f"✅ Completed analysis for {len(results)} files from {source_type}")
            logging.info(f"Completed analysis for {len(results)} files from {source_type}")

        except Exception as e:
            print(f"❌ Error analyzing {source_type}: {e}")
            logging.error(f"Error analyzing {source_type}: {e}", exc_info=True)
            all_results[source_type] = {'error': str(e)}

    # Generate summary report
    generate_audio_analysis_summary(all_results, audio_analysis_dir)

    print(f"\n=== Advanced Audio Analysis Complete ===")
    print(f"📁 Analysis results saved to: {audio_analysis_dir}")
    print(f"📊 Total files analyzed: {total_files}")
    print(f"📋 Detailed logs: {logfile}")
    logging.info(f"Advanced audio analysis complete. Total files: {total_files}")

def generate_audio_analysis_summary(all_results, output_dir):
    """Generate a summary report of audio analysis results."""
    try:
        summary = {
            'analysis_timestamp': str(datetime.datetime.now()),
            'total_sources': len(all_results),
            'summary_stats': {},
            'quality_insights': {},
            'engagement_insights': {},
            'accessibility_insights': {},
            'recommendations': []
        }

        # Aggregate statistics across all analyzed files
        all_quality_metrics = []
        all_engagement_metrics = []
        all_accessibility_metrics = []

        for source_type, results in all_results.items():
            if isinstance(results, dict) and 'error' not in results:
                source_files = 0
                for file_path, analysis in results.items():
                    if isinstance(analysis, dict) and 'error' not in analysis:
                        source_files += 1

                        # Collect quality metrics
                        if 'quality_metrics' in analysis:
                            all_quality_metrics.append(analysis['quality_metrics'])

                        # Collect engagement metrics
                        if 'engagement_metrics' in analysis:
                            all_engagement_metrics.append(analysis['engagement_metrics'])

                        # Collect accessibility metrics
                        if 'accessibility' in analysis:
                            all_accessibility_metrics.append(analysis['accessibility'])

                summary['summary_stats'][source_type] = {
                    'files_analyzed': source_files,
                    'files_with_errors': len([r for r in results.values() if isinstance(r, dict) and 'error' in r])
                }

        # Generate insights
        if all_quality_metrics:
            summary['quality_insights'] = analyze_quality_trends(all_quality_metrics)

        if all_engagement_metrics:
            summary['engagement_insights'] = analyze_engagement_trends(all_engagement_metrics)

        if all_accessibility_metrics:
            summary['accessibility_insights'] = analyze_accessibility_trends(all_accessibility_metrics)

        # Generate recommendations
        summary['recommendations'] = generate_recommendations(summary)

        # Save summary
        summary_path = Path(output_dir) / "audio_analysis_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"📊 Analysis summary saved to: {summary_path}")
        logging.info(f"Analysis summary saved to: {summary_path}")

    except Exception as e:
        print(f"❌ Error generating summary: {e}")
        logging.error(f"Error generating summary: {e}", exc_info=True)

def analyze_quality_trends(quality_metrics):
    """Analyze trends in audio quality."""
    try:
        import numpy as np

        snr_values = [q.get('snr_estimate', 0) for q in quality_metrics if q.get('snr_estimate')]
        rms_values = [q.get('rms_energy', 0) for q in quality_metrics if q.get('rms_energy')]

        insights = {}
        if snr_values:
            insights['avg_snr'] = float(np.mean(snr_values))
            insights['min_snr'] = float(np.min(snr_values))
            insights['max_snr'] = float(np.max(snr_values))
            insights['low_quality_files'] = len([snr for snr in snr_values if snr < 10])

        if rms_values:
            insights['avg_energy'] = float(np.mean(rms_values))
            insights['energy_consistency'] = float(np.std(rms_values))

        return insights
    except:
        return {}

def analyze_engagement_trends(engagement_metrics):
    """Analyze trends in engagement metrics."""
    try:
        import numpy as np

        pitch_variations = [e.get('pitch_variation_coefficient', 0) for e in engagement_metrics if e.get('pitch_variation_coefficient')]
        volume_dynamics = [e.get('volume_dynamics', 0) for e in engagement_metrics if e.get('volume_dynamics')]

        insights = {}
        if pitch_variations:
            insights['avg_pitch_variation'] = float(np.mean(pitch_variations))
            insights['high_variation_files'] = len([p for p in pitch_variations if p > 0.1])

        if volume_dynamics:
            insights['avg_volume_dynamics'] = float(np.mean(volume_dynamics))
            insights['dynamic_files'] = len([v for v in volume_dynamics if v > 0.2])

        return insights
    except:
        return {}

def analyze_accessibility_trends(accessibility_metrics):
    """Analyze trends in accessibility metrics."""
    try:
        import numpy as np

        speaking_rates = [a.get('speaking_rate_wpm', 0) for a in accessibility_metrics if a.get('speaking_rate_wpm')]
        clarity_scores = [a.get('clarity_score', 0) for a in accessibility_metrics if a.get('clarity_score')]

        insights = {}
        if speaking_rates:
            insights['avg_speaking_rate'] = float(np.mean(speaking_rates))
            insights['fast_speakers'] = len([r for r in speaking_rates if r > 200])
            insights['slow_speakers'] = len([r for r in speaking_rates if r < 140])

        if clarity_scores:
            insights['avg_clarity'] = float(np.mean(clarity_scores))
            insights['low_clarity_files'] = len([c for c in clarity_scores if c < 1000])

        return insights
    except:
        return {}

def generate_recommendations(summary):
    """Generate actionable recommendations based on analysis."""
    recommendations = []

    # Quality recommendations
    quality = summary.get('quality_insights', {})
    if quality.get('low_quality_files', 0) > 0:
        recommendations.append({
            'category': 'Audio Quality',
            'issue': f"{quality['low_quality_files']} files have low SNR (< 10dB)",
            'recommendation': 'Consider noise reduction or re-recording in quieter environments',
            'priority': 'high' if quality['low_quality_files'] > 2 else 'medium'
        })

    # Engagement recommendations
    engagement = summary.get('engagement_insights', {})
    if engagement.get('avg_pitch_variation', 0) < 0.05:
        recommendations.append({
            'category': 'Engagement',
            'issue': 'Low pitch variation detected across audio files',
            'recommendation': 'Encourage more dynamic speech patterns and vocal variety',
            'priority': 'medium'
        })

    # Accessibility recommendations
    accessibility = summary.get('accessibility_insights', {})
    if accessibility.get('fast_speakers', 0) > 0:
        recommendations.append({
            'category': 'Accessibility',
            'issue': f"{accessibility['fast_speakers']} files have speaking rate > 200 WPM",
            'recommendation': 'Consider slower delivery for better comprehension',
            'priority': 'medium'
        })

    if accessibility.get('slow_speakers', 0) > 0:
        recommendations.append({
            'category': 'Accessibility',
            'issue': f"{accessibility['slow_speakers']} files have speaking rate < 140 WPM",
            'recommendation': 'Consider slightly faster delivery to maintain engagement',
            'priority': 'low'
        })

    return recommendations

if __name__ == "__main__":
    main()
