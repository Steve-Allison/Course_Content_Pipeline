import subprocess
from pathlib import Path
import sys
from course_compiler.file_utils import sanitize_filename

def extract_audio_from_media(media_path, wav_path, log):
    """
    Extracts (or converts) audio from audio or video file to mono 16kHz WAV using ffmpeg.
    """
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", str(media_path),
            "-ar", "16000",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-f", "wav", str(wav_path)
        ], check=True, capture_output=True)
        log.append(f"SUCCESS: Audio extracted {media_path} -> {wav_path}")
        return True
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.decode() if e.stderr else str(e)
        log.append(f"ERROR: Failed audio extraction {media_path} -> {wav_path}: {err_msg}")
        return False

def convert_to_iframe_only_mp4(video_path, output_path, log):
    """
    Converts video to MP4 (I-frame only), for alignment/analysis.
    """
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", str(video_path),
            "-c:v", "libx264", "-preset", "fast", "-g", "1",
            "-r", "25", "-pix_fmt", "yuv420p",
            "-an",
            str(output_path)
        ], check=True, capture_output=True)
        log.append(f"SUCCESS: I-frame MP4 {video_path} -> {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.decode() if e.stderr else str(e)
        log.append(f"ERROR: Failed I-frame MP4 {video_path} -> {output_path}: {err_msg}")
        return False

def extract_video_frames(video_path, frames_dir, every_n=30, log=None):
    """
    Extracts frames from video every n frames.
    """
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = sanitize_filename(Path(video_path).stem)
    out_pattern = frames_dir / (safe_stem + "_frame_%05d.png")
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", str(video_path),
            "-vf", f"select=not(mod(n\\,{every_n}))",
            "-vsync", "vfr", str(out_pattern)
        ], check=True, capture_output=True)
        if log is not None:
            log.append(f"SUCCESS: Frames extracted from {video_path} to {frames_dir}")
        return True
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.decode() if e.stderr else str(e)
        if log is not None:
            log.append(f"ERROR: Frame extraction failed {video_path} -> {frames_dir}: {err_msg}")
        return False

def batch_video_processing(input_folder, output_folder, log_path=None, process_iframe=True, process_audio=True, process_frames=False, frame_interval=30, skip_existing=True):
    """
    Batch-process all media files in input_folder.
    For each, create:
      - I-frame only MP4 (if process_iframe, video files only)
      - audio WAV (if process_audio, from both video & audio files)
      - extracted frames (if process_frames, video files only)
    Logs all actions.
    """
    # EXTENSIONS for both audio and video
    video_exts = [".mp4", ".mov", ".m4v", ".avi", ".mkv"]
    audio_exts = [".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"]
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    log = []
    all_files = [f for f in input_folder.rglob("*") if f.is_file()]
    videos = [f for f in all_files if f.suffix.lower() in video_exts]
    audios = [f for f in all_files if f.suffix.lower() in audio_exts]
    log.append(f"Media processing started. Found {len(videos)} video files and {len(audios)} audio files.")

    # Audio extraction from ALL (audio + video)
    if process_audio:
        media_files = videos + audios
        for media in media_files:
            safe_stem = sanitize_filename(media.stem)
            wav_out = output_folder / (safe_stem + ".wav")
            if skip_existing and wav_out.exists() and wav_out.stat().st_size > 0:
                log.append(f"SKIP: WAV exists {wav_out}")
            else:
                extract_audio_from_media(media, wav_out, log)

    # I-frame and frames extraction only for videos
    for vid in videos:
        safe_stem = sanitize_filename(vid.stem)
        if process_iframe:
            iframe_out = output_folder / (safe_stem + "_iframe.mp4")
            if skip_existing and iframe_out.exists() and iframe_out.stat().st_size > 0:
                log.append(f"SKIP: I-frame exists {iframe_out}")
            else:
                convert_to_iframe_only_mp4(vid, iframe_out, log)
        if process_frames:
            frames_dir = output_folder / (safe_stem + "_frames")
            if skip_existing and frames_dir.exists() and any(frames_dir.iterdir()):
                log.append(f"SKIP: Frames exist {frames_dir}")
            else:
                extract_video_frames(vid, frames_dir, every_n=frame_interval, log=log)

    log.append("Batch media processing complete.")
    if log_path:
        with open(log_path, "w") as f:
            for line in log:
                f.write(line + "\n")
    else:
        for line in log:
            print(line)

    return log

if __name__ == "__main__":
    # Command-line usage:
    # python audio_prep.py <input_folder> <output_folder> [log_path] [--no-iframe] [--no-audio] [--frames] [--frame-interval=N] [--no-skip]
    import argparse

    parser = argparse.ArgumentParser(description="Batch audio/video processing for instructional pipeline.")
    parser.add_argument("input_folder", help="Folder containing source audio/video files")
    parser.add_argument("output_folder", help="Folder to save processed outputs")
    parser.add_argument("--log", dest="log_path", help="Path to save log file (default: print to console)", default=None)
    parser.add_argument("--no-iframe", action="store_true", help="Do not extract I-frame only MP4s (video files only)")
    parser.add_argument("--no-audio", action="store_true", help="Do not extract audio WAVs")
    parser.add_argument("--frames", action="store_true", help="Extract video frames (video files only)")
    parser.add_argument("--frame-interval", type=int, default=30, help="Frame extraction interval")
    parser.add_argument("--no-skip", action="store_true", help="Do not skip existing outputs")
    args = parser.parse_args()

    batch_video_processing(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        log_path=args.log_path,
        process_iframe=not args.no_iframe,
        process_audio=not args.no_audio,
        process_frames=args.frames,
        frame_interval=args.frame_interval,
        skip_existing=not args.no_skip
    )
