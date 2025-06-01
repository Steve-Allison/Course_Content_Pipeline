import subprocess
from pathlib import Path

def extract_audio_from_video(video_path, wav_path, log):
    """
    Extracts audio from video file and converts to mono 16kHz WAV.
    """
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn",
            "-ar", "16000",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-f", "wav", str(wav_path)
        ], check=True, capture_output=True)
        log.append(f"SUCCESS: Audio extracted {video_path} -> {wav_path}")
        return True
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.decode() if e.stderr else str(e)
        log.append(f"ERROR: Failed audio extraction {video_path} -> {wav_path}: {err_msg}")
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
    out_pattern = frames_dir / (Path(video_path).stem.replace(" ", "_") + "_frame_%05d.png")
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
    Batch-process all video files in input_folder.
    For each, create:
      - I-frame only MP4 (if process_iframe)
      - audio WAV (if process_audio)
      - extracted frames (if process_frames)
    Logs all actions.
    """
    supported_exts = [".mp4", ".mov", ".m4v"]
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    log = []
    videos = [f for f in input_folder.rglob("*") if f.suffix.lower() in supported_exts]
    log.append(f"Video processing started. Found {len(videos)} video files.")

    for vid in videos:
        stem = vid.stem.replace(" ", "_")
        if process_iframe:
            iframe_out = output_folder / (stem + "_iframe.mp4")
            if skip_existing and iframe_out.exists() and iframe_out.stat().st_size > 0:
                log.append(f"SKIP: I-frame exists {iframe_out}")
            else:
                convert_to_iframe_only_mp4(vid, iframe_out, log)
        if process_audio:
            wav_out = output_folder / (stem + ".wav")
            if skip_existing and wav_out.exists() and wav_out.stat().st_size > 0:
                log.append(f"SKIP: WAV exists {wav_out}")
            else:
                extract_audio_from_video(vid, wav_out, log)
        if process_frames:
            frames_dir = output_folder / (stem + "_frames")
            if skip_existing and frames_dir.exists() and any(frames_dir.iterdir()):
                log.append(f"SKIP: Frames exist {frames_dir}")
            else:
                extract_video_frames(vid, frames_dir, every_n=frame_interval, log=log)

    log.append("Batch video processing complete.")
    if log_path:
        with open(log_path, "w") as f:
            for line in log:
                f.write(line + "\n")
    else:
        for line in log:
            print(line)

    return log



# This script prepares video files from a specified input folder, processing them according to the
# specified parameters and saving the results to an output folder. It logs the process to a timestamped
# log file. The script uses the `batch_video_processing` function from the `video_prep` module.
# It ensures the output directory exists, sets up logging, and processes video files with options
# for iframe extraction, audio processing, and frame extraction.
# It also handles existing files based on the `skip_existing` parameter.