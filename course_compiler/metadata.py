import subprocess
import json
from pathlib import Path

def ffprobe_metadata(file_path):
    """
    Uses ffprobe to extract media metadata as a dict.
    Works for audio and video.
    """
    command = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(file_path)
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        return {"error": f"ffprobe failed: {e.stderr if e.stderr else e}"}

def get_audio_metadata(wav_path):
    """
    Extracts audio-specific metadata using ffprobe.
    Returns a dict with duration, sample_rate, channels, etc.
    """
    meta = ffprobe_metadata(wav_path)
    audio_streams = [s for s in meta.get("streams", []) if s.get("codec_type") == "audio"]
    if not audio_streams:
        return {"error": "No audio stream found"}
    stream = audio_streams[0]
    return {
        "duration_sec": float(stream.get("duration", meta.get("format", {}).get("duration", 0))),
        "sample_rate": int(stream.get("sample_rate", 0)),
        "channels": int(stream.get("channels", 0)),
        "bit_rate": int(stream.get("bit_rate", 0)),
        "format": meta.get("format", {}).get("format_long_name", ""),
        "size_bytes": int(meta.get("format", {}).get("size", 0)),
    }

def get_video_metadata(video_path):
    """
    Extracts video-specific metadata using ffprobe.
    Returns a dict with duration, frame_rate, resolution, etc.
    """
    meta = ffprobe_metadata(video_path)
    video_streams = [s for s in meta.get("streams", []) if s.get("codec_type") == "video"]
    if not video_streams:
        return {"error": "No video stream found"}
    stream = video_streams[0]
    # Parse frame rate (can be like "30000/1001")
    try:
        r = stream.get("r_frame_rate", "0/1")
        frame_rate = float(r.split('/')[0]) / float(r.split('/')[1]) if "/" in r else float(r)
    except Exception:
        frame_rate = 0.0
    return {
        "duration_sec": float(stream.get("duration", meta.get("format", {}).get("duration", 0))),
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "frame_rate": frame_rate,
        "codec": stream.get("codec_long_name", ""),
        "bit_rate": int(stream.get("bit_rate", 0)),
        "format": meta.get("format", {}).get("format_long_name", ""),
        "size_bytes": int(meta.get("format", {}).get("size", 0)),
    }
