import logging
import datetime
from pathlib import Path
from course_compiler.config import INPUT_ROOT, OUTPUT_ROOT
from course_compiler.audio_prep import batch_video_processing
from course_compiler.file_utils import ensure_dirs

# Ensure output directory exists for logs and audio
Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
logfile = Path(OUTPUT_ROOT) / f"audio_prep_{timestamp}.log"
logfile = Path(str(logfile).replace(" ", "_"))

logging.basicConfig(
    filename=logfile,
    filemode="a",
    format="%(asctime)s %(levelname)s:%(message)s",
    level=logging.INFO
)

def main():
    audio_input_folder = INPUT_ROOT
    wav_output_folder = f"{OUTPUT_ROOT}/audio_prepped"
    ensure_dirs(OUTPUT_ROOT, ["audio_prepped"])
    log_file = Path(OUTPUT_ROOT) / f"audio_prep_batch_{timestamp}.log"
    log_file = Path(str(log_file).replace(" ", "_"))

    print(f"Preparing audio files from {audio_input_folder} to {wav_output_folder}")
    batch_video_processing(
        audio_input_folder,
        wav_output_folder,
        log_path=log_file,
        process_audio=True,
        process_iframe=False,    # Do NOT extract I-frames
        process_frames=False,    # Do NOT extract video frames
        skip_existing=True
    )
    print(f"Audio prep done. See log for details: {log_file}")

if __name__ == "__main__":
    main()
