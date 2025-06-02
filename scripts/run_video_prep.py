import logging
import datetime
from pathlib import Path
from course_compiler.config import INPUT_ROOT, OUTPUT_ROOT
from course_compiler.video_prep import batch_video_processing
from course_compiler.file_utils import ensure_dirs

Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
logfile = Path(OUTPUT_ROOT) / f"video_prep_{timestamp}.log"
logfile = Path(str(logfile).replace(" ", "_"))

logging.basicConfig(
    filename=logfile,
    filemode="a",
    format="%(asctime)s %(levelname)s:%(message)s",
    level=logging.INFO
)

def main():
    video_input_folder = INPUT_ROOT
    video_output_folder = f"{OUTPUT_ROOT}/video_prepped"
    ensure_dirs(OUTPUT_ROOT, ["video_prepped"])
    log_file = Path(OUTPUT_ROOT) / f"video_prep_batch_{timestamp}.log"
    log_file = Path(str(log_file).replace(" ", "_"))

    print(f"Preparing video files from {video_input_folder} to {video_output_folder}")
    batch_video_processing(
        video_input_folder,
        video_output_folder,
        log_path=log_file,
        process_iframe=True,
        process_audio=False,  # Only do audio in audio prep!
        process_frames=False,      # Set to True to enable frame extraction
        frame_interval=30,         # Frames extracted every 30 frames if enabled
        skip_existing=True
    )
    print(f"Video prep done. Outputs in {video_output_folder}")

if __name__ == "__main__":
    main()
# This script prepares video files from a specified input folder, processing them according to the
# specified parameters and saving the results to an output folder. It logs the process to a timestamped
# log file. The script uses the `batch_video_processing` function from the `video_prep` module.