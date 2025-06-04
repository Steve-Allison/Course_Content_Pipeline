# course_compiler/config.py
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class Config:
    def __init__(self):
        # Default paths can be overridden by environment variables
        self.ASSET_ROOT = os.getenv('ASSET_ROOT', os.path.join(os.path.expanduser('~'), 'course_assets'))
        self.INPUT_ROOT = os.getenv('INPUT_ROOT', os.path.join(self.ASSET_ROOT, 'original_input'))
        self.OUTPUT_ROOT = os.getenv('OUTPUT_ROOT', os.path.join(self.ASSET_ROOT, 'output'))

        # Organized output subdirectories
        self.LOGS_DIR = os.path.join(self.OUTPUT_ROOT, 'logs')
        self.PROCESSED_DIR = os.path.join(self.OUTPUT_ROOT, 'processed')
        self.SUMMARY_DIR = os.path.join(self.OUTPUT_ROOT, 'summary')
        self.TEMP_DIR = os.path.join(self.OUTPUT_ROOT, 'temp')

        # Processed content subdirectories
        self.INSTRUCTIONAL_JSON_DIR = os.path.join(self.PROCESSED_DIR, 'instructional_json')
        self.CAPTION_PREPPED_DIR = os.path.join(self.PROCESSED_DIR, 'caption_prepped')
        self.AUDIO_PREPPED_DIR = os.path.join(self.PROCESSED_DIR, 'audio_prepped')
        self.VIDEO_PREPPED_DIR = os.path.join(self.PROCESSED_DIR, 'video_prepped')
        self.MEDIA_METADATA_DIR = os.path.join(self.PROCESSED_DIR, 'media_metadata')
        self.IMAGES_DIR = os.path.join(self.INSTRUCTIONAL_JSON_DIR, 'images')

        # Summary subdirectories (AI-ready files)
        self.GLOSSARY_DIR = os.path.join(self.SUMMARY_DIR, 'glossary')
        self.ANALYSIS_DIR = os.path.join(self.SUMMARY_DIR, 'analysis')
        self.PROMPTS_DIR = os.path.join(self.SUMMARY_DIR, 'prompts')

        # Create all directories if they don't exist
        for directory in [
            self.ASSET_ROOT, self.INPUT_ROOT, self.OUTPUT_ROOT,
            self.LOGS_DIR, self.PROCESSED_DIR, self.SUMMARY_DIR, self.TEMP_DIR,
            self.INSTRUCTIONAL_JSON_DIR, self.CAPTION_PREPPED_DIR,
            self.AUDIO_PREPPED_DIR, self.VIDEO_PREPPED_DIR, self.MEDIA_METADATA_DIR,
            self.IMAGES_DIR, self.GLOSSARY_DIR, self.ANALYSIS_DIR, self.PROMPTS_DIR
        ]:
            os.makedirs(directory, exist_ok=True)

        # Logging configuration
        self.LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
        self.LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

        # Processing settings
        self.MAX_WORKERS = int(os.getenv('MAX_WORKERS', str(os.cpu_count() or 4)))

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as a dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

# Global configuration instance
config = Config()

# For backward compatibility
ASSET_ROOT = config.ASSET_ROOT
INPUT_ROOT = config.INPUT_ROOT
OUTPUT_ROOT = config.OUTPUT_ROOT

# New organized paths
LOGS_DIR = config.LOGS_DIR
PROCESSED_DIR = config.PROCESSED_DIR
SUMMARY_DIR = config.SUMMARY_DIR
TEMP_DIR = config.TEMP_DIR
INSTRUCTIONAL_JSON_DIR = config.INSTRUCTIONAL_JSON_DIR
CAPTION_PREPPED_DIR = config.CAPTION_PREPPED_DIR
AUDIO_PREPPED_DIR = config.AUDIO_PREPPED_DIR
VIDEO_PREPPED_DIR = config.VIDEO_PREPPED_DIR
MEDIA_METADATA_DIR = config.MEDIA_METADATA_DIR
IMAGES_DIR = config.IMAGES_DIR
GLOSSARY_DIR = config.GLOSSARY_DIR
ANALYSIS_DIR = config.ANALYSIS_DIR
PROMPTS_DIR = config.PROMPTS_DIR
