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

        # Create directories if they don't exist
        for directory in [self.ASSET_ROOT, self.INPUT_ROOT, self.OUTPUT_ROOT]:
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
