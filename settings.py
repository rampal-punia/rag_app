from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler

from decouple import config

# Log file path
cwd = Path.cwd()
log_directory = cwd / 'logs'
if not log_directory.exists():
    log_directory.mkdir(exist_ok=True)
log_filename = 'application.log'
log_filepath = str(log_directory / log_filename)

# Configure the root logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# File handler that logs even debug messages
file_handler = RotatingFileHandler(
    log_filepath, maxBytes=5*1024*1024, backupCount=5)
file_handler.setLevel(logging.DEBUG)

# Formatter for the handler
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Root logger added to the file handler
logger = logging.getLogger()
logger.addHandler(file_handler)

HUGGINGFACEHUB_API_TOKEN = config('HUGGINGFACEHUB_API_TOKEN')

logging.basicConfig(level=logging.info)
logger = logging.getLogger()

CHUNK_SIZE = 350
CHUNK_OVERLAP = 80


# Use the environment variable if set, otherwise default to localhost
REDIS_URL = config("REDIS_URL", "redis://localhost:6379")
print(f"Connecting to Redis at: {REDIS_URL}")
