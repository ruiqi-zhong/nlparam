import logging
import time

# Define the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set the logging level to DEBUG to capture all messages

# Define the format string
format_str = (
    "%(asctime)s - %(filename)s - Line %(lineno)d - %(levelname)s - %(message)s"
)
date_format = "%Y-%m-%d %H:%M:%S"  # Define a date format

# Create a console handler and set its level to DEBUG
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter(format_str, datefmt=date_format))

# Create a file handler and set its level to DEBUG
f_name = f"logs/{time.strftime('%Y-%m-%d-%H-%M-%S')}.log"
file_handler = logging.FileHandler(f_name)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(format_str, datefmt=date_format))

# Add the handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)
