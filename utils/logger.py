import logging
import sys

# ANSI Escape Sequences for Colors
RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"
WHITE = "\033[37m"

class ColoredFormatter(logging.Formatter):
    """Custom Formatter to add colors to log levels."""
    
    COLORS = {
        logging.DEBUG: CYAN,
        logging.INFO: GREEN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: RED + "\033[1m" # Bold Red
    }

    def format(self, record):
        log_color = self.COLORS.get(record.levelno, WHITE)
        feature = f"[{record.name}]"
        
        # Format: [TIME] [LEVEL] [LOGGER_NAME] Message
        # We override standard formatting to make it pretty
        timestamp = self.formatTime(record, self.datefmt)
        level_name = record.levelname
        message = record.getMessage()
        
        formatted_msg = (
            f"{WHITE}[{timestamp}]{RESET} "
            f"{log_color}{level_name:<8}{RESET} "
            f"{BLUE}{feature:<15}{RESET} : "
            f"{message}"
        )
        return formatted_msg

def setup_logger(name="HeatMapApp", level=logging.INFO):
    """
    Sets up a logger with the specified name and level.
    Uses the ColoredFormatter for console output.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Check if handlers already exist to avoid duplicate logs
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Create formatter and add it to the handler
        formatter = ColoredFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        
    return logger
