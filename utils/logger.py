import logging
import os

def setup_logger(name, log_file, level=logging.INFO):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Avoid duplicate handlers
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        logger.handlers.clear()
        
    handler = logging.FileHandler(log_file, mode='a')        
    handler.setFormatter(formatter)
    
    console = logging.StreamHandler()
    console.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(console)
    # Prevent propagation to avoid double logging
    logger.propagate = False
    
    return logger
