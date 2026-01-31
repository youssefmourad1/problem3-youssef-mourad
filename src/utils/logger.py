import logging
import sys

def setup_logger(name="ProductionLinePSO", log_file="pso_run.log"):
    """
    Sets up a logger with console and file handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    
    # Add handler
    if not logger.handlers:
        logger.addHandler(ch)
        
    return logger
