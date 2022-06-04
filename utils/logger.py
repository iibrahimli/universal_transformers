"""
Logger
"""

import sys
from loguru import logger


class Logger:
    def __init__(self, local_rank: int):
        self.local_rank = local_rank

        # configure logger
        logger.remove()
        log_fmt = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | {message}"
        logger.add(sys.stderr, format=log_fmt)
        
        self.loguru_logger = logger
    
    def log(self, *args, **kwargs):
        self.loguru_logger.info(*args, **kwargs)