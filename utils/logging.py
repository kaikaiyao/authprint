import logging

# Custom logging filter that only allows messages from rank 0
class LogRankFilter(logging.Filter):
    def __init__(self, rank):
        super().__init__()
        self.rank = rank
        
    def filter(self, record):
        # Only allow log messages from rank 0
        return self.rank == 0

# Set up logging
def setup_logging(log_file, rank=0):
    """
    Set up logging configuration with strict rank filtering.
    Only messages from rank 0 will be logged.
    
    Args:
        log_file: Path to the log file
        rank: Process rank (0 for main process)
    """
    # Clear existing loggers first
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Only rank 0 should log anything
    if rank != 0:
        # Set up a null handler that discards all messages for non-zero ranks
        null_handler = logging.NullHandler()
        logger.addHandler(null_handler)
        return
    
    # For rank 0, set up normal logging
    # Create a file handler for writing logs to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)

    # Create a stream handler for printing logs to the terminal
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    stream_handler.setFormatter(stream_formatter)

    # Add both handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)