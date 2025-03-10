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
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers if any
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create a rank filter
    rank_filter = LogRankFilter(rank)

    # Create a file handler for writing logs to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    file_handler.addFilter(rank_filter)  # Add rank filter to file handler

    # Create a stream handler for printing logs to the terminal
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    stream_handler.setFormatter(stream_formatter)
    stream_handler.addFilter(rank_filter)  # Add rank filter to stream handler

    # Add both handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)