import os
import random
import logging
from utils.logging import LogRankFilter

def apply_rank_filter(rank=0):
    """Apply rank filtering to ensure only rank 0 logs messages"""
    root_logger = logging.getLogger()
    
    # First remove existing rank filters if any
    for handler in root_logger.handlers:
        for filter in handler.filters[:]:
            if isinstance(filter, LogRankFilter):
                handler.removeFilter(filter)
        
        # Add our rank filter to each handler
        handler.addFilter(LogRankFilter(rank))

def flip_key(input_key: bytes, flip_key_type: str, rank=0) -> bytes:
    """Flips bits in a 256-bit (32-byte) key according to the chosen flip_key_type.
    
    flip_key_type options:
      "1"      - Flip exactly one random bit.
      "10"     - Flip exactly 10 unique random bits.
      "random" - Flip every bit with a random mask, resulting in a completely random key.
    """
    # Apply rank filtering to ensure only rank 0 logs messages
    apply_rank_filter(rank)
    
    if len(input_key) != 32:
        raise ValueError("Input key must be exactly 32 bytes long")

    # Create a mutable copy of the key
    mutable_key = bytearray(input_key)
    
    if flip_key_type == "1":
        # Choose one random bit position (0 to 255)
        bit_to_flip = random.randint(0, 255)
        logging.info(f"Flipping bit #{bit_to_flip}")
        byte_index = bit_to_flip // 8
        bit_index = bit_to_flip % 8
        mutable_key[byte_index] ^= (1 << (7 - bit_index))  # Big-endian bit order
        return bytes(mutable_key)
    
    elif flip_key_type == "10":
        # Choose 10 unique random bit positions (0 to 255)
        bit_positions = random.sample(range(256), 10)
        logging.info(f"Flipping bits at positions: {bit_positions}")
        for bit in bit_positions:
            byte_index = bit // 8
            bit_index = bit % 8
            mutable_key[byte_index] ^= (1 << (7 - bit_index))
        return bytes(mutable_key)
    
    elif flip_key_type == "random":
        # Generate a random 256-bit mask and XOR it with the input key.
        random_mask = os.urandom(32)
        logging.info("Flipping key to a completely random key using XOR with a random mask")
        new_key = bytes(b ^ m for b, m in zip(mutable_key, random_mask))
        return new_key
    
    else:
        raise ValueError("Invalid flip_key_type. Choose '1', '10', or 'random'.")
