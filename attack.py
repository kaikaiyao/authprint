#!/usr/bin/env python
"""
Entry point for attacking the StyleGAN fingerprinting model.
"""
import os
import sys

# Import the main function directly from scripts
from scripts.attack import main

if __name__ == "__main__":
    # Execute the main function
    main() 