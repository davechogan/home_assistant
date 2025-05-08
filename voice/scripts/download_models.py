"""
Script to download required models for the voice pipeline.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point."""
    try:
        # Create models directory
        models_dir = Path(__file__).parent.parent / 'models'
        models_dir.mkdir(exist_ok=True)
        logger.info("Models directory created/verified")
        
    except Exception as e:
        logger.error(f"Failed to set up models directory: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 