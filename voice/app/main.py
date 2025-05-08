"""
Main entry point for the voice pipeline.
Handles wake word detection, speech-to-text, and text-to-speech processing.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional
import os
import whisper  # OpenAI Whisper for STT
import numpy as np
import sounddevice as sd

from shared.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO if not config.DEBUG else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VoicePipeline:
    """Main voice pipeline class that orchestrates the voice processing components."""
    
    def __init__(self, test_mode: bool = False):
        """
        Initialize the voice pipeline components.
        
        Args:
            test_mode: If True, run in test mode with simulated wake word detection
        """
        self.is_listening = False
        self.test_mode = test_mode
        self.wake_word_detector = None
        
    async def initialize(self):
        """Initialize all voice pipeline components."""
        try:
            # Check for Porcupine access key
            access_key = os.getenv('PORCUPINE_ACCESS_KEY')
            if not access_key:
                logger.error("PORCUPINE_ACCESS_KEY environment variable not set")
                logger.info("Please set PORCUPINE_ACCESS_KEY in your .env file")
                logger.info("You can get a free key from https://console.picovoice.ai/")
                raise ValueError("PORCUPINE_ACCESS_KEY not set")

            logger.info("Starting voice pipeline initialization...")
            logger.info(f"Wake word: {config.WAKE_WORD}")
            logger.info(f"Wake word sensitivity: {config.WAKE_WORD_SENSITIVITY}")
            logger.info(f"Audio input device: {config.AUDIO_INPUT_DEVICE or 'default'}")
            logger.info(f"Test mode: {self.test_mode}")
            
            # Initialize wake word detector
            from voice.app.wake_word import WakeWordDetector
            self.wake_word_detector = WakeWordDetector(
                wake_word=config.WAKE_WORD,
                sensitivity=config.WAKE_WORD_SENSITIVITY,
                device_id=config.AUDIO_INPUT_DEVICE,
                test_mode=self.test_mode
            )
            await self.wake_word_detector.initialize()
            logger.info("Voice pipeline initialized successfully")
            logger.info("Listening for wake word... (Press Ctrl+C to exit)")
            
        except Exception as e:
            logger.error(f"Failed to initialize voice pipeline: {e}")
            raise
    
    async def start(self):
        """Start the voice pipeline."""
        if not self.wake_word_detector:
            raise RuntimeError("Wake word detector not initialized")
        
        self.is_listening = True
        logger.info("Starting voice pipeline")
        
        # Load Whisper model once for efficiency
        stt_model = whisper.load_model("base")
        logger.info("Loaded Whisper STT model: base")
        
        try:
            while self.is_listening:
                # Wait for wake word
                if await self.wake_word_detector.detect():
                    logger.info("Wake word detected")
                    # --- STT Integration ---
                    logger.info("Listening for command...")
                    
                    # Record audio directly using sounddevice
                    LISTEN_TIMEOUT = 5  # seconds
                    rec = sd.rec(
                        int(LISTEN_TIMEOUT * 16000),
                        samplerate=16000,
                        channels=1,
                        dtype='float32'
                    )
                    sd.wait()
                    
                    if rec.size == 0:
                        logger.warning("No audio data captured after wake word.")
                        continue
                    
                    # Convert stereo to mono if needed
                    if len(rec.shape) > 1:
                        rec = np.mean(rec, axis=1)
                    
                    # Normalize audio
                    if rec.max() > 1.0 or rec.min() < -1.0:
                        rec = np.clip(rec, -1.0, 1.0)
                    
                    # Transcribe
                    result = stt_model.transcribe(rec, language='en')
                    transcript = result.get('text', '').strip()
                    logger.info(f"Transcribed text: {transcript}")
                    # TODO: Pass transcript to LLM/command processor
                    
        except Exception as e:
            logger.error(f"Error in voice pipeline: {e}")
            raise
        finally:
            self.is_listening = False
    
    async def stop(self):
        """Stop the voice pipeline."""
        self.is_listening = False
        logger.info("Stopping voice pipeline")
        
        # Clean up resources
        if self.wake_word_detector:
            await self.wake_word_detector.cleanup()

async def main():
    """Main entry point."""
    # Check if running in test mode
    test_mode = os.getenv('VOICE_TEST_MODE', '').lower() == 'true'
    
    pipeline = VoicePipeline(test_mode=test_mode)
    try:
        await pipeline.initialize()
        await pipeline.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    finally:
        await pipeline.stop()

if __name__ == "__main__":
    asyncio.run(main())
