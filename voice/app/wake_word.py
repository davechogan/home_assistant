"""
Wake word detection module using Porcupine for keyword spotting.
"""

import asyncio
import logging
import queue
import sounddevice as sd
import numpy as np
import os
import platform
from typing import Optional, Tuple, List

from shared.config.config import config

logger = logging.getLogger(__name__)

# Try to import Porcupine, but handle the case where it's not supported
try:
    import pvporcupine
    PORCUPINE_AVAILABLE = True
except NotImplementedError as e:
    logger.warning(f"Porcupine not available for this CPU architecture: {e}")
    PORCUPINE_AVAILABLE = False
except ImportError as e:
    logger.warning(f"Failed to import Porcupine: {e}")
    PORCUPINE_AVAILABLE = False

class WakeWordDetector:
    """Wake word detection using Porcupine for keyword spotting."""
    
    def __init__(self, wake_word: str, sensitivity: float = 0.5, device_id: Optional[int] = None, test_mode: bool = False):
        """
        Initialize the wake word detector.
        
        Args:
            wake_word: The wake word to detect
            sensitivity: Detection sensitivity (0.0 to 1.0)
            device_id: Audio input device ID (None for default)
            test_mode: If True, simulate wake word detection for testing
        """
        self.wake_word = wake_word.lower()
        self.sensitivity = sensitivity
        self.device_id = device_id or config.AUDIO_INPUT_DEVICE
        self.test_mode = test_mode or not PORCUPINE_AVAILABLE
        self.porcupine = None
        self.audio_queue = queue.Queue()
        self.is_running = False
        self.sample_rate = config.AUDIO_SAMPLE_RATE
        self.channels = config.AUDIO_CHANNELS
        self.frame_length = config.AUDIO_FRAME_LENGTH
        
    async def initialize(self):
        """Initialize the Porcupine wake word detector."""
        try:
            if self.test_mode:
                logger.info("Running in test mode - wake word detection simulated")
                return
                
            # Check for access key
            access_key = os.getenv('PORCUPINE_ACCESS_KEY')
            if not access_key:
                raise ValueError("PORCUPINE_ACCESS_KEY environment variable not set")
            
            # List available audio devices with detailed information
            devices = sd.query_devices()
            logger.info("Available audio devices:")
            for i, device in enumerate(devices):
                logger.info(f"Device {i}: {device['name']}")
                logger.info(f"  Input channels: {device.get('max_input_channels', 0)}")
                logger.info(f"  Output channels: {device.get('max_output_channels', 0)}")
                logger.info(f"  Default sample rate: {device.get('default_samplerate', 'N/A')}")
                logger.info(f"  Host API: {device.get('hostapi', 'N/A')}")
            
            # Get default input device
            default_input = sd.query_devices(kind='input')
            logger.info(f"Default input device: {default_input['name']}")
            
            # Log selected device
            if self.device_id is not None:
                selected_device = sd.query_devices(self.device_id)
                logger.info(f"Selected input device: {selected_device['name']}")
            else:
                logger.info("Using default input device")
            
            # Log system information for debugging
            logger.info(f"System: {platform.system()}")
            logger.info(f"Machine: {platform.machine()}")
            logger.info(f"Processor: {platform.processor()}")
            
            if not PORCUPINE_AVAILABLE:
                logger.warning("Porcupine is not available for this CPU architecture")
                self.test_mode = True
                return
            
            # Initialize Porcupine
            self.porcupine = pvporcupine.create(
                access_key=access_key,
                keywords=[self.wake_word],
                sensitivities=[self.sensitivity]
            )
            logger.info(f"Initialized wake word detector for '{self.wake_word}'")
            
        except Exception as e:
            logger.error(f"Failed to initialize wake word detector: {e}")
            raise
    
    def audio_callback(self, indata, frames, time, status):
        """Callback for audio input stream."""
        if status:
            logger.warning(f"Audio callback status: {status}")
        self.audio_queue.put(indata.copy())
    
    async def detect(self) -> bool:
        """
        Detect the wake word in the audio stream.
        
        Returns:
            bool: True if wake word detected, False otherwise
        """
        if self.test_mode:
            # Simulate wake word detection every 5 seconds in test mode
            await asyncio.sleep(5)
            logger.info("Test mode: Simulating wake word detection")
            return True
            
        if not self.is_running:
            try:
                self.stream = sd.InputStream(
                    device=self.device_id,
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    blocksize=self.frame_length,
                    callback=self.audio_callback
                )
                self.stream.start()
                self.is_running = True
                logger.info("Started audio input stream")
            except Exception as e:
                logger.error(f"Failed to start audio stream: {e}")
                raise
        
        try:
            # Get audio data from queue
            audio_data = self.audio_queue.get(timeout=1.0)
            
            # Convert to mono if stereo
            if audio_data.shape[1] > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Convert to 16-bit PCM
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # Process with Porcupine
            keyword_index = self.porcupine.process(audio_int16)
            if keyword_index >= 0:
                logger.info(f"Wake word detected with confidence: {self.sensitivity}")
                return True
            
            return False
            
        except queue.Empty:
            return False
        except Exception as e:
            logger.error(f"Error in wake word detection: {e}")
            return False
    
    async def get_audio_input(self) -> np.ndarray:
        """
        Get audio input after wake word detection.
        
        Returns:
            np.ndarray: Audio data as numpy array
        """
        if self.test_mode:
            # Return empty array in test mode
            return np.array([])
            
        audio_chunks = []
        silence_threshold = 0.01
        silence_duration = 0.5  # seconds
        silence_samples = int(silence_duration * self.sample_rate)
        silence_count = 0
        
        while True:
            try:
                audio_data = self.audio_queue.get(timeout=0.1)
                audio_chunks.append(audio_data)
                
                # Check for silence
                if np.max(np.abs(audio_data)) < silence_threshold:
                    silence_count += len(audio_data)
                    if silence_count >= silence_samples:
                        break
                else:
                    silence_count = 0
                    
            except queue.Empty:
                break
        
        if not audio_chunks:
            return np.array([])
        
        return np.concatenate(audio_chunks)
    
    async def cleanup(self):
        """Clean up resources."""
        self.is_running = False
        if hasattr(self, 'stream'):
            try:
                self.stream.stop()
                self.stream.close()
                logger.info("Closed audio input stream")
            except Exception as e:
                logger.error(f"Error closing audio stream: {e}")
        if self.porcupine:
            try:
                self.porcupine.delete()
                logger.info("Cleaned up Porcupine resources")
            except Exception as e:
                logger.error(f"Error cleaning up Porcupine: {e}")
        self.audio_queue.queue.clear() 