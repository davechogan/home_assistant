"""
Voice command processing pipeline.
"""

import asyncio
from typing import Optional, Dict, Tuple, List
from ..services.chroma_service import ChromaService
from ..services.ha_service import HomeAssistantService
from ..services.command_processor import CommandProcessor
import json
import os
from datetime import datetime

class VoicePipeline:
    """Voice command processing pipeline."""
    
    def __init__(self,
                 chroma_service: ChromaService,
                 ha_service: HomeAssistantService,
                 command_processor: CommandProcessor):
        """
        Initialize the voice pipeline.
        
        Args:
            chroma_service: ChromaDB service instance
            ha_service: Home Assistant service instance
            command_processor: Command processor instance
        """
        self.chroma = chroma_service
        self.ha = ha_service
        self.command_processor = command_processor
        
    async def process_voice_command(self,
                                  audio_data: bytes,
                                  room: Optional[str] = None) -> Tuple[bool, str, Optional[Dict]]:
        """
        Process a voice command from audio data.
        
        Args:
            audio_data: Raw audio data
            room: Optional room context
            
        Returns:
            Tuple of (success, response message, entity data)
        """
        try:
            # TODO: Implement STT to convert audio to text
            # For now, we'll use a placeholder
            command_text = "turn on living room light"
            
            # Process the command
            success, response, entity_data = await self.command_processor.process_command(
                command_text,
                room
            )
            
            # TODO: Implement TTS to convert response to speech
            # For now, we'll just return the text response
            
            return success, response, entity_data
            
        except Exception as e:
            print(f"Error processing voice command: {e}")
            return False, "Sorry, I encountered an error processing your command.", None
            
    async def sync_entities(self) -> bool:
        """
        Sync Home Assistant entities with ChromaDB.
        
        Returns:
            bool: True if successful
        """
        try:
            # Get all entities from Home Assistant
            entities = await self.ha.get_entities()
            
            # Sync with ChromaDB
            return self.chroma.sync_ha_entities(entities)
            
        except Exception as e:
            print(f"Error syncing entities: {e}")
            return False
            
    async def get_rooms(self) -> List[str]:
        """
        Get list of all rooms/zones.
        
        Returns:
            List of room names
        """
        return await self.ha.get_rooms()
        
    async def get_entity_state(self, entity_id: str) -> Optional[Dict]:
        """
        Get the current state of an entity.
        
        Args:
            entity_id: The entity ID to query
            
        Returns:
            Entity state dictionary if found, None otherwise
        """
        return await self.ha.get_entity_state(entity_id)
        
    async def get_entity_history(self,
                               entity_id: str,
                               start_time: datetime,
                               end_time: Optional[datetime] = None) -> List[Dict]:
        """
        Get entity state history.
        
        Args:
            entity_id: The entity ID to query
            start_time: Start time for history
            end_time: End time for history (defaults to now)
            
        Returns:
            List of state changes
        """
        return await self.ha.get_entity_history(
            entity_id,
            start_time,
            end_time
        ) 