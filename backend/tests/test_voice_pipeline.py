"""
Test script for the voice pipeline.
"""

import asyncio
import os
from datetime import datetime, timedelta
from ..services.chroma_service import ChromaService
from ..services.ha_service import HomeAssistantService
from ..services.command_processor import CommandProcessor
from ..voice.pipeline import VoicePipeline

async def test_voice_pipeline():
    """Test the voice pipeline functionality."""
    print("Initializing services...")
    
    # Initialize services
    chroma = ChromaService()
    ha = HomeAssistantService(
        base_url=os.getenv("HA_URL", "http://localhost:8123"),
        token=os.getenv("HA_TOKEN")
    )
    processor = CommandProcessor(chroma, ha)
    pipeline = VoicePipeline(chroma, ha, processor)
    
    print("\nSyncing entities with ChromaDB...")
    success = await pipeline.sync_entities()
    if not success:
        print("Failed to sync entities")
        return
        
    print("\nGetting rooms...")
    rooms = await pipeline.get_rooms()
    print(f"Found rooms: {', '.join(rooms)}")
    
    print("\nTesting command processing...")
    test_commands = [
        "turn on living room light",
        "what's the status of the kitchen light",
        "set the bedroom light to 50% brightness",
        "turn off the bathroom light"
    ]
    
    for command in test_commands:
        print(f"\nTesting command: {command}")
        success, response, entity_data = await pipeline.process_command(command)
        print(f"Success: {success}")
        print(f"Response: {response}")
        if entity_data:
            print(f"Entity: {entity_data['entity_id']}")
            print(f"State: {entity_data['state']}")
            
    print("\nTesting entity history...")
    if entity_data:
        history = await pipeline.get_entity_history(
            entity_data["entity_id"],
            datetime.now() - timedelta(hours=1)
        )
        print(f"Found {len(history)} state changes in the last hour")
        
    print("\nTest completed!")

if __name__ == "__main__":
    asyncio.run(test_voice_pipeline()) 