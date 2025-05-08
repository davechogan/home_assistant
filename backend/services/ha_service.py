"""
Home Assistant API service for entity control and state management.
"""

import aiohttp
import asyncio
from typing import Dict, List, Optional, Any
import json
import os
from datetime import datetime

class HomeAssistantService:
    """Service for interacting with Home Assistant API."""
    
    def __init__(self, 
                 base_url: str = "http://localhost:8123",
                 token: Optional[str] = None):
        """
        Initialize the Home Assistant service.
        
        Args:
            base_url: Home Assistant instance URL
            token: Long-lived access token
        """
        self.base_url = base_url.rstrip('/')
        self.token = token or os.getenv('HA_TOKEN')
        if not self.token:
            raise ValueError("Home Assistant token is required")
            
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        
    async def get_entities(self) -> List[Dict]:
        """
        Get all entities from Home Assistant.
        
        Returns:
            List of entity dictionaries
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/api/states",
                headers=self.headers
            ) as response:
                if response.status == 200:
                    states = await response.json()
                    return [
                        {
                            "entity_id": state["entity_id"],
                            "name": state["attributes"].get("friendly_name", state["entity_id"]),
                            "type": state["entity_id"].split(".")[0],
                            "state": state["state"],
                            "attributes": state["attributes"],
                            "room": state["attributes"].get("room", None),
                            "description": state["attributes"].get("description", None)
                        }
                        for state in states
                    ]
                return []
                
    async def call_service(self,
                          domain: str,
                          service: str,
                          entity_id: Optional[str] = None,
                          data: Optional[Dict] = None) -> bool:
        """
        Call a Home Assistant service.
        
        Args:
            domain: Service domain (e.g., 'light', 'switch')
            service: Service name (e.g., 'turn_on', 'turn_off')
            entity_id: Target entity ID
            data: Additional service data
            
        Returns:
            bool: True if successful
        """
        service_data = data or {}
        if entity_id:
            service_data["entity_id"] = entity_id
            
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/services/{domain}/{service}",
                headers=self.headers,
                json=service_data
            ) as response:
                return response.status == 200
                
    async def get_entity_state(self, entity_id: str) -> Optional[Dict]:
        """
        Get the current state of an entity.
        
        Args:
            entity_id: The entity ID to query
            
        Returns:
            Entity state dictionary if found, None otherwise
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/api/states/{entity_id}",
                headers=self.headers
            ) as response:
                if response.status == 200:
                    return await response.json()
                return None
                
    async def get_rooms(self) -> List[str]:
        """
        Get list of all rooms/zones.
        
        Returns:
            List of room names
        """
        entities = await self.get_entities()
        rooms = set()
        for entity in entities:
            if entity.get("room"):
                rooms.add(entity["room"])
        return sorted(list(rooms))
        
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
        if not end_time:
            end_time = datetime.now()
            
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/api/history/period/{start_time.isoformat()}",
                params={
                    "filter_entity_id": entity_id,
                    "end_time": end_time.isoformat()
                },
                headers=self.headers
            ) as response:
                if response.status == 200:
                    return await response.json()
                return [] 