"""
Command processor service for handling voice commands and executing them through Home Assistant.
"""

from typing import Dict, List, Optional, Tuple
from .chroma_service import ChromaService
from .ha_service import HomeAssistantService
import re
from datetime import datetime

class CommandProcessor:
    """Service for processing and executing voice commands."""
    
    def __init__(self,
                 chroma_service: ChromaService,
                 ha_service: HomeAssistantService):
        """
        Initialize the command processor.
        
        Args:
            chroma_service: ChromaDB service instance
            ha_service: Home Assistant service instance
        """
        self.chroma = chroma_service
        self.ha = ha_service
        
        # Common command patterns
        self.command_patterns = {
            "turn_on": r"turn on|switch on|activate|enable",
            "turn_off": r"turn off|switch off|deactivate|disable",
            "set_brightness": r"set (?:brightness|level) to (\d+)%?",
            "set_color": r"set color to (\w+)",
            "set_temperature": r"set temperature to (\d+)",
            "get_state": r"what is|what's|how is|how's|status of|state of"
        }
        
    async def process_command(self, 
                            command: str,
                            room: Optional[str] = None) -> Tuple[bool, str, Optional[Dict]]:
        """
        Process a voice command and execute it.
        
        Args:
            command: The voice command text
            room: Optional room context
            
        Returns:
            Tuple of (success, response message, entity data)
        """
        try:
            # First, try to find matching entities
            matches = self.chroma.search_entities(
                query=command,
                room_filter=room
            )
            
            if not matches:
                return False, "I couldn't find any matching devices.", None
                
            # Get the best match
            best_match = matches[0]
            if best_match["confidence"] < 0.7:
                return False, "I'm not sure which device you want to control.", None
                
            # Parse the command intent
            intent, params = self._parse_command_intent(command)
            if not intent:
                return False, "I'm not sure what you want to do with that device.", None
                
            # Execute the command
            success = await self._execute_command(
                intent,
                best_match["entity_id"],
                params
            )
            
            if success:
                # Get updated state
                state = await self.ha.get_entity_state(best_match["entity_id"])
                return True, self._generate_response(intent, best_match, state), state
            else:
                return False, "I couldn't execute that command.", None
                
        except Exception as e:
            print(f"Error processing command: {e}")
            return False, "Sorry, I encountered an error processing your command.", None
            
    def _parse_command_intent(self, command: str) -> Tuple[Optional[str], Dict]:
        """
        Parse the command intent and parameters.
        
        Args:
            command: The command text
            
        Returns:
            Tuple of (intent, parameters)
        """
        command = command.lower()
        
        # Check each pattern
        for intent, pattern in self.command_patterns.items():
            match = re.search(pattern, command)
            if match:
                params = {}
                if intent == "set_brightness":
                    params["brightness"] = int(match.group(1))
                elif intent == "set_color":
                    params["color"] = match.group(1)
                elif intent == "set_temperature":
                    params["temperature"] = int(match.group(1))
                return intent, params
                
        return None, {}
        
    async def _execute_command(self,
                             intent: str,
                             entity_id: str,
                             params: Dict) -> bool:
        """
        Execute a command through Home Assistant.
        
        Args:
            intent: The command intent
            entity_id: Target entity ID
            params: Command parameters
            
        Returns:
            bool: True if successful
        """
        try:
            # Map intent to service
            if intent == "turn_on":
                return await self.ha.call_service(
                    entity_id.split(".")[0],
                    "turn_on",
                    entity_id
                )
            elif intent == "turn_off":
                return await self.ha.call_service(
                    entity_id.split(".")[0],
                    "turn_off",
                    entity_id
                )
            elif intent == "set_brightness":
                return await self.ha.call_service(
                    entity_id.split(".")[0],
                    "turn_on",
                    entity_id,
                    {"brightness_pct": params["brightness"]}
                )
            elif intent == "set_color":
                return await self.ha.call_service(
                    entity_id.split(".")[0],
                    "turn_on",
                    entity_id,
                    {"color_name": params["color"]}
                )
            elif intent == "set_temperature":
                return await self.ha.call_service(
                    entity_id.split(".")[0],
                    "turn_on",
                    entity_id,
                    {"color_temp": params["temperature"]}
                )
            elif intent == "get_state":
                return True  # State will be returned separately
                
            return False
            
        except Exception as e:
            print(f"Error executing command: {e}")
            return False
            
    def _generate_response(self,
                          intent: str,
                          entity: Dict,
                          state: Optional[Dict]) -> str:
        """
        Generate a response message for the user.
        
        Args:
            intent: The executed intent
            entity: Entity information
            state: Current entity state
            
        Returns:
            Response message
        """
        if intent == "get_state":
            if not state:
                return f"I couldn't get the state of {entity['name']}."
            return f"{entity['name']} is {state['state']}."
            
        action = {
            "turn_on": "turned on",
            "turn_off": "turned off",
            "set_brightness": f"set to {state['attributes'].get('brightness', 0)}% brightness",
            "set_color": f"set to {state['attributes'].get('color_name', 'unknown')} color",
            "set_temperature": f"set to {state['attributes'].get('color_temp', 0)}K"
        }.get(intent, "updated")
        
        return f"I've {action} {entity['name']}." 