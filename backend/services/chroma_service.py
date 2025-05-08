"""
ChromaDB service for semantic search of Home Assistant entities.
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional, Tuple
import json
import os

class ChromaService:
    """Service for managing ChromaDB operations."""
    
    def __init__(self, persist_directory: str = "data/chroma"):
        """Initialize the ChromaDB service."""
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        self.collection = self.client.get_or_create_collection(
            name="ha_entities",
            metadata={"hnsw:space": "cosine"}
        )
        
    def add_entity(self, 
                  entity_id: str,
                  name: str,
                  type: str,
                  room: Optional[str] = None,
                  description: Optional[str] = None,
                  metadata: Optional[Dict] = None) -> bool:
        """
        Add a Home Assistant entity to the database.
        
        Args:
            entity_id: The Home Assistant entity ID
            name: Human-readable name
            type: Entity type (light, switch, scene, etc.)
            room: Room/zone where the entity is located
            description: Additional description
            metadata: Additional metadata
            
        Returns:
            bool: True if successful
        """
        try:
            # Create document text from entity information
            doc_text = f"{name} {type}"
            if room:
                doc_text += f" in {room}"
            if description:
                doc_text += f" {description}"
                
            # Create metadata
            entity_metadata = {
                "entity_id": entity_id,
                "type": type,
                "room": room,
                **(metadata or {})
            }
            
            # Add to collection
            self.collection.add(
                documents=[doc_text],
                metadatas=[entity_metadata],
                ids=[entity_id]
            )
            return True
            
        except Exception as e:
            print(f"Error adding entity to ChromaDB: {e}")
            return False
            
    def search_entities(self, 
                       query: str,
                       n_results: int = 5,
                       type_filter: Optional[str] = None,
                       room_filter: Optional[str] = None) -> List[Dict]:
        """
        Search for entities using semantic similarity.
        
        Args:
            query: The search query
            n_results: Number of results to return
            type_filter: Filter by entity type
            room_filter: Filter by room
            
        Returns:
            List of matching entities with their metadata
        """
        try:
            # Build where clause for filtering
            where = {}
            if type_filter:
                where["type"] = type_filter
            if room_filter:
                where["room"] = room_filter
                
            # Search collection
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where if where else None
            )
            
            # Format results
            matches = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )):
                matches.append({
                    "entity_id": metadata["entity_id"],
                    "name": doc.split()[0],  # First word is the name
                    "type": metadata["type"],
                    "room": metadata.get("room"),
                    "confidence": 1 - distance,  # Convert distance to confidence
                    "metadata": metadata
                })
                
            return matches
            
        except Exception as e:
            print(f"Error searching ChromaDB: {e}")
            return []
            
    def sync_ha_entities(self, entities: List[Dict]) -> bool:
        """
        Sync Home Assistant entities with ChromaDB.
        
        Args:
            entities: List of entity dictionaries from Home Assistant
            
        Returns:
            bool: True if successful
        """
        try:
            # Clear existing collection
            self.collection.delete(where={})
            
            # Add all entities
            for entity in entities:
                self.add_entity(
                    entity_id=entity["entity_id"],
                    name=entity["name"],
                    type=entity["type"],
                    room=entity.get("room"),
                    description=entity.get("description"),
                    metadata=entity.get("metadata")
                )
            return True
            
        except Exception as e:
            print(f"Error syncing entities: {e}")
            return False
            
    def get_entity(self, entity_id: str) -> Optional[Dict]:
        """
        Get a specific entity by ID.
        
        Args:
            entity_id: The entity ID to look up
            
        Returns:
            Entity metadata if found, None otherwise
        """
        try:
            results = self.collection.get(
                ids=[entity_id],
                include=["metadatas"]
            )
            if results["ids"]:
                return results["metadatas"][0]
            return None
            
        except Exception as e:
            print(f"Error getting entity: {e}")
            return None 