from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, Text, JSON, Float
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class User(Base):
    """User profile and preferences"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    preferences = Column(JSON)  # User preferences and settings
    voice_interactions = relationship("VoiceInteraction", back_populates="user")

class VoiceInteraction(Base):
    """Log of all voice interactions and training data"""
    __tablename__ = "voice_interactions"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    location = Column(String(100))  # Room/location where command was given
    raw_audio_path = Column(String(255))  # Path to stored audio file
    transcribed_text = Column(Text)  # STT output
    command_executed = Column(Text)  # Actual command sent to HA
    system_response = Column(Text)  # TTS response
    success = Column(Boolean, nullable=True)  # User feedback (yes/no) - NULL if no feedback given
    error_message = Column(Text, nullable=True)  # Error if any
    used_in_training = Column(Boolean, default=False)  # Whether this interaction has been used for training
    training_round = Column(Integer, nullable=True)  # Which training iteration used this data
    user = relationship("User", back_populates="voice_interactions")

class ModelTrainingLog(Base):
    """Log of model training sessions"""
    __tablename__ = "model_training_logs"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    training_round = Column(Integer)
    data_points_used = Column(Integer)
    accuracy_before = Column(Float)
    accuracy_after = Column(Float)
    training_duration = Column(Integer)  # Duration in seconds
    lora_weights_path = Column(String(255))  # Path to stored LoRA weights
    training_metadata = Column(JSON)  # Additional training metadata 