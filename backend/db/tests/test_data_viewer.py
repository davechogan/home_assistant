#!/usr/bin/env python3
"""
Test script to view and validate the mock data in the database.
Shows user statistics, interaction samples, and training progress.
"""

from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
import sys
import os

# Add the parent directory to the Python path so we can import our models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import Base, User, VoiceInteraction, ModelTrainingLog

def view_mock_data():
    # Create database connection
    engine = create_engine('sqlite:///test.db')
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # 1. User Statistics
        print("\n=== User Statistics ===")
        users = session.query(User).all()
        for user in users:
            total_interactions = session.query(VoiceInteraction).filter_by(user_id=user.id).count()
            feedback_interactions = session.query(VoiceInteraction).filter_by(
                user_id=user.id
            ).filter(VoiceInteraction.success.isnot(None)).count()
            
            print(f"\nUser: {user.username}")
            print(f"Total interactions: {total_interactions}")
            print(f"Interactions with feedback: {feedback_interactions}")
            print(f"Feedback rate: {(feedback_interactions/total_interactions)*100:.1f}%")
            print(f"Preferences: {user.preferences}")

        # 2. Sample Voice Interactions
        print("\n=== Sample Voice Interactions ===")
        interactions = session.query(VoiceInteraction).limit(5).all()
        for interaction in interactions:
            print(f"\nTimestamp: {interaction.timestamp}")
            print(f"Location: {interaction.location}")
            print(f"Command: {interaction.transcribed_text}")
            print(f"Response: {interaction.system_response}")
            print(f"Success: {interaction.success}")
            if interaction.error_message:
                print(f"Error: {interaction.error_message}")

        # 3. Training Progress
        print("\n=== Training Progress ===")
        training_logs = session.query(ModelTrainingLog).order_by(ModelTrainingLog.training_round).all()
        for log in training_logs:
            print(f"\nTraining Round {log.training_round}")
            print(f"Data points used: {log.data_points_used}")
            print(f"Accuracy: {log.accuracy_before:.1%} → {log.accuracy_after:.1%}")
            print(f"Duration: {log.training_duration/60:.1f} minutes")
            print(f"Model weights: {log.lora_weights_path}")
            print(f"Training settings: {log.training_metadata}")

        # 4. Training Data Statistics
        print("\n=== Training Data Statistics ===")
        total_interactions = session.query(VoiceInteraction).count()
        trainable_interactions = session.query(VoiceInteraction).filter(
            VoiceInteraction.success.isnot(None)
        ).count()
        used_in_training = session.query(VoiceInteraction).filter_by(
            used_in_training=True
        ).count()

        print(f"Total interactions: {total_interactions}")
        print(f"Interactions with feedback: {trainable_interactions}")
        print(f"Interactions used in training: {used_in_training}")
        print(f"Unused training data: {trainable_interactions - used_in_training}")

        # 5. Command Distribution
        print("\n=== Command Distribution ===")
        command_counts = session.query(
            VoiceInteraction.command_executed,
            func.count(VoiceInteraction.id).label('count')
        ).group_by(VoiceInteraction.command_executed).all()

        for command, count in command_counts:
            print(f"{command}: {count} times")

    except Exception as e:
        print(f"Error viewing mock data: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    view_mock_data() 