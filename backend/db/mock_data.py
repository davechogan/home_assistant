from datetime import datetime, timedelta
import random
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base, User, VoiceInteraction, ModelTrainingLog

# Sample data
SAMPLE_COMMANDS = [
    "turn on the living room lights",
    "set the temperature to 72 degrees",
    "play some jazz music",
    "what's the weather like",
    "turn off the kitchen lights",
    "set an alarm for 7 am",
    "what time is it",
    "turn on the coffee maker",
    "close the garage door",
    "dim the bedroom lights to 50 percent"
]

SAMPLE_RESPONSES = [
    "I've turned on the living room lights",
    "Setting the temperature to 72 degrees",
    "Playing jazz music in the living room",
    "It's currently 75 degrees and sunny",
    "I've turned off the kitchen lights",
    "I've set an alarm for 7 am",
    "It's currently 3:45 PM",
    "Starting the coffee maker",
    "Closing the garage door",
    "Dimming the bedroom lights to 50 percent"
]

LOCATIONS = ["living room", "kitchen", "bedroom", "office", "garage"]

def create_mock_data():
    # Create SQLite database engine and session
    engine = create_engine('sqlite:///test.db')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Create test users
        users = []
        for i in range(3):
            user = User(
                username=f"user{i+1}",
                created_at=datetime.utcnow() - timedelta(days=30),
                preferences={
                    "voice": "en-US",
                    "temperature_unit": "fahrenheit",
                    "timezone": "America/New_York"
                }
            )
            users.append(user)
            session.add(user)
        session.commit()

        # Create voice interactions
        for user in users:
            # Create 20 interactions per user
            for i in range(20):
                # Randomly select command and response
                cmd_idx = random.randint(0, len(SAMPLE_COMMANDS) - 1)
                command = SAMPLE_COMMANDS[cmd_idx]
                response = SAMPLE_RESPONSES[cmd_idx]
                
                # Generate timestamp (spread over last 30 days)
                timestamp = datetime.utcnow() - timedelta(
                    days=random.randint(0, 30),
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59)
                )

                # Create interaction
                interaction = VoiceInteraction(
                    user_id=user.id,
                    timestamp=timestamp,
                    location=random.choice(LOCATIONS),
                    raw_audio_path=f"/audio/user{user.id}/interaction_{i}.wav",
                    transcribed_text=command,
                    command_executed=command,
                    system_response=response,
                    # 70% chance of having feedback
                    success=random.random() < 0.7 and random.choice([True, False]),
                    # 10% chance of error
                    error_message=random.random() < 0.1 and "Device not found" or None,
                    # 50% chance of being used in training
                    used_in_training=random.random() < 0.5,
                    training_round=random.randint(1, 3) if random.random() < 0.5 else None
                )
                session.add(interaction)

        # Create training logs
        for round_num in range(1, 4):
            log = ModelTrainingLog(
                timestamp=datetime.utcnow() - timedelta(days=round_num * 7),
                training_round=round_num,
                data_points_used=random.randint(50, 100),
                accuracy_before=random.uniform(0.7, 0.85),
                accuracy_after=random.uniform(0.85, 0.95),
                training_duration=random.randint(300, 1800),  # 5-30 minutes
                lora_weights_path=f"/models/lora_weights_round_{round_num}.pt",
                training_metadata={
                    "learning_rate": 0.0001,
                    "batch_size": 32,
                    "epochs": 3
                }
            )
            session.add(log)

        session.commit()
        print("Mock data created successfully!")

    except Exception as e:
        print(f"Error creating mock data: {e}")
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    create_mock_data() 