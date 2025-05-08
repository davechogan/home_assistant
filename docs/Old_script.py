import os, subprocess, numpy as np, requests, json, sounddevice as sd
from resemblyzer import VoiceEncoder
from dotenv import load_dotenv
import chromadb
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import scipy.io.wavfile
import pickle
from datetime import datetime, timedelta
import time
import sys
import re
import signal
import threading
import queue
from chromadb.utils import embedding_functions
from chromadb.config import Settings
import platform

# Try to import pytz, install if missing
try:
    import pytz
    TIMEZONE_AVAILABLE = True
except ImportError:
    print("⚠️ pytz module not found. Time-based features will use local time.")
    print("💡 To install: pip install pytz")
    TIMEZONE_AVAILABLE = False

# Try to import pvporcupine, install if missing
try:
    import pvporcupine
    PORCUPINE_AVAILABLE = True
except ImportError:
    print("⚠️ pvporcupine module not found. Wake word detection will not be available.")
    print("💡 To install: pip install pvporcupine")
    PORCUPINE_AVAILABLE = False

# Try to import text-to-speech libraries
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
    print("✅ pyttsx3 module is available as fallback")
except ImportError:
    PYTTSX3_AVAILABLE = False
    print("⚠️ pyttsx3 module not found. Will use system TTS only.")
    print("💡 To install: pip install pyttsx3")

try:
    from gtts import gTTS
    import pygame
    GTTS_AVAILABLE = True
    print("✅ gTTS module is available as fallback")
except ImportError:
    GTTS_AVAILABLE = False
    print("⚠️ gTTS module not found. Will use system TTS only.")
    print("💡 To install: pip install gtts pygame")

# Initialize pygame mixer for audio playback if gTTS is available
if GTTS_AVAILABLE:
    pygame.mixer.init()

# Load environment variables
load_dotenv()

# Get access key from environment
PORCUPINE_ACCESS_KEY = os.getenv("PORCUPINE_ACCESS_KEY")

# === Configuration ===
# Base directory for all data files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# File paths
HA_DATA_FILE = os.path.join(BASE_DIR, "entities.jsonl")
WAV_FILE = os.path.join(BASE_DIR, "mic_input.wav")
WHISPER_BIN = os.path.join(BASE_DIR, "whisper.cpp/build/bin/whisper-cli")
MODEL_PATH = os.path.join(BASE_DIR, "whisper.cpp/models/ggml-base.en.bin")
PROFILE_PATH = os.path.join(BASE_DIR, "speaker_profiles.npy")
CONTEXT_PATH = os.path.join(BASE_DIR, "conversation_context.pkl")
LEARNING_PATH = os.path.join(BASE_DIR, "learning_examples.json")
CONTEXT_HISTORY_PATH = os.path.join(BASE_DIR, "conversation_context_history.jsonl")
CONTEXT_RETENTION_DAYS = 7  # 1 week

# TTS Configuration
VOICE_LANGUAGE = "en"
VOICE_RATE = 150
VOICE_VOLUME = 1.0
VOICE_ID = None  # Set to a specific voice ID if needed
TTS_FILE = os.path.join(BASE_DIR, "tts_output.mp3")
MACOS_SAY_VOICE = "Moira"  # Irish English female voice

# === Ollama Configuration ===
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mixtral")
OLLAMA_REQUEST_TIMEOUT = 120  # Seconds

# Ollama client for persistent connection
class OllamaClient:
    """Client for maintaining persistent connection to Ollama server"""
    
    def __init__(self, api_url=OLLAMA_API_URL, model=OLLAMA_MODEL, timeout=OLLAMA_REQUEST_TIMEOUT):
        self.api_url = api_url
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()
        self.initialized = False
        self.last_used = time.time()
        self.lock = threading.Lock()
        
    def initialize(self,):
        """Initialize the model by sending a small prompt to ensure it's loaded"""
        try:
            print(f"🔄 Initializing Ollama with model: {self.model}")
            response = self.session.post(
                f"{self.api_url}/generate",
                json={"model": self.model, "prompt": "Hello", "stream": False},
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                self.initialized = True
                self.last_used = time.time()
                print(f"✅ Successfully initialized Ollama model: {self.model}")
                return True
            else:
                print(f"❌ Failed to initialize Ollama model: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error initializing Ollama: {str(e)}")
            return False
    
    def generate(self, prompt):
        """Generate a response from the model"""
        with self.lock:
            # Check if too much time has passed since last use
            if self.initialized and time.time() - self.last_used > 600:  # 10 minutes
                print("⚠️ Ollama connection might have timed out. Reinitializing...")
                self.initialized = False
            
            # Initialize if needed
            if not self.initialized:
                if not self.initialize():
                    raise Exception("Failed to initialize Ollama model")
            
            try:
                response = self.session.post(
                    f"{self.api_url}/generate",
                    json={"model": self.model, "prompt": prompt, "stream": False},
                    timeout=self.timeout
                )
                
                self.last_used = time.time()
                
                if response.status_code == 200:
                    return response.json().get("response", "")
                else:
                    print(f"❌ Ollama API error: {response.status_code}")
                    self.initialized = False
                    return None
            except Exception as e:
                print(f"❌ Error communicating with Ollama: {str(e)}")
                self.initialized = False
                return None

# === Global Variables ===
tts_engine_busy = False
assistant_speaking = False
tts_engine_lock = threading.Lock()
tts_engine = None
shutdown_in_progress = False
ha_data = None  # Replace chroma_db with ha_data
voice_encoder = None
wake_word_thread = None
ollama_client = None

# Set tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Audio queue for wake word detection
audio_queue = queue.Queue()

# Event for wake word detection
wake_word_detected = threading.Event()

# Running flag
running = True

# Timing variables
start_time = time.time()
last_checkpoint = start_time

# Home Assistant Configuration
HA_URL = "http://192.168.50.7:8123/api"
HA_TOKEN = os.getenv("HA_TOKEN")  # Load from .env file
if not HA_TOKEN:
    print("⚠️ HA_TOKEN not found in environment variables. Home Assistant integration will be limited.")
    HA_TOKEN = "your_token_here"  # Placeholder token

HEADERS = {
    "Authorization": f"Bearer {HA_TOKEN}",
    "Content-Type": "application/json"
}

# === Debug Mode ===
DEBUG = True  # Set to False to disable detailed debugging
SKIP_RECORDING = False  # Set to True to skip recording and use existing WAV file
SKIP_SPEAKER_ID = False  # Set to True to skip speaker identification
DEFAULT_USER = "Dave"  # Default user if speaker identification is skipped

# === Wake Word Configuration ===
WAKE_WORD = "jarvis"  # Options: "jarvis", "computer", "alexa", "hey google", "hey siri", etc.
WAKE_WORD_SENSITIVITY = 0.5  # Between 0 and 1, higher is more sensitive
LISTEN_TIMEOUT = 5  # Seconds to listen after wake word before timing out

# === Voice Configuration ===
VOICE_RATE = 175  # Words per minute (pyttsx3 only)
VOICE_VOLUME = 0.9  # Volume level (0.0 to 1.0)
VOICE_GENDER = "female"  # "male" or "female" (pyttsx3 only)

# === User Account Information ===
# Define user accounts and preferences
user_accounts = {
    "David": {
        "media_player": "media_player.living_room_speaker",
        "spotify_account": "spotify:user:daviduser",
        "preferred_temp": 72,
        "preferred_brightness": 70
    },
    "Dave": {
        "media_player": "media_player.bedroom_speaker",
        "spotify_account": "spotify:user:daveuser",
        "preferred_temp": 70,
        "preferred_brightness": 60
    }
}

# === Home Assistant Helper Class ===
class HomeAssistantAPI:
    """Helper class for Home Assistant API calls"""
    
    def __init__(self, base_url, token):
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
    def get_domains(self):
        """Get available domains from Home Assistant"""
        try:
            response = requests.get(f"{self.base_url}/services", headers=self.headers, timeout=5)
            if response.status_code == 200:
                services = response.json()
                return list(services.keys())
            else:
                print(f"⚠️ Error fetching domains: {response.status_code}")
                return ["light", "switch", "media_player", "climate", "cover"]
        except Exception as e:
            print(f"⚠️ Error fetching domains: {str(e)}")
            return ["light", "switch", "media_player", "climate", "cover"]
            
    def get_services(self):
        """Get available services from Home Assistant"""
        try:
            response = requests.get(f"{self.base_url}/services", headers=self.headers, timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"⚠️ Error fetching services: {response.status_code}")
                return {}
        except Exception as e:
            print(f"⚠️ Error fetching services: {str(e)}")
            return {}
            
    def get_devices(self):
        """Get available devices from Home Assistant"""
        try:
            response = requests.get(f"{self.base_url}/states", headers=self.headers, timeout=5)
            if response.status_code == 200:
                states = response.json()
                devices = []
                
                for state in states:
                    entity_id = state.get("entity_id", "")
                    if "." in entity_id:
                        domain = entity_id.split(".")[0]
                        name = state.get("attributes", {}).get("friendly_name", entity_id)
                        devices.append(DeviceInfo(entity_id, name, domain))
                
                return devices
            else:
                print(f"⚠️ Error fetching devices: {response.status_code}")
                return []
        except Exception as e:
            print(f"⚠️ Error fetching devices: {str(e)}")
            return []

# === Device Information Class ===
class DeviceInfo:
    """Simple class to hold device information"""
    
    def __init__(self, entity_id, name, domain, area=None):
        self.entity_id = entity_id
        self.name = name
        self.domain = domain
        self.area = area
        
    def __str__(self):
        area_str = f" in {self.area}" if self.area else ""
        return f"{self.name} ({self.entity_id}){area_str}"

# Initialize Home Assistant API
ha = HomeAssistantAPI(HA_URL, HA_TOKEN)

# === Performance Tracking ===
def checkpoint(label):
    """Print timing information for a checkpoint"""
    global last_checkpoint
    now = time.time()
    elapsed = now - last_checkpoint
    total = now - start_time
    print(f"⏱️ {label}: {elapsed:.2f}s (Total: {total:.2f}s)")
    last_checkpoint = now

# === Text-to-Speech Functions ===
def speak(text):
    """Convert text to speech and play it"""
    global tts_engine_busy, assistant_speaking
    
    if not text:
        return
        
    print(f"🔊 Speaking: {text}")
    
    if tts_engine_busy:
        print(f"⚠️ TTS engine busy, skipping speech")
        print(f"💬 Assistant would say: {text}")
        return
    
    try:
        tts_engine_busy = True
        assistant_speaking = True
        
        # First try to use macOS 'say' command if available
        if platform.system() == "Darwin":
            try:
                # Use a shorter timeout for the say command
                subprocess.run(['say', '-v', MACOS_SAY_VOICE, text], check=True, timeout=5)
                return
            except (subprocess.SubprocessError, FileNotFoundError) as e:
                print(f"⚠️ macOS 'say' command failed: {str(e)}")
                print("🔄 Falling back to alternative TTS methods...")
            except subprocess.TimeoutExpired:
                print("⚠️ TTS command timed out, skipping speech")
                return
        
        # Fallback to pyttsx3 if available
        if PYTTSX3_AVAILABLE:
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', VOICE_RATE)
                engine.setProperty('volume', VOICE_VOLUME)
                if VOICE_ID:
                    engine.setProperty('voice', VOICE_ID)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
                del engine
                return
            except Exception as e:
                print(f"⚠️ pyttsx3 TTS engine error: {str(e)}")
                print("🔄 Falling back to gTTS...")
        
        # Fallback to gTTS if available
        if GTTS_AVAILABLE:
            try:
                tts = gTTS(text=text, lang=VOICE_LANGUAGE, slow=False)
                tts.save(TTS_FILE)
                if os.path.exists(TTS_FILE):
                    pygame.mixer.init()
                    pygame.mixer.music.load(TTS_FILE)
                    pygame.mixer.music.set_volume(VOICE_VOLUME)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
                return
            except Exception as e:
                print(f"⚠️ gTTS engine error: {str(e)}")
        
        # If all TTS methods fail, just print the text
        print(f"💬 Assistant would say: {text}")
        
    except Exception as e:
        print(f"❌ TTS Error: {str(e)}")
        print(f"💬 Assistant would say: {text}")
    finally:
        tts_engine_busy = False
        assistant_speaking = False

# === Time Awareness ===
# Set your timezone (will use local time if pytz is not available)
TIMEZONE_NAME = 'America/New_York'  # Change to your timezone

def get_time_context():
    """Get time-based context information"""
    if TIMEZONE_AVAILABLE:
        try:
            timezone = pytz.timezone(TIMEZONE_NAME)
            now = datetime.now(timezone)
        except pytz.exceptions.UnknownTimeZoneError:
            print(f"⚠️ Unknown timezone: {TIMEZONE_NAME}. Using local time.")
            now = datetime.now()
    else:
        now = datetime.now()
    
    # Time of day categories
    hour = now.hour
    if 5 <= hour < 9:
        time_of_day = "early_morning"
    elif 9 <= hour < 12:
        time_of_day = "morning"
    elif 12 <= hour < 17:
        time_of_day = "afternoon"
    elif 17 <= hour < 20:
        time_of_day = "evening"
    elif 20 <= hour < 23:
        time_of_day = "night"
    else:
        time_of_day = "late_night"
    
    # Brightness recommendations based on time of day
    brightness_presets = {
        "early_morning": 20,  # Dim in early morning
        "morning": 60,
        "afternoon": 80,
        "evening": 60,
        "night": 40,
        "late_night": 15   # Very dim late at night
    }
    
    # Color temperature recommendations (if your lights support it)
    color_temp_presets = {
        "early_morning": 2700,  # Warm in morning
        "morning": 4000,
        "afternoon": 5500,      # Cool/neutral during day
        "evening": 3500,
        "night": 2700,          # Warm at night
        "late_night": 2200      # Very warm late at night
    }
    
    return {
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "time_of_day": time_of_day,
        "is_weekend": now.weekday() >= 5,
        "recommended_brightness": brightness_presets[time_of_day],
        "recommended_color_temp": color_temp_presets[time_of_day]
    }

# === Context Management ===
def save_context_history(context):
    """
    Append the current context to the history file as a JSON line.
    The timestamp is stored in ISO format for easy parsing.
    """
    context_copy = context.copy()
    context_copy['timestamp'] = context_copy['timestamp'].isoformat()
    with open(CONTEXT_HISTORY_PATH, 'a') as f:
        f.write(json.dumps(context_copy) + '\n')
    print(f"💾 Appended context to history at {context_copy['timestamp']}")


def load_recent_contexts():
    """
    Load all contexts from the last CONTEXT_RETENTION_DAYS days from the history file.
    Returns a list of context dicts, most recent last.
    """
    if not os.path.exists(CONTEXT_HISTORY_PATH):
        return []
    cutoff = datetime.now() - timedelta(days=CONTEXT_RETENTION_DAYS)
    recent_contexts = []
    with open(CONTEXT_HISTORY_PATH, 'r') as f:
        for line in f:
            try:
                ctx = json.loads(line)
                ts = datetime.fromisoformat(ctx['timestamp'])
                if ts >= cutoff:
                    ctx['timestamp'] = ts
                    recent_contexts.append(ctx)
            except Exception as e:
                print(f"⚠️ Skipping invalid context line: {e}")
    return recent_contexts


def get_latest_context():
    """
    Get the most recent context within the retention period.
    Returns a context dict or a new context if none found.
    """
    contexts = load_recent_contexts()
    if contexts:
        return contexts[-1]
    else:
        return create_new_context()


def print_recent_contexts():
    """
    Print all recent contexts from the last CONTEXT_RETENTION_DAYS days.
    """
    contexts = load_recent_contexts()
    for ctx in contexts:
        print(json.dumps(ctx, indent=2, default=str))

def create_new_context():
    """Create a new conversation context"""
    return {
        'timestamp': datetime.now(),
        'history': [],
        'last_transcript': '',
        'last_intent': None,
        'last_result': None,
        'user': None,
        'last_devices': [],  # Track last devices that were controlled
        'last_action': None,  # Track last action performed
        'last_parameters': {}  # Track last parameters used
    }

def save_context(context):
    """
    Save conversation context to file and append to history.
    The latest context is still saved as a pickle for backward compatibility.
    """
    try:
        # Update timestamp
        context['timestamp'] = datetime.now()
        # Save latest context (for backward compatibility)
        with open(CONTEXT_PATH, 'wb') as f:
            pickle.dump(context, f)
        print(f"💾 Saved context at {context['timestamp'].strftime('%H:%M:%S')}")
        # Also append to history
        save_context_history(context)
    except Exception as e:
        print(f"⚠️ Error saving context: {str(e)}")

def load_context():
    """
    Load the most recent conversation context from history within retention period.
    If none found, create a new context.
    """
    recent_contexts = load_recent_contexts()
    if recent_contexts:
        context = recent_contexts[-1]
        print(f"🔄 Loaded context from history at {context['timestamp']}")
        return context
    else:
        return create_new_context()

# === Home Assistant Integration ===
def send_actions_to_home_assistant(intent_json, context, test_mode=False):
    """Send actions to Home Assistant based on the intent JSON"""
    print("\n🏠 Sending actions to Home Assistant...")
    
    success = True  # Track overall success
    controlled_devices = []
    last_action = None
    last_parameters = {}
    response_message = ""
    action_device_map = {}  # Map action to set of devices
    error_messages = set()  # Track unique error messages

    # Get devices from ha_data instead of ChromaDB
    devices = []
    if ha_data and "entities" in ha_data:
        for entity in ha_data["entities"]:
            devices.append(DeviceInfo(
                entity_id=entity["entity_id"],
                name=entity.get("name", entity["entity_id"]),
                domain=entity.get("domain", entity["entity_id"].split(".")[0]),
                area=entity.get("area", "")
            ))

    # Check if we're in test mode
    if test_mode:
        speak("I'll simulate these actions without actually controlling your devices.")
        print("🧪 TEST MODE: Actions will be simulated, not sent to Home Assistant")
        for action in intent_json.get("actions", []):
            device_type = action.get("device_type", "unknown")
            action_name = action.get("action", "unknown")
            entity = action.get("parameters", {}).get("entity_id", "unknown device")
            
            if isinstance(entity, list):
                entity = ", ".join(entity)
                
            response_message += f"I would {action_name} your {entity}. "
            
            print(f"✅ Would trigger: {device_type}.{action_name} with parameters:")
            print(json.dumps(action.get("parameters", {}), indent=2))
        
        speak(response_message)
        return True
    
    timeout = 10  # Increased timeout to 10 seconds
    
    for action in intent_json.get("actions", []):
        domain = action.get("device_type")
        service = action.get("action")
        entity_id = action.get("entity_id", action.get("parameters", {}).get("entity_id"))
        
        # Defensive: Try to infer entity_id if missing
        if not entity_id:
            inferred_room = intent_json.get("inferred_room", "")
            # If the action is for "light" and a room is specified, get all lights in that room by area
            if domain == "light" and inferred_room:
                norm_room = normalize_area(inferred_room)
                possible_devices = [
                    d for d in devices
                    if d.domain == "light" and d.area and normalize_area(d.area) == norm_room
                ]
                if not possible_devices:
                    # Fallback to substring match
                    possible_devices = [
                        d for d in devices
                        if d.domain == "light" and (
                            norm_room in normalize_area(d.name) or
                            norm_room in normalize_area(d.entity_id)
                        )
                    ]
                if possible_devices:
                    entity_id = [d.entity_id for d in possible_devices]
                    print(f"🔎 Inferred all lights in area '{inferred_room}': {entity_id}")
                    print(f"Selected devices for area '{inferred_room}': {[d.entity_id for d in possible_devices]}")
                elif context.get('last_devices'):
                    entity_id = context['last_devices']
                    print(f"🔎 Using last controlled devices from context: {entity_id}")
                else:
                    print(f"❌ No lights found in area '{inferred_room}'")
                    speak(f"I couldn't find any lights in the {inferred_room}. Please specify the device.")
                    success = False
                    continue
            else:
                # Fallback to your existing logic for other domains or no room specified
                possible_devices = [
                    d for d in devices
                    if d.domain == domain and (inferred_room.lower() in d.name.lower() or inferred_room.lower() in d.entity_id.lower())
                ]
                if possible_devices:
                    entity_id = [d.entity_id for d in possible_devices]
                    print(f"🔎 Inferred entity_id(s): {entity_id}")
                elif context.get('last_devices'):
                    entity_id = context['last_devices']
                    print(f"🔎 Using last controlled devices from context: {entity_id}")
                else:
                    print(f"❌ No entity_id found for domain '{domain}' and room '{inferred_room}'")
                    speak(f"I couldn't determine which {domain} you meant in the {inferred_room}. Please specify the device.")
                    success = False
                    continue

        # If entity_id is a list of one, flatten it
        if isinstance(entity_id, list) and len(entity_id) == 1:
            entity_id = entity_id[0]
        
        # Debug information
        print(f"Processing action: {json.dumps(action, indent=2)}")
        
        if not domain or not service:
            print(f"❌ Missing domain or service in action: {action}")
            success = False
            continue
            
        # Prepare service URL and payload
        service_url = f"{HA_URL}/services/{domain}/{service}"
        
        # Extract all parameters from the action
        payload = {}
        if "parameters" in action and isinstance(action["parameters"], dict):
            payload.update(action["parameters"])
        
        # Add entity_id if available
        if entity_id:
            payload["entity_id"] = entity_id
            
        print(f"🔄 Calling {service_url} with payload: {json.dumps(payload, indent=2)}")
        
        try:
            # Add timeout to avoid hanging
            resp = requests.post(service_url, headers=HEADERS, json=payload, timeout=timeout)
            if resp.status_code == 200:
                print(f"✅ Successfully triggered {domain}.{service}")
                # Only add to action_device_map if successful
                key = f"{domain}.{service}"
                if key not in action_device_map:
                    action_device_map[key] = set()
                if isinstance(entity_id, list):
                    action_device_map[key].update(entity_id)
                else:
                    action_device_map[key].add(entity_id)
            else:
                print(f"❌ Failed to trigger {domain}.{service}: {resp.status_code}")
                print(f"Response: {resp.text}")
                error_messages.add(f"I couldn't {service.replace('_', ' ')} your {entity_id if isinstance(entity_id, str) else ', '.join(entity_id)}.")
                success = False
        except requests.exceptions.Timeout:
            print(f"⏱️ Request timed out. Check if Home Assistant is running at {HA_URL}")
            error_messages.add("I couldn't reach your Home Assistant server.")
            success = False
        except requests.exceptions.ConnectionError:
            print(f"🔌 Connection error. Check if Home Assistant is running and accessible")
            error_messages.add("I couldn't connect to your Home Assistant server.")
            success = False
        except Exception as e:
            print(f"❌ Error sending request: {str(e)}")
            error_messages.add(f"I encountered an error: {str(e)}.")
            success = False
    
    # Build a more natural response message
    if action_device_map:
        # Group actions by domain and service
        for key, devices_set in action_device_map.items():
            domain, service = key.split(".")
            device_count = len(devices_set)
            
            # Get the room/area if available
            room = intent_json.get("inferred_room", "")
            room_str = f" in the {room}" if room else ""
            
            # Get device names instead of entity_ids
            device_names = []
            for device_id in devices_set:
                # Find the device in our list
                device = next((d for d in devices if d.entity_id == device_id), None)
                if device:
                    device_names.append(device.name)
                else:
                    device_names.append(device_id)
            
            # Create natural language response based on device count and action
            if device_count > 1:
                if domain == "light":
                    if service == "turn_on":
                        response_message += f"I've turned on all the lights{room_str} for you. "
                    elif service == "turn_off":
                        response_message += f"I've turned off all the lights{room_str} for you. "
                    else:
                        response_message += f"I've {service.replace('_', ' ')} all the lights{room_str}. "
                else:
                    response_message += f"I've {service.replace('_', ' ')} all the {domain}s{room_str}. "
            else:
                device_name = device_names[0] if device_names else next(iter(devices_set))
                response_message += f"I've {service.replace('_', ' ')} the {device_name}{room_str}. "
    
    # Add unique error messages
    if error_messages:
        response_message = " ".join(error_messages) + " " + response_message
    
    # Update the context with the controlled devices
    if controlled_devices:
        context['last_devices'] = controlled_devices
        context['last_action'] = last_action
        context['last_parameters'] = last_parameters
        print(f"📝 Updated context with controlled devices: {', '.join(controlled_devices)}")
    
    # Speak the response
    if response_message:
        speak(response_message)
    elif success:
        speak("I've completed your request successfully.")
    else:
        speak("I had trouble completing your request. Please check the logs for details.")
    
    return success

# === Wake Word Detection ===
def porcupine_callback(frame_count):
    """Callback function for Porcupine wake word detection"""
    def callback(indata, frame_count, time_info, status):
        if status:
            print(f"⚠️ Audio callback status: {status}")
        audio_queue.put(indata.copy())
    return callback

def wake_word_listener():
    """Listen for wake word in a separate thread"""
    global running
    
    if not PORCUPINE_AVAILABLE:
        print("❌ Porcupine not available. Wake word detection disabled.")
        return
    
    try:
        # Initialize Porcupine with the selected wake word
        porcupine = pvporcupine.create(
            access_key=PORCUPINE_ACCESS_KEY,
            keywords=[WAKE_WORD],
            sensitivities=[WAKE_WORD_SENSITIVITY]
        )
        
        # Audio parameters
        sample_rate = porcupine.sample_rate
        frame_length = porcupine.frame_length
        
        # Start audio stream
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype='int16',
            blocksize=frame_length,
            callback=porcupine_callback(frame_length)
        ):
            print(f"🎤 Listening for wake word: '{WAKE_WORD}'...")
            
            while running:
                # Get audio frame from queue
                try:
                    frame = audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                # Process audio frame
                pcm = frame.flatten().astype(np.int16)
                
                # Check for wake word
                result = porcupine.process(pcm)
                if result >= 0:
                    print(f"🔔 Wake word detected! ({WAKE_WORD})")
                    speak("I'm listening")
                    wake_word_detected.set()
                    
                    # Wait for processing to complete before listening again
                    time.sleep(10)
                    wake_word_detected.clear()
                    print(f"🎤 Listening for wake word: '{WAKE_WORD}'...")
    
    except Exception as e:
        print(f"❌ Error in wake word detection: {str(e)}")
    finally:
        if 'porcupine' in locals():
            porcupine.delete()

# === Signal Handling ===
def signal_handler(sig, frame):
    """Handle Ctrl+C and other signals"""
    global running, shutdown_in_progress
    
    if shutdown_in_progress:
        print("\n⚠️ Shutdown already in progress, forcing exit...")
        sys.exit(1)
    
    print("\n⏹️ Stopping voice assistant...")
    shutdown_in_progress = True
    
    # Set running flag to False
    running = False
    
    # Clear any pending wake word detection
    wake_word_detected.clear()
    
    # Clean up resources
    cleanup_resources()
    
    # Final goodbye
    print("👋 Voice assistant stopped.")
    sys.exit(0)

def cleanup_resources():
    """Clean up all resources"""
    global running, wake_word_thread, ha_data, voice_encoder, tts_engine, ollama_client, shutdown_in_progress
    
    if not shutdown_in_progress:
        return
        
    print("\n🧹 Cleaning up resources...")
    
    # Set running flag to False
    running = False
    
    # Clear any pending wake word detection
    wake_word_detected.clear()
    
    # Stop any ongoing TTS
    if 'tts_engine' in globals() and tts_engine:
        try:
            tts_engine.stop()
        except:
            pass
    
    # Clean up global instances
    if 'voice_encoder' in globals() and voice_encoder:
        try:
            del voice_encoder
        except:
            pass
    
    if 'ha_data' in globals() and ha_data:
        try:
            del ha_data
        except:
            pass
    
    if 'ollama_client' in globals() and ollama_client:
        try:
            del ollama_client
        except:
            pass
    
    # Wait for wake word thread to finish with timeout
    if 'wake_word_thread' in globals() and wake_word_thread and wake_word_thread.is_alive():
        print("Waiting for wake word detection to stop...")
        wake_word_thread.join(timeout=2)
        if wake_word_thread.is_alive():
            print("⚠️ Wake word thread did not stop gracefully")
    
    print("✅ Resources cleaned up.")

# === ChromaDB Integration ===
def initialize_chromadb():
    """Initialize ChromaDB for storing and retrieving smart home information"""
    try:
        # Create a persistent client
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        
        # Get existing collections
        collections = {}
        
        # Get devices collection (contains entities)
        try:
            devices_collection = chroma_client.get_or_create_collection(
                name="devices",
                metadata={"description": "Smart home devices with area information"}
            )
            collections["devices"] = devices_collection
            print(f"✅ Found devices collection with {devices_collection.count()} items")
        except Exception as e:
            print(f"⚠️ Devices collection not found: {str(e)}")
            collections["devices"] = None
        
        # Get services collection (contains available actions)
        try:
            services_collection = chroma_client.get_or_create_collection(
                name="ha_services",
                metadata={"description": "Home Assistant services with area context"}
            )
            collections["services"] = services_collection
            print(f"✅ Found ha_services collection with {services_collection.count()} items")
        except Exception as e:
            print(f"⚠️ Services collection not found: {str(e)}")
            collections["services"] = None
        
        # Get areas collection (contains room/area information)
        try:
            areas_collection = chroma_client.get_or_create_collection(
                name="areas",
                metadata={"description": "Smart home areas and their relationships"}
            )
            collections["areas"] = areas_collection
            print(f"✅ Found areas collection with {areas_collection.count()} items")
        except Exception as e:
            print(f"⚠️ Areas collection not found: {str(e)}")
            collections["areas"] = None
        
        return {
            "client": chroma_client,
            **collections
        }
    except Exception as e:
        print(f"⚠️ Error initializing ChromaDB: {str(e)}")
        return None

def get_devices_from_chromadb(db=None):
    """Get devices from ChromaDB"""
    global ha_data
    db = db or ha_data
    
    if not db or "devices" not in db or db["devices"] is None:
        print("⚠️ Devices collection not available")
        return None
            
    try:
        # Check if collection has data
        if db["devices"].count() == 0:
            print("⚠️ Devices collection is empty")
            return []
            
        # Query all devices
        result = db["devices"].get()
        
        if not result or not result["metadatas"] or len(result["metadatas"]) == 0:
            print("⚠️ No device data returned from ChromaDB")
            return []
        
        devices = []
        domains = set()  # Track unique domains
        
        for i, metadata in enumerate(result["metadatas"]):
            if metadata:
                entity_id = metadata.get("entity_id", "")
                if not entity_id and "ids" in result and i < len(result["ids"]):
                    # Use ID as entity_id if not in metadata
                    entity_id = result["ids"][i]
                
                # Extract domain from entity_id
                domain = ""
                if "." in entity_id:
                    domain = entity_id.split(".")[0]
                elif "domain" in metadata:
                    domain = metadata["domain"]
                
                # Get name from metadata or documents
                name = metadata.get("name", entity_id)
                if not name and "documents" in result and i < len(result["documents"]):
                    # Try to extract name from document
                    doc = result["documents"][i]
                    if "labeled '" in doc:
                        name_part = doc.split("labeled '")[1]
                        if "'" in name_part:
                            name = name_part.split("'")[0]
                
                # Get area from metadata
                area = metadata.get("area", "")
                
                if entity_id and domain:
                    devices.append(DeviceInfo(entity_id, name, domain, area))
                    domains.add(domain)
        
        print(f"📊 Retrieved {len(devices)} devices across {len(domains)} domains from ChromaDB")
        return devices
    except Exception as e:
        print(f"⚠️ Error retrieving devices from ChromaDB: {str(e)}")
        return None

def get_domains_from_devices(devices):
    """Extract unique domains from device list"""
    if not devices:
        return []
    
    domains = set()
    for device in devices:
        if device.domain:
            domains.add(device.domain)
    
    return sorted(list(domains))

def get_services_from_chromadb(db):
    """Get services from ChromaDB"""
    if not db or "services" not in db or db["services"] is None:
        print("⚠️ Services collection not available")
        return None
            
    try:
        # Check if collection has data
        if db["services"].count() == 0:
            print("⚠️ Services collection is empty")
            return {}
            
        # Query all services
        result = db["services"].get()
        
        if not result or not result["metadatas"] or len(result["metadatas"]) == 0:
            print("⚠️ No service data returned from ChromaDB")
            return {}
        
        services = {}
        
        for i, metadata in enumerate(result["metadatas"]):
            if metadata:
                domain = metadata.get("domain", "")
                action = metadata.get("action", "")
                
                if not domain and "ids" in result and i < len(result["ids"]):
                    # Try to extract domain from ID
                    id_parts = result["ids"][i].split(".")
                    if len(id_parts) >= 2:
                        domain = id_parts[0]
                        action = id_parts[1]
                
                if domain and action:
                    if domain not in services:
                        services[domain] = {}
                    
                    # Get description from document if available
                    description = ""
                    if "documents" in result and i < len(result["documents"]):
                        description = result["documents"][i]
                    
                    services[domain][action] = {
                        "description": description
                    }
        
        print(f"📊 Retrieved services for {len(services)} domains from ChromaDB")
        return services
    except Exception as e:
        print(f"⚠️ Error retrieving services from ChromaDB: {str(e)}")
        return None

def populate_chromadb_from_export_script():
    """Populate ChromaDB by running the ha_ws_area_map.py script"""
    print("\n🔄 Populating ChromaDB using ha_ws_area_map.py...")
    
    try:
        # Check if the export script exists
        export_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ha_ws_area_map.py")
        if not os.path.exists(export_script_path):
            print(f"❌ Export script not found at {export_script_path}")
            return False
        
        # Run the export script as a subprocess
        print("🔄 Running ha_ws_area_map.py...")
        result = subprocess.run(
            [sys.executable, export_script_path],
            capture_output=True,
            text=True,
            check=False
        )
        
        # Check if the script ran successfully
        if result.returncode != 0:
            print(f"❌ Export script failed with exit code {result.returncode}")
            print(f"Error output: {result.stderr}")
            return False
        
        # Print the output from the script
        print(f"📋 Export script output:")
        for line in result.stdout.splitlines():
            print(f"  {line}")
        
        print("✅ Successfully ran ha_ws_area_map.py")
        
        # Check if ChromaDB was populated
        db = initialize_chromadb()
        if db is None:
            print("❌ Failed to initialize ChromaDB after export")
            return False
        
        # Check if collections have data
        devices_count = db["devices"].count() if "devices" in db and db["devices"] is not None else 0
        services_count = db["services"].count() if "services" in db and db["services"] is not None else 0
        areas_count = db["areas"].count() if "areas" in db and db["areas"] is not None else 0
        
        print(f"📊 ChromaDB status after export:")
        print(f"  - Devices: {devices_count}")
        print(f"  - Services: {services_count}")
        print(f"  - Areas: {areas_count}")
        
        if devices_count > 0:
            return True
        else:
            print("⚠️ Export script ran but ChromaDB still appears to be empty")
            return False
        
    except Exception as e:
        print(f"❌ Error running export script: {str(e)}")
        return False

def populate_services_fallback(db):
    """Create a fallback services collection with common Home Assistant services"""
    print("\n🔄 Creating fallback services collection...")
    
    try:
        # Get the ChromaDB client
        client = db["client"]
        
        # Try to delete existing empty collection if it exists
        try:
            client.delete_collection("ha_services")
        except:
            pass
        
        # Create a new services collection
        services_collection = client.create_collection(name="ha_services")
        
        # Common Home Assistant services
        common_services = [
            {"domain": "light", "action": "turn_on", "description": "Turn on one or more lights"},
            {"domain": "light", "action": "turn_off", "description": "Turn off one or more lights"},
            {"domain": "light", "action": "toggle", "description": "Toggle one or more lights"},
            {"domain": "switch", "action": "turn_on", "description": "Turn on one or more switches"},
            {"domain": "switch", "action": "turn_off", "description": "Turn off one or more switches"},
            {"domain": "switch", "action": "toggle", "description": "Toggle one or more switches"},
            {"domain": "cover", "action": "open_cover", "description": "Open one or more covers"},
            {"domain": "cover", "action": "close_cover", "description": "Close one or more covers"},
            {"domain": "cover", "action": "stop_cover", "description": "Stop one or more covers"},
            {"domain": "media_player", "action": "turn_on", "description": "Turn on one or more media players"},
            {"domain": "media_player", "action": "turn_off", "description": "Turn off one or more media players"},
            {"domain": "media_player", "action": "play_media", "description": "Play media on one or more media players"},
            {"domain": "media_player", "action": "volume_up", "description": "Turn up volume on one or more media players"},
            {"domain": "media_player", "action": "volume_down", "description": "Turn down volume on one or more media players"},
            {"domain": "media_player", "action": "volume_set", "description": "Set volume on one or more media players"},
            {"domain": "media_player", "action": "media_play", "description": "Play media on one or more media players"},
            {"domain": "media_player", "action": "media_pause", "description": "Pause media on one or more media players"},
            {"domain": "media_player", "action": "media_stop", "description": "Stop media on one or more media players"},
            {"domain": "scene", "action": "turn_on", "description": "Activate a scene"},
            {"domain": "script", "action": "turn_on", "description": "Run a script"},
            {"domain": "automation", "action": "trigger", "description": "Trigger an automation"},
            {"domain": "climate", "action": "set_temperature", "description": "Set temperature on one or more climate devices"},
            {"domain": "climate", "action": "set_hvac_mode", "description": "Set HVAC mode on one or more climate devices"},
            {"domain": "fan", "action": "turn_on", "description": "Turn on one or more fans"},
            {"domain": "fan", "action": "turn_off", "description": "Turn off one or more fans"},
            {"domain": "fan", "action": "set_percentage", "description": "Set fan speed percentage"}
        ]
        
        # Prepare data for ChromaDB
        ids = []
        metadatas = []
        documents = []
        
        for service in common_services:
            service_id = f"{service['domain']}.{service['action']}"
            ids.append(service_id)
            metadatas.append({
                "domain": service["domain"],
                "action": service["action"]
            })
            documents.append(f"Home Assistant service action '{service['action']}' under domain '{service['domain']}': {service['description']}")
        
        # Add to ChromaDB
        services_collection.add(
            ids=ids,
            metadatas=metadatas,
            documents=documents
        )
        
        print(f"✅ Created fallback services collection with {len(ids)} common services")
        return services_collection
    except Exception as e:
        print(f"❌ Error creating fallback services: {str(e)}")
        return None

def normalize_area(area):
    return area.lower().replace(" ", "").replace("_", "")

# === Process LLM Response ===
def process_llm_response(raw_response):
    """Process the raw response from the LLM to extract the JSON intent"""
    if not raw_response:
        return None
        
    # Try to clean the response to extract just the JSON part
    # Sometimes LLMs add markdown code blocks or extra text
    cleaned_response = raw_response.strip()
    
    # Remove markdown code blocks if present
    if cleaned_response.startswith("```json") and cleaned_response.endswith("```"):
        cleaned_response = cleaned_response[7:-3].strip()
    elif cleaned_response.startswith("```") and cleaned_response.endswith("```"):
        cleaned_response = cleaned_response[3:-3].strip()
    
    # Try to find JSON object in the text
    try:
        # Look for the first { and last }
        start = cleaned_response.find("{")
        end = cleaned_response.rfind("}") + 1
        if start >= 0 and end > start:
            cleaned_response = cleaned_response[start:end]
    except:
        pass

    print("\n📦 Cleaned Response:")
    print(cleaned_response)

    try:
        intent_json = json.loads(cleaned_response)
        return intent_json
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse JSON: {str(e)}")
        return None

# === Process Voice Command ===
def process_voice_command():
    """Process a single voice command"""
    global start_time, last_checkpoint, assistant_speaking, ha_data, voice_encoder
    
    # Reset timing
    start_time = time.time()
    last_checkpoint = start_time
    
    print(f"\n🚀 Processing started at {datetime.now().strftime('%H:%M:%S')}")
    
    # Load context
    context = load_context()
    checkpoint("Loading context")
    
    # Load HA data if needed
    if ha_data is None:
        if not load_ha_data():
            print("❌ Failed to load HA data")
            speak("I'm having trouble accessing my device information. Please check the setup.")
            return
    
    # Wait for TTS to finish before recording
    waited = 0
    while assistant_speaking and waited < 10:  # 10 seconds max
        time.sleep(0.05)

    if not SKIP_RECORDING:
        print("🎤 Speak your command...")
        rec = sd.rec(int(LISTEN_TIMEOUT * 16000), samplerate=16000, channels=1, dtype='float32')
        sd.wait()
        scipy.io.wavfile.write(WAV_FILE, 16000, (rec * 32767).astype("int16"))
        checkpoint("Recording audio")
    else:
        print("⏩ Skipping recording, using existing WAV file")
        rec = scipy.io.wavfile.read(WAV_FILE)[1].astype('float32') / 32767
        rec = rec.reshape(-1, 1)
        checkpoint("Loading existing audio")
    
    # === Identify Speaker ===
    if not SKIP_SPEAKER_ID and os.path.exists(PROFILE_PATH):
        try:
            if voice_encoder is None:
                print("🔄 Initializing voice encoder...")
                voice_encoder = VoiceEncoder()
            profiles = np.load(PROFILE_PATH, allow_pickle=True).item()
            embedding = voice_encoder.embed_utterance(rec[:, 0])
            user = max(profiles, key=lambda name: np.inner(embedding, profiles[name]))
            print(f"🧠 Identified speaker: {user}")
            checkpoint("Speaker identification")
        except Exception as e:
            print(f"⚠️ Speaker identification failed: {str(e)}")
            user = DEFAULT_USER
    else:
        user = DEFAULT_USER
        print(f"⏩ Using default user: {user}")
    
    # === Transcribe via whisper.cpp ===
    cmd = [WHISPER_BIN, "-m", MODEL_PATH, "-f", WAV_FILE, "-nt"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    lines = result.stdout.strip().splitlines()
    transcript = lines[-1] if lines else ""
    print(f"📝 Transcribed: {transcript}")
    checkpoint("Transcription")
    
    if not transcript:
        print("❌ No speech detected or transcription failed")
        speak("I didn't catch that. Could you please repeat?")
        return
    
    # === Process Command ===
    # Check for correction or context commands
    is_correction_command = any(phrase in transcript.lower() for phrase in 
                              ["no", "incorrect", "wrong", "that's not right", "i meant", "i said"])
    is_context_command = any(phrase in transcript.lower() for phrase in 
                           ["again", "repeat", "retry", "try again", "one more time"])
    is_adjustment_command = any(phrase in transcript.lower() for phrase in 
                              ["more", "less", "brighter", "dimmer", "louder", "quieter", "warmer", "cooler"])
    
    # Handle retry/context commands
    if is_context_command and context['last_transcript']:
        print(f"🔄 Detected context command, reusing previous transcript: '{context['last_transcript']}'")
        enhanced_transcript = context['last_transcript']
    else:
        enhanced_transcript = transcript
    
    # Update context with current transcript
    if not is_context_command:
        context['last_transcript'] = transcript
    
    # === Generate Home Assistant Context ===
    # Find relevant entities and services
    relevant_entities = find_relevant_entities(enhanced_transcript)
    relevant_services = find_relevant_services(enhanced_transcript)
    relevant_areas = get_relevant_areas(enhanced_transcript)
    
    # Format device context
    device_context = "Available devices:\n"
    if relevant_entities:
        device_context += "\n".join([
            f"- {entity.name} ({entity.entity_id}) in {entity.area}"
            for entity in relevant_entities
        ])
    
    # Format service context
    service_context = "Available services:\n"
    for domain, services in relevant_services.items():
        service_context += f"\n{domain}:\n"
        for service_name, service_info in services.items():
            description = service_info.get("description", "")
            service_context += f"  - {service_name}: {description}\n"
    
    # Get relevant learning examples
    relevant_examples = []
    if not is_correction_command:
        relevant_examples = get_relevant_examples(enhanced_transcript)
        if relevant_examples:
            print(f"📚 Found {len(relevant_examples)} relevant examples")
    
    # Get time context
    time_context = get_time_context()
    print(f"\n🕒 Current time context: {time_context['time_of_day']} ({time_context['datetime']})")
    print(f"💡 Recommended brightness: {time_context['recommended_brightness']}%")
    
    # Get user account information
    current_user_account = user_accounts.get(user, {})
    user_account_context = json.dumps(current_user_account, indent=2)
    
    # Build context section based on previous interactions
    if context['last_devices'] and (is_adjustment_command or "it" in enhanced_transcript.lower() or "them" in enhanced_transcript.lower()):
        context_section = f"""
Previous devices controlled: {', '.join(context['last_devices'])}
Previous action: {context['last_action']}
Previous parameters: {json.dumps(context['last_parameters'], indent=2)}

If the user's command refers to "it", "them", or is otherwise ambiguous, 
ALWAYS use the last device(s) controlled: {', '.join(context['last_devices'])}.
"""
    elif context['last_result'] and is_context_command:
        context_section = f"""
Previous command: "{context['last_transcript']}"
Result: {"Success" if context['last_result'] == 'success' else "Failed"}
"""
    elif is_correction_command:
        context_section = f"""
Previous command: "{context['last_transcript']}"
Previous intent: {json.dumps(context['last_intent'], indent=2)}
This is a correction request. The user is indicating the previous response was incorrect.
"""
    else:
        context_section = ""
    
    # Add learning examples to the prompt if available
    examples_section = ""
    if relevant_examples:
        examples_section = "\nHere are some examples of similar commands and their correct intents:\n"
        for i, example in enumerate(relevant_examples):
            examples_section += f"""
Example {i+1}:
Command: "{example['command']}"
Correct intent: {json.dumps(example['corrected_intent'], indent=2)}
"""
    
    # Add time context to the prompt
    time_section = f"""
Current time information:
- Date and time: {time_context['datetime']}
- Time of day: {time_context['time_of_day']}
- Is weekend: {"Yes" if time_context['is_weekend'] else "No"}
- Recommended brightness: {time_context['recommended_brightness']}%
- Recommended color temperature: {time_context['recommended_color_temp']}K
"""
    
    # Add area context to the prompt
    area_section = f"""
Available areas in the smart home:
{device_context}

Relevant areas based on the command:
{json.dumps(relevant_areas, indent=2)}

When inferring the area from the command:
1. First check if an area is explicitly mentioned (e.g., "in the office", "office", "in office")
2. If not, use the relevant areas from the search
3. If still unclear, use the last controlled area from context
4. If no area context exists, make a reasonable inference based on:
   - Time of day (e.g., bedroom at night)
   - User's location (if known)
   - Common patterns (e.g., living room for entertainment)
   - Previous interactions

IMPORTANT: For brightness-related commands:
- If the command mentions "too bright", use turn_off for the specified area
- If the command mentions "too dark", use turn_on for the specified area
- Always include the area mentioned in the command in the inferred_room field
"""
    
    # Build a more focused prompt
    prompt = f"""You are a smart home voice assistant. Your job is to understand the user's command and return a JSON object describing the intent.

COMMAND TO PROCESS: "{enhanced_transcript}"

IMPORTANT: You must process THIS EXACT COMMAND. Do not make up or modify the command. Do not process a different command.

{context_section}

{examples_section}

{time_section}

{area_section}

{device_context}

{service_context}

Current user: {user}
User preferences: {user_account_context}

IMPORTANT RULES:
1. Process ONLY the command provided above. Do not make up or modify the command.
2. For brightness-related commands:
   - If the command mentions "too bright", use turn_off for the specified area
   - If the command mentions "too dark", use turn_on for the specified area
   - Always include the area mentioned in the command in the inferred_room field
3. If the command explicitly mentions a room/area, you MUST use that exact room name.
4. Never use generic placeholders like "room_name" - always use the actual room name mentioned.
5. When a specific room is mentioned, find devices in that room first.
6. If no room is mentioned, use the last controlled area from context.
7. If still unclear, make a reasonable inference based on time of day and user location.

FOUND AREAS:
The following areas were found in the command:
{json.dumps(relevant_areas, indent=2)}

You MUST use one of these found areas in the inferred_room field if they exist. Do not make up or use a different area.

DEVICE SELECTION RULES:
1. The entity_id in the parameters MUST match the room specified in inferred_room
2. Look at ALL available devices to find the correct entity_id
3. Match the room name from the command to the area in the device description
4. If multiple devices exist in the room, choose the most appropriate one based on the command
5. You can use any device that matches the room, even if it wasn't in the search results
6. NEVER use a device from a different room than what was specified

RESPONSE FORMAT:
You must respond with ONLY a JSON object, no other text. The JSON must follow this exact structure:
{{
  "user": "{user}",
  "inferred_room": "exact_room_name_from_command",
  "actions": [
    {{
      "device_type": "domain",
      "action": "service_name",
      "parameters": {{
        "entity_id": "entity_id"
      }}
    }}
  ]
}}

For the command "{enhanced_transcript}":
1. If it mentions "too bright", use turn_off for the specified area
2. If it mentions "too dark", use turn_on for the specified area
3. Always include the area mentioned in the command in the inferred_room field
4. Use the light domain and appropriate service based on the brightness command

DO NOT include any text before or after the JSON object. DO NOT include markdown formatting. DO NOT include explanations or comments. DO NOT make up or modify the command.
"""
    checkpoint("Building prompt")
    
    # === Send to LLM using persistent Ollama client ===
    print("\n🔎 Sending prompt to Ollama...")
    
    raw_response = ollama_client.generate(prompt)
    checkpoint("LLM response")
    
    # === Process LLM Response ===
    print("\n📦 Raw LLM Response:")
    print(raw_response)
    
    if raw_response:
        intent_json = process_llm_response(raw_response)
        
        if intent_json:
            print("\n✅ Successfully parsed JSON:")
            print(json.dumps(intent_json, indent=2))
            
            # Handle correction commands
            if is_correction_command and context['last_intent']:
                print("\n🔄 Processing correction command...")
                # TODO: Implement correction handling
                speak("I've noted your correction. I'll try to do better next time.")
            else:
                # Regular command processing - always use real mode (not test mode)
                success = send_actions_to_home_assistant(intent_json, context, test_mode=False)
                
                # Update context with the result
                context['last_intent'] = intent_json
                context['last_result'] = 'success' if success else 'failed'
                save_context(context)
                
                # Add to learning database
                add_learning_example(enhanced_transcript, intent_json)
        else:
            print("❌ Failed to parse response from LLM")
            speak("I'm sorry, I couldn't understand how to process your request.")
    else:
        print("❌ No response from LLM")
        speak("I'm having trouble connecting to my brain. Please try again later.")
    
    # === Update Context ===
    context['user'] = user
    save_context(context)
    
    # === Performance Summary ===
    total_time = time.time() - start_time
    print(f"\n⏱️ Total execution time: {total_time:.2f} seconds")
    print("\n✨ Command processing complete!")

# === Learning Database Functions ===
def get_relevant_examples(query, max_examples=3):
    """Get relevant examples from the learning database"""
    if not os.path.exists(LEARNING_PATH):
        print("⚠️ Learning database not found")
        return []
        
    try:
        # Load learning examples
        with open(LEARNING_PATH, 'r') as f:
            examples = json.load(f)
            
        if not examples:
            return []
            
        # Simple relevance scoring based on word overlap
        query_words = set(query.lower().split())
        scored_examples = []
        
        for example in examples:
            command = example.get('command', '')
            command_words = set(command.lower().split())
            
            # Calculate word overlap score
            overlap = len(query_words.intersection(command_words))
            if overlap > 0:
                scored_examples.append((overlap, example))
        
        # Sort by relevance score (descending)
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        
        # Return top examples
        return [example for _, example in scored_examples[:max_examples]]
        
    except Exception as e:
        print(f"⚠️ Error retrieving learning examples: {str(e)}")
        return []

def add_learning_example(command, intent, corrected_intent=None):
    """Add a new example to the learning database"""
    try:
        # Load existing examples or create new database
        if os.path.exists(LEARNING_PATH):
            with open(LEARNING_PATH, 'r') as f:
                examples = json.load(f)
        else:
            examples = []
            
        # Create new example
        example = {
            'command': command,
            'original_intent': intent,
            'corrected_intent': corrected_intent or intent,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Add to database
        examples.append(example)
        
        # Save database
        with open(LEARNING_PATH, 'w') as f:
            json.dump(examples, f, indent=2)
            
        print(f"✅ Added new learning example: '{command}'")
        return True
        
    except Exception as e:
        print(f"⚠️ Error adding learning example: {str(e)}")
        return False

def get_areas_from_chromadb(db):
    """Get areas from ChromaDB"""
    if not db or "areas" not in db or db["areas"] is None:
        print("⚠️ Areas collection not available")
        return None
            
    try:
        # Check if collection has data
        if db["areas"].count() == 0:
            print("⚠️ Areas collection is empty")
            return []
            
        # Query all areas
        result = db["areas"].get()
        
        if not result or not result["metadatas"] or len(result["metadatas"]) == 0:
            print("⚠️ No area data returned from ChromaDB")
            return []
        
        areas = []
        for i, metadata in enumerate(result["metadatas"]):
            if metadata:
                area_id = metadata.get("area_id", "")
                name = metadata.get("name", "")
                parent_area = metadata.get("parent_area", "")
                
                if area_id and name:
                    areas.append({
                        "area_id": area_id,
                        "name": name,
                        "parent_area": parent_area
                    })
        
        print(f"📊 Retrieved {len(areas)} areas from ChromaDB")
        return areas
    except Exception as e:
        print(f"⚠️ Error retrieving areas from ChromaDB: {str(e)}")
        return None

def get_relevant_areas(query):
    """Get relevant areas based on the command"""
    if not ha_data or 'areas' not in ha_data:
        return []
    
    # Convert command to lowercase for case-insensitive matching
    cmd_lower = query.lower()
    
    # Extract potential area names from the command
    area_patterns = [
        r"in the (\w+(?:\s+\w+)*)",  # "in the upstairs hall"
        r"the (\w+(?:\s+\w+)*) (?:light|lights|room|area)",  # "the upstairs hall light"
        r"(\w+(?:\s+\w+)*) (?:light|lights|room|area)",  # "upstairs hall light"
        r"in (\w+(?:\s+\w+)*)",  # "in office"
        r"(\w+(?:\s+\w+)*)$"  # "office" at end of sentence
    ]
    
    potential_areas = []
    for pattern in area_patterns:
        matches = re.findall(pattern, cmd_lower)
        potential_areas.extend(matches)
    
    # Also check for direct area mentions without patterns
    words = cmd_lower.split()
    for word in words:
        if len(word) > 2:  # Avoid single letters and very short words
            potential_areas.append(word)
    
    # Find matching areas
    relevant_areas = []
    for area in ha_data['areas']:
        area_name = area.get('name', '').lower()
        area_id = area.get('area_id', '').lower()
        
        # Check if area matches any potential area
        if any(potential.lower() in area_name or 
               potential.lower() in area_id or
               area_name in potential.lower() or
               area_id in potential.lower()
               for potential in potential_areas):
            relevant_areas.append(area)
    
    return relevant_areas

def load_ha_data():
    """Load Home Assistant data from file"""
    global ha_data
    try:
        # Initialize empty data structure
        ha_data = {
            "entities": [],
            "services": {},  # Dictionary to store services by domain
            "areas": set(),  # Using set to avoid duplicates
            "domains": set()  # Using set to track unique domains
        }
        
        # Read JSONL file line by line
        with open(HA_DATA_FILE, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    
                    # Handle entity data
                    if "entity_id" in data:
                        # Extract domain from entity_id
                        domain = data["entity_id"].split(".")[0] if "." in data["entity_id"] else data.get("domain", "")
                        
                        # Add to entities list
                        ha_data["entities"].append({
                            "entity_id": data["entity_id"],
                            "name": data.get("name", data["entity_id"]),
                            "area": data.get("area", ""),
                            "domain": domain
                        })
                        
                        # Track unique areas and domains
                        if data.get("area"):
                            ha_data["areas"].add(data["area"])
                        if domain:
                            ha_data["domains"].add(domain)
                    
                    # Handle service data
                    elif "action" in data:
                        domain = data["domain"]
                        action = data["action"]
                        
                        # Initialize domain in services if not exists
                        if domain not in ha_data["services"]:
                            ha_data["services"][domain] = {}
                        
                        # Add service to domain
                        ha_data["services"][domain][action] = {
                            "description": data.get("description", f"Service action {action} for domain {domain}")
                        }
                        
                        # Track domain
                        ha_data["domains"].add(domain)
                    
                    # Handle area data
                    elif "area" in data:
                        ha_data["areas"].add(data["area"])
                    
                    # Handle domain data
                    elif "domain" in data:
                        ha_data["domains"].add(data["domain"])
                        
                except json.JSONDecodeError as e:
                    print(f"⚠️ Skipping invalid JSON line: {str(e)}")
                    continue
        
        # Convert sets to sorted lists for consistent ordering
        ha_data["areas"] = sorted(list(ha_data["areas"]))
        ha_data["domains"] = sorted(list(ha_data["domains"]))
        
        print(f"✅ Loaded {len(ha_data['entities'])} entities from {HA_DATA_FILE}")
        print(f"📊 Found {len(ha_data['areas'])} areas and {len(ha_data['domains'])} domains")
        print(f"🔧 Loaded services for {len(ha_data['services'])} domains")
        return True
    except Exception as e:
        print(f"❌ Error loading HA data: {str(e)}")
        return False

def find_relevant_entities(command):
    """Find relevant entities based on the command"""
    if not ha_data or 'entities' not in ha_data:
        return []
    
    # Convert command to lowercase for case-insensitive matching
    cmd_lower = command.lower()
    
    # Extract potential room/area names
    room_patterns = [
        r"in the (\w+(?:\s+\w+)*)",  # "in the upstairs hall"
        r"the (\w+(?:\s+\w+)*) (?:light|lights|room|area)",  # "the upstairs hall light"
        r"(\w+(?:\s+\w+)*) (?:light|lights|room|area)",  # "upstairs hall light"
    ]
    
    potential_rooms = []
    for pattern in room_patterns:
        matches = re.findall(pattern, cmd_lower)
        potential_rooms.extend(matches)
    
    # Find entities that match the command
    relevant_entities = []
    for entity in ha_data['entities']:
        # Check if entity matches any potential room
        entity_area = entity.get('area', '').lower()
        entity_name = entity.get('name', '').lower()
        entity_id = entity.get('entity_id', '').lower()
        
        # Check for room matches
        room_match = any(room.lower() in entity_area or 
                        room.lower() in entity_name or 
                        room.lower() in entity_id 
                        for room in potential_rooms)
        
        # Check for direct matches in command
        direct_match = (entity_name in cmd_lower or 
                       entity_id in cmd_lower or 
                       entity_area in cmd_lower)
        
        if room_match or direct_match:
            relevant_entities.append(DeviceInfo(
                entity_id=entity['entity_id'],
                name=entity.get('name', entity['entity_id']),
                domain=entity['entity_id'].split('.')[0],
                area=entity.get('area', '')
            ))
    
    return relevant_entities

def find_relevant_services(command):
    """Find relevant services based on the command"""
    if not ha_data or 'services' not in ha_data:
        return {}
    
    # Convert command to lowercase for case-insensitive matching
    cmd_lower = command.lower()
    
    # Common action words and their corresponding services
    action_mapping = {
        'turn on': 'turn_on',
        'turn off': 'turn_off',
        'toggle': 'toggle',
        'dim': 'turn_on',
        'brighten': 'turn_on',
        'open': 'open_cover',
        'close': 'close_cover',
        'stop': 'stop_cover',
        'play': 'media_play',
        'pause': 'media_pause',
        'stop': 'media_stop',
        'volume up': 'volume_up',
        'volume down': 'volume_down',
        'set temperature': 'set_temperature',
        'set mode': 'set_hvac_mode'
    }
    
    relevant_services = {}
    for action_word, service in action_mapping.items():
        if action_word in cmd_lower:
            # Find all services that match this action
            for domain, services in ha_data['services'].items():
                if service in services:
                    if domain not in relevant_services:
                        relevant_services[domain] = {}
                    relevant_services[domain][service] = {
                        'description': services[service].get('description', '')
                    }
    
    return relevant_services

def initialize_globals():
    """Initialize global instances that should be loaded once"""
    global ha_data, voice_encoder
    
    print("\n🔄 Initializing global instances...")
    
    # Initialize HA data
    try:
        if not load_ha_data():
            print("❌ Failed to load HA data")
            return False
    except Exception as e:
        print(f"❌ Error initializing HA data: {str(e)}")
        return False
    
    # Initialize voice encoder
    try:
        if not SKIP_SPEAKER_ID:
            voice_encoder = VoiceEncoder()
            print("✅ Voice encoder initialized")
    except Exception as e:
        print(f"❌ Error initializing voice encoder: {str(e)}")
        return False
    
    return True

# === Main Function ===
def main():
    """Main function"""
    global running, wake_word_thread, ollama_client, ha_data, voice_encoder, tts_engine, shutdown_in_progress
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("\n🎙️ Voice Assistant Starting...")
    
    try:
        # Initialize global instances
        if not initialize_globals():
            print("❌ Failed to initialize global instances")
            return
        
        # Initialize Ollama client first
        print("\n🤖 Initializing Ollama client...")
        ollama_client = OllamaClient()
        if not ollama_client.initialize():
            print("⚠️ Failed to initialize Ollama. Will retry when needed.")
        
        # Start wake word detection in a separate thread
        if PORCUPINE_AVAILABLE:
            wake_word_thread = threading.Thread(target=wake_word_listener)
            wake_word_thread.daemon = True  # Make thread daemon so it exits when main thread exits
            wake_word_thread.start()
        
        # Welcome message
        speak("Voice assistant is now active. Say the wake word to begin.")
        
        # Main loop
        running = True
        while running:
            try:
                # Wait for wake word or manual trigger
                if PORCUPINE_AVAILABLE:
                    # Wait for wake word detection event
                    if wake_word_detected.wait(0.1):
                        wake_word_detected.clear()
                        process_voice_command()
                else:
                    # Manual trigger mode (for testing)
                    print("\n🔄 Press Enter to simulate wake word detection, or 'q' to quit...")
                    user_input = input()
                    
                    if user_input.lower() == 'q':
                        running = False
                        break
                    
                    process_voice_command()
                    
                    # Add a small delay to prevent CPU hogging
                    time.sleep(0.1)
            
            except KeyboardInterrupt:
                running = False
                break
            except Exception as e:
                print(f"❌ Error in main loop: {str(e)}")
                time.sleep(1)  # Prevent tight error loops
    
    except Exception as e:
        print(f"❌ Error in main: {str(e)}")
    finally:
        if not shutdown_in_progress:
            cleanup_resources()
        print("👋 Voice assistant stopped.")

if __name__ == "__main__":
    # Initialize TTS engine if pyttsx3 is available
    if PYTTSX3_AVAILABLE:
        import pyttsx3
        tts_engine = pyttsx3.init()
        tts_engine.setProperty('rate', VOICE_RATE)
        tts_engine.setProperty('volume', VOICE_VOLUME)
        if VOICE_ID:
            tts_engine.setProperty('voice', VOICE_ID)
    main()
