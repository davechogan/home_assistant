# Home Assistant Voice LLM - System Flow Diagram

## High-Level System Architecture

```mermaid
graph TB
    subgraph Frontend
        UI[React Frontend]
        VoiceUI[Voice Interface]
    end

    subgraph Backend
        API[FastAPI Backend]
        DB[(PostgreSQL)]
        Chroma[(ChromaDB)]
    end

    subgraph Voice Pipeline
        Wake[Wake Word Detection]
        STT[Speech-to-Text]
        TTS[Text-to-Speech]
        SpeakerID[Speaker Identification]
    end

    subgraph Home Assistant
        HA[Home Assistant API]
        Devices[Smart Devices]
    end

    %% Frontend connections
    UI --> API
    VoiceUI --> Wake
    Wake --> STT
    STT --> API
    API --> TTS
    TTS --> VoiceUI

    %% Backend connections
    API --> DB
    API --> Chroma
    API --> HA
    HA --> Devices

    %% Voice Pipeline connections
    SpeakerID --> API
```

## Voice Command Processing Flow

```mermaid
sequenceDiagram
    participant User
    participant Wake as Wake Word Detection
    participant STT as Speech-to-Text
    participant API as Backend API
    participant LLM as Language Model
    participant HA as Home Assistant
    participant TTS as Text-to-Speech
    participant Training as Training Manager

    User->>Wake: Speaks wake word
    Wake->>STT: Activates STT
    User->>STT: Speaks command
    STT->>API: Sends transcribed text
    API->>LLM: Processes command
    LLM->>HA: Sends device control
    HA-->>API: Confirms action
    API->>TTS: Generates response
    TTS-->>User: Speaks response
    API->>Training: Logs interaction
    Training->>TTS: Asks for feedback
    TTS-->>User: "Was this the correct command?"
    User->>STT: Responds yes/no
    STT->>Training: Sends feedback
    Training->>Training: Updates training data
    Training->>LLM: Periodically fine-tunes model
```

## Training Flow

```mermaid
graph TD
    subgraph Training Pipeline
        Log[Log Interaction]
        Feedback[Collect Feedback]
        Store[Store Training Data]
        Train[Fine-tune Model]
        Deploy[Deploy Updated Model]
    end

    subgraph Training Data
        Raw[Raw Interactions]
        Labeled[Labeled Data]
        Model[LoRA Weights]
    end

    subgraph Configuration
        Settings[Training Settings]
        Threshold[Feedback Threshold]
        Schedule[Training Schedule]
    end

    Log --> Raw
    Feedback --> Labeled
    Raw --> Labeled
    Labeled --> Train
    Train --> Model
    Model --> Deploy

    Settings --> Log
    Settings --> Feedback
    Settings --> Train
    Threshold --> Feedback
    Schedule --> Train
```

## Data Flow and Storage

```mermaid
graph LR
    subgraph Input
        Voice[Voice Input]
        Text[Text Input]
    end

    subgraph Processing
        STT[Speech-to-Text]
        LLM[Language Model]
        TTS[Text-to-Speech]
    end

    subgraph Storage
        DB[(PostgreSQL)]
        Chroma[(ChromaDB)]
    end

    subgraph Output
        VoiceOut[Voice Response]
        Action[Device Action]
    end

    Voice --> STT
    Text --> LLM
    STT --> LLM
    LLM --> TTS
    LLM --> Action
    TTS --> VoiceOut

    LLM --> DB
    LLM --> Chroma
    DB --> LLM
    Chroma --> LLM
```

## Component Dependencies

```mermaid
graph TD
    subgraph Core Services
        API[FastAPI Backend]
        Voice[Voice Pipeline]
        Frontend[React Frontend]
    end

    subgraph Dependencies
        Postgres[PostgreSQL]
        Chroma[ChromaDB]
        Whisper[Whisper STT]
        Coqui[Coqui TTS]
        HA[Home Assistant]
    end

    API --> Postgres
    API --> Chroma
    API --> HA
    Voice --> Whisper
    Voice --> Coqui
    Frontend --> API
    Frontend --> Voice
```

## Notes
- The system uses a modular architecture allowing for easy updates and maintenance
- Voice pipeline components can be swapped out (e.g., different STT/TTS engines)
- ChromaDB stores conversation context and embeddings
- PostgreSQL stores user data, preferences, and interaction history
- Home Assistant integration enables control of smart home devices
- The frontend provides both voice and traditional UI interfaces
- Training feedback loop enables continuous model improvement
- LoRA fine-tuning allows efficient model updates without full retraining
- Training settings can be configured for:
  - Feedback collection frequency
  - Training schedule
  - Minimum feedback threshold
  - Model update criteria 