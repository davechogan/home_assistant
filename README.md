# HomeAssistant Voice LLM Project

A modular, context-aware voice assistant for smart home control, leveraging LLMs, natural voice, and user personalization.

## Project Overview
This project aims to build a full-stack, extensible smart home voice assistant platform. It integrates:
- **Backend:** Handles API, LLM orchestration, user logic, and database management.
- **Voice Pipeline:** Wake word detection, speech-to-text (STT), text-to-speech (TTS), and speaker identification.
- **Frontend:** A React-based UI for user feedback, corrections, and monitoring.
- **Shared:** Common types, configuration, and utilities for cross-module consistency.

## Tech Stack
- **Backend:** Python (FastAPI), PostgreSQL, ChromaDB, LangChain, Home Assistant API
- **Voice:** Wake word detection (Porcupine), STT (Whisper), TTS (Coqui/ElevenLabs), Speaker ID (Resemblyzer)
- **Frontend:** React (Create React App)
- **Vector DB:** ChromaDB

## Directory Structure
```
backend/      # API, LLM, Home Assistant integration, user logic, feedback, DB
voice/        # Wake word, STT, TTS, speaker ID, audio utils
frontend/     # React UI for feedback, corrections, monitoring
shared/       # Shared config, types, and utilities
```

## Frontend (React) Initialization
The `frontend/` directory contains a React app scaffolded with Create React App. This UI will:
- Allow users to provide feedback and corrections to the voice assistant
- Monitor system status and interactions
- Serve as the main interface for user interaction and configuration

**To get started with the frontend:**
1. `cd frontend`
2. `npm install` (if dependencies are not yet installed)
3. `npm start` to launch the development server at [http://localhost:3000](http://localhost:3000)

## Getting Started
- See each module's README for setup instructions.
- Use `docker-compose.yml` to orchestrate services locally (to be configured).

---

*This README documents the project structure, purpose, and how the new React frontend fits into the overall architecture. For more details, see the README in each module.*
