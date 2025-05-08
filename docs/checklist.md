# Home Assistant Voice LLM – Development Checklist

## 1. Project Setup
- [x] Clone the repository and set up the directory structure.
- [x] Create a virtual environment for Python dependencies.
- [x] Install core dependencies (FastAPI, uvicorn, etc.).
- [x] Set up Docker and docker-compose for orchestration.
- [x] Set up the database (PostgreSQL) and ORM (SQLAlchemy).
  - [x] Define PostgreSQL service in docker-compose
  - [x] Create database models
  - [x] Set up SQLAlchemy models
  - [x] Create test data and validation scripts
  - [ ] Configure production database connections
- [ ] Set up shared configuration
  - [x] Create basic directory structure
  - [ ] Create environment variables template
  - [ ] Set up configuration management
  - [ ] Implement cross-service configuration sharing

## 2. Backend Development
- [x] Scaffold the FastAPI app (`backend/app/main.py`).
- [x] Implement the health check endpoint (`/health`).
- [x] Create user routes (`backend/app/api/routes_user.py`).
- [ ] Build user service and models.
- [ ] Integrate with Home Assistant API.
- [x] Set up database models for:
  - [x] User profiles
  - [x] Voice interactions
  - [x] Training data
  - [x] Model training logs
- [ ] Implement context/memory management.
- [ ] Add feedback and learning mechanisms.
- [ ] Write unit and integration tests.

## 3. Voice Pipeline Development
- [x] Scaffold the voice pipeline entry point (`voice/app/main.py`).
- [x] Implement wake word detection.
- [x] Integrate speech-to-text (STT) using Whisper.
- [x] Integrate basic text-to-speech (TTS) using macOS say command
- [ ] Enhance TTS with Coqui/ElevenLabs for better voice quality
- [ ] Implement speaker identification.
- [x] Set up audio I/O and buffering.
- [x] Add feedback collection structure
  - [x] Database schema for feedback
  - [x] Mock data generation
  - [ ] Real-time feedback collection
- [ ] Write unit and integration tests.

## 4. Frontend Development
- [x] Scaffold the React app (`frontend/`).
- [ ] Create core components (VoiceInput, DeviceControl, FeedbackPanel, UserProfile).
- [ ] Build pages (Dashboard, Settings, History).
- [ ] Integrate with backend API.
- [ ] Implement real-time updates and feedback UI.
- [ ] Write unit and integration tests.

## 5. Shared Module Development
- [x] Set up shared configuration.
- [ ] Define shared types and interfaces.
- [ ] Create shared configuration and utilities.
- [ ] Ensure cross-module consistency.

## 6. Integration and Testing
- [ ] Integrate backend, voice pipeline, and frontend.
- [ ] Test end-to-end flows (wake word → STT → LLM → TTS → action).
- [ ] Validate performance and reliability.
- [ ] Conduct user acceptance testing.

## 7. Documentation and Deployment
- [x] Update README and architecture docs.
- [ ] Write API documentation.
- [x] Prepare deployment scripts and Docker configurations.
- [ ] Deploy to staging/production.

## 8. Post-Launch
- [ ] Monitor performance and user feedback.
- [ ] Iterate on features based on user input.
- [ ] Plan for future enhancements (multi-language, mobile app, etc.). 