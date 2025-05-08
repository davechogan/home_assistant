from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes_user import router as user_router

app = FastAPI(title="Home Assistant Voice LLM Backend")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include user routes
app.include_router(user_router)

@app.get("/health")
async def health_check():
    return {"status": "ok"}
