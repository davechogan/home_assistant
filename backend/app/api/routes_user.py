from fastapi import APIRouter

router = APIRouter(prefix="/users", tags=["users"])

@router.get("/{user_id}")
async def get_user(user_id: int):
    # Placeholder: replace with actual user retrieval logic
    return {"user_id": user_id, "name": "Test User"} 