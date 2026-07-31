from fastapi import APIRouter

from app.api.v1 import auth, conversation, document, health, knowledge_base

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(knowledge_base.router)
api_router.include_router(document.router)
api_router.include_router(conversation.router)