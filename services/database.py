# services/database.py
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class MongoStore:
    def __init__(self, uri: str, db_name: str):
        self.client = AsyncIOMotorClient(uri)
        self.db = self.client[db_name]

        self.uploads = self.db["uploads"]
        self.sessions = self.db["chat_sessions"]
        self.messages = self.db["chat_messages"]
        logger.info(f"[mongo] connected to {uri}/{db_name}")

    async def ensure_indexes(self):
        """Create indexes for common queries."""
        await self.uploads.create_index("document_id", unique=True)
        await self.uploads.create_index("uploaded_at")
        await self.sessions.create_index("session_id", unique=True)
        await self.sessions.create_index("created_at")
        await self.messages.create_index("session_id")
        await self.messages.create_index("created_at")

    # -------- Uploads --------

    async def save_upload(
        self,
        document_id: str,
        filename: str,
        pages_processed: int,
        size_bytes: int,
    ) -> str:
        doc = {
            "document_id": document_id,
            "filename": filename,
            "pages_processed": pages_processed,
            "size_bytes": size_bytes,
            "uploaded_at": datetime.utcnow(),
        }
        await self.uploads.insert_one(doc)
        return document_id

    async def list_uploads(self) -> List[Dict[str, Any]]:
        cursor = self.uploads.find().sort("uploaded_at", -1)
        out = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            out.append(doc)
        return out

    # -------- Chat sessions --------

    async def create_session(self, session_id: str) -> str:
        await self.sessions.update_one(
            {"session_id": session_id},
            {"$setOnInsert": {
                "session_id": session_id,
                "created_at": datetime.utcnow(),
            }},
            upsert=True,
        )
        return session_id

    async def list_sessions(self) -> List[Dict[str, Any]]:
        cursor = self.sessions.find().sort("created_at", -1)
        out = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            out.append(doc)
        return out

    async def delete_session(self, session_id: str):
        await self.sessions.delete_one({"session_id": session_id})
        await self.messages.delete_many({"session_id": session_id})

    # -------- Messages --------

    async def save_message(
        self,
        session_id: str,
        role: str,  # "user" | "assistant"
        content: str,
        citations: Optional[List[int]] = None,
        sources: Optional[List[Dict[str, Any]]] = None,
    ):
        await self.messages.insert_one({
            "session_id": session_id,
            "role": role,
            "content": content,
            "citations": citations or [],
            "sources": sources or [],
            "created_at": datetime.utcnow(),
        })

    async def get_history(
        self, session_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        cursor = (
            self.messages
            .find({"session_id": session_id})
            .sort("created_at", 1)
            .limit(limit)
        )
        out = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            out.append(doc)
        return out

    async def clear_history(self, session_id: str) -> int:
        result = await self.messages.delete_many({"session_id": session_id})
        return result.deleted_count