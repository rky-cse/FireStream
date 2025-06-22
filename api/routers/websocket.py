# api/routers/websocket.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from typing import Dict, List, Set
import json
import logging
from datetime import datetime
import uuid
import asyncio
import time

# Import your chat pipeline
from ingestion.chat.chat_pipeline import store_message, get_recent_messages, generate_summary

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

class ConnectionManager:
    def __init__(self):
        # Store active connections per content_id
        self.active_connections: Dict[str, Dict[str, WebSocket]] = {}
        # Store user info per content_id
        self.room_users: Dict[str, Dict[str, dict]] = {}
        # Track last summary time per room
        self.last_summary_time: Dict[str, float] = {}
        # Summary interval in seconds
        self.summary_interval = 30
    
    async def connect(self, websocket: WebSocket, content_id: str, user_id: str):
        await websocket.accept()
        
        # Initialize room if doesn't exist
        if content_id not in self.active_connections:
            self.active_connections[content_id] = {}
            self.room_users[content_id] = {}
            self.last_summary_time[content_id] = time.time()
        
        # Add connection
        self.active_connections[content_id][user_id] = websocket
        self.room_users[content_id][user_id] = {
            "user_id": user_id,
            "username": f"User{user_id[-4:]}",  # Simple username
            "joined_at": datetime.now().isoformat()
        }
        
        logger.info(f"User {user_id} connected to room {content_id}")
        
        # Notify others that user joined
        await self.broadcast_to_room(content_id, {
            "type": "user_joined",
            "user_id": user_id,
            "username": self.room_users[content_id][user_id]["username"],
            "timestamp": datetime.now().isoformat()
        }, exclude_user=user_id)
        
        # Send current online users to the new user
        await self.send_online_users(content_id, user_id)
        
        # Send latest summary to new user
        await self.send_chat_summary(content_id, user_id)
    
    def disconnect(self, content_id: str, user_id: str):
        if content_id in self.active_connections and user_id in self.active_connections[content_id]:
            del self.active_connections[content_id][user_id]
            
            # Get username before removing from room_users
            username = None
            if content_id in self.room_users and user_id in self.room_users[content_id]:
                username = self.room_users[content_id][user_id]["username"]
                del self.room_users[content_id][user_id]
            
            logger.info(f"User {user_id} disconnected from room {content_id}")
            
            # Clean up empty rooms
            if not self.active_connections[content_id]:
                del self.active_connections[content_id]
                if content_id in self.room_users:
                    del self.room_users[content_id]
                if content_id in self.last_summary_time:
                    del self.last_summary_time[content_id]
            else:
                # Notify others that user left
                return username
        return None
    
    async def send_personal_message(self, message: dict, content_id: str, user_id: str):
        if (content_id in self.active_connections and 
            user_id in self.active_connections[content_id]):
            websocket = self.active_connections[content_id][user_id]
            try:
                await websocket.send_text(json.dumps(message))
            except:
                # Connection is broken, remove it
                self.disconnect(content_id, user_id)
    
    async def broadcast_to_room(self, content_id: str, message: dict, exclude_user: str = None):
        if content_id not in self.active_connections:
            return
        
        disconnected_users = []
        for user_id, websocket in self.active_connections[content_id].items():
            if exclude_user and user_id == exclude_user:
                continue
            
            try:
                await websocket.send_text(json.dumps(message))
            except:
                # Connection is broken, mark for removal
                disconnected_users.append(user_id)
        
        # Clean up disconnected users
        for user_id in disconnected_users:
            self.disconnect(content_id, user_id)
    
    async def send_online_users(self, content_id: str, user_id: str = None):
        if content_id not in self.room_users:
            return
        
        users_list = list(self.room_users[content_id].values())
        message = {
            "type": "online_users",
            "users": users_list
        }
        
        if user_id:
            await self.send_personal_message(message, content_id, user_id)
        else:
            await self.broadcast_to_room(content_id, message)
    
    async def send_chat_summary(self, content_id: str, user_id: str = None):
        try:
            # Get recent messages from Redis
            recent_messages = get_recent_messages()
            
            if recent_messages:
                # Generate summary using your pipeline
                summary = await asyncio.get_event_loop().run_in_executor(
                    None, generate_summary, recent_messages
                )
                
                message = {
                    "type": "chat_summary",
                    "summary": summary,
                    "message_count": len(recent_messages),
                    "timestamp": datetime.now().isoformat()
                }
                
                if user_id:
                    await self.send_personal_message(message, content_id, user_id)
                else:
                    await self.broadcast_to_room(content_id, message)
        except Exception as e:
            logger.error(f"Error generating chat summary: {e}")
    
    async def process_message_and_check_summary(self, content_id: str, message: str):
        """Process message through pipeline and check if summary update is needed"""
        try:
            # Store message using your pipeline (includes spam filtering)
            await asyncio.get_event_loop().run_in_executor(
                None, store_message, message, time.time()
            )
            
            # Check if it's time for a summary update
            current_time = time.time()
            if (content_id not in self.last_summary_time or 
                current_time - self.last_summary_time[content_id] >= self.summary_interval):
                
                self.last_summary_time[content_id] = current_time
                await self.send_chat_summary(content_id)
                
        except Exception as e:
            logger.error(f"Error processing message for summary: {e}")

# Global connection manager
manager = ConnectionManager()

@router.websocket("/ws/chat/{content_id}/{user_id}")
async def websocket_endpoint(websocket: WebSocket, content_id: str, user_id: str):
    await manager.connect(websocket, content_id, user_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Handle different message types
            if message_data.get("type") == "message":
                message_text = message_data.get("message", "")
                
                # Create message object
                chat_message = {
                    "type": "message",
                    "id": str(uuid.uuid4()),
                    "user_id": user_id,
                    "username": manager.room_users[content_id][user_id]["username"],
                    "message": message_text,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Broadcast message to all users in the room
                await manager.broadcast_to_room(content_id, chat_message)
                
                # Process message through pipeline for summary generation
                await manager.process_message_and_check_summary(content_id, message_text)
                
            elif message_data.get("type") == "get_online_users":
                # Send online users list
                await manager.send_online_users(content_id, user_id)
            
            elif message_data.get("type") == "get_summary":
                # Manual summary request
                await manager.send_chat_summary(content_id, user_id)
            
            else:
                logger.warning(f"Unknown message type: {message_data.get('type')}")
                
    except WebSocketDisconnect:
        username = manager.disconnect(content_id, user_id)
        if username:
            # Notify others that user left
            await manager.broadcast_to_room(content_id, {
                "type": "user_left",
                "user_id": user_id,
                "username": username,
                "timestamp": datetime.now().isoformat()
            })
            
            # Update online users for remaining users
            await manager.send_online_users(content_id)
    
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id} in room {content_id}: {str(e)}")
        username = manager.disconnect(content_id, user_id)
        if username:
            await manager.broadcast_to_room(content_id, {
                "type": "user_left",
                "user_id": user_id,
                "username": username,
                "timestamp": datetime.now().isoformat()
            })

# Optional: Add endpoint to get room statistics
@router.get("/chat/stats/{content_id}")
async def get_room_stats(content_id: str):
    if content_id not in manager.room_users:
        return {
            "content_id": content_id,
            "online_users": 0,
            "users": []
        }
    
    return {
        "content_id": content_id,
        "online_users": len(manager.room_users[content_id]),
        "users": list(manager.room_users[content_id].values())
    }

# Manual summary endpoint
@router.post("/chat/summary/{content_id}")
async def trigger_summary(content_id: str):
    """Manually trigger a summary update for a room"""
    await manager.send_chat_summary(content_id)
    return {"message": "Summary update triggered"}