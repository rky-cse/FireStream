# api/routers/websocket.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from typing import Dict, List, Set
import json
import logging
from datetime import datetime
import uuid

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
    
    async def connect(self, websocket: WebSocket, content_id: str, user_id: str):
        await websocket.accept()
        
        # Initialize room if doesn't exist
        if content_id not in self.active_connections:
            self.active_connections[content_id] = {}
            self.room_users[content_id] = {}
        
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
                # Create message object
                chat_message = {
                    "type": "message",
                    "id": str(uuid.uuid4()),
                    "user_id": user_id,
                    "username": manager.room_users[content_id][user_id]["username"],
                    "message": message_data.get("message", ""),
                    "timestamp": datetime.now().isoformat()
                }
                
                # Broadcast message to all users in the room
                await manager.broadcast_to_room(content_id, chat_message)
                
            elif message_data.get("type") == "get_online_users":
                # Send online users list
                await manager.send_online_users(content_id, user_id)
            
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