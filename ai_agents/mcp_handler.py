"""
Model Context Protocol (MCP) Handler - Enhanced inter-agent communication
Provides structured, context-rich communication between AI agents
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
import redis
import uuid

logger = logging.getLogger(__name__)

class MCPMessageType(Enum):
    """Types of MCP messages"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    HEARTBEAT = "heartbeat"
    CAPABILITY_ANNOUNCEMENT = "capability_announcement"
    ERROR = "error"
    SHUTDOWN = "shutdown"

class MCPPriority(Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5

@dataclass
class MCPCapability:
    """Represents an agent capability"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    version: str = "1.0"

@dataclass
class MCPMessage:
    """Standard MCP message structure"""
    message_id: str
    protocol_version: str
    sender_agent: str
    receiver_agent: str
    message_type: MCPMessageType
    timestamp: str
    priority: MCPPriority
    correlation_id: Optional[str]
    context: Dict[str, Any]
    payload: Dict[str, Any]
    capabilities: Optional[List[str]] = None
    ttl: Optional[int] = None  # Time to live in seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization"""
        data = asdict(self)
        data["message_type"] = self.message_type.value
        data["priority"] = self.priority.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPMessage':
        """Create message from dictionary"""
        data["message_type"] = MCPMessageType(data["message_type"])
        data["priority"] = MCPPriority(data["priority"])
        return cls(**data)

class MCPHandler:
    """
    Model Context Protocol handler for enhanced inter-agent communication
    Provides structured, context-rich messaging with capability discovery
    """
    
    def __init__(
        self, 
        agent_id: str,
        redis_client: Optional[redis.Redis] = None,
        protocol_version: str = "1.0"
    ):
        self.agent_id = agent_id
        self.redis_client = redis_client
        self.protocol_version = protocol_version
        
        # Agent capabilities registry
        self.capabilities: Dict[str, MCPCapability] = {}
        self.discovered_agents: Dict[str, Dict[str, Any]] = {}
        
        # Message handling
        self.message_handlers: Dict[MCPMessageType, Callable] = {}
        self.response_callbacks: Dict[str, Callable] = {}
        self.message_history: List[MCPMessage] = []
        
        # Communication channels
        self.channels = {
            "broadcast": "mcp:broadcast",
            "agent_specific": f"mcp:agent:{agent_id}",
            "capability_discovery": "mcp:capabilities",
            "heartbeat": "mcp:heartbeat"
        }
        
        # State management
        self.is_active = False
        self.last_heartbeat = None
        self.message_stats = {
            "sent": 0,
            "received": 0,
            "errors": 0,
            "timeouts": 0
        }
        
        # Initialize default handlers
        self._initialize_default_handlers()
        
        logger.info(f"🔄 MCP Handler initialized for agent: {agent_id}")
    
    def _initialize_default_handlers(self):
        """Initialize default message handlers"""
        self.message_handlers[MCPMessageType.HEARTBEAT] = self._handle_heartbeat
        self.message_handlers[MCPMessageType.CAPABILITY_ANNOUNCEMENT] = self._handle_capability_announcement
        self.message_handlers[MCPMessageType.REQUEST] = self._handle_request
        self.message_handlers[MCPMessageType.RESPONSE] = self._handle_response
        self.message_handlers[MCPMessageType.ERROR] = self._handle_error
        self.message_handlers[MCPMessageType.SHUTDOWN] = self._handle_shutdown
    
    async def register_capability(
        self, 
        name: str, 
        description: str,
        input_schema: Dict[str, Any],
        output_schema: Dict[str, Any],
        handler: Callable
    ) -> bool:
        """Register a new capability for this agent"""
        try:
            capability = MCPCapability(
                name=name,
                description=description,
                input_schema=input_schema,
                output_schema=output_schema
            )
            
            self.capabilities[name] = capability
            
            # Register handler for this capability
            self.message_handlers[name] = handler
            
            logger.info(f"✅ Capability registered: {name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to register capability {name}: {e}")
            return False
    
    async def start(self) -> bool:
        """Start the MCP handler and begin listening for messages"""
        try:
            if not self.redis_client:
                logger.warning("⚠️ Redis client not available - MCP handler running in limited mode")
                return False
            
            self.is_active = True
            
            # Announce capabilities
            await self._announce_capabilities()
            
            # Start heartbeat
            asyncio.create_task(self._heartbeat_loop())
            
            # Start message listener
            asyncio.create_task(self._message_listener())
            
            logger.info(f"🚀 MCP Handler started for agent: {self.agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start MCP handler: {e}")
            return False
    
    async def stop(self):
        """Stop the MCP handler gracefully"""
        try:
            self.is_active = False
            
            # Send shutdown notification
            await self._send_shutdown_notification()
            
            logger.info(f"🛑 MCP Handler stopped for agent: {self.agent_id}")
            
        except Exception as e:
            logger.error(f"❌ Error stopping MCP handler: {e}")
    
    async def send_message(
        self,
        receiver_agent: str,
        message_type: MCPMessageType,
        payload: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        priority: MCPPriority = MCPPriority.NORMAL,
        correlation_id: Optional[str] = None,
        ttl: Optional[int] = None
    ) -> str:
        """Send an MCP message to another agent"""
        
        message_id = str(uuid.uuid4())
        
        try:
            message = MCPMessage(
                message_id=message_id,
                protocol_version=self.protocol_version,
                sender_agent=self.agent_id,
                receiver_agent=receiver_agent,
                message_type=message_type,
                timestamp=datetime.now().isoformat(),
                priority=priority,
                correlation_id=correlation_id,
                context=context or {},
                payload=payload,
                ttl=ttl
            )
            
            # Determine target channel
            if receiver_agent == "broadcast":
                channel = self.channels["broadcast"]
            else:
                channel = f"mcp:agent:{receiver_agent}"
            
            # Send message via Redis
            if self.redis_client:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.redis_client.lpush,
                    channel,
                    json.dumps(message.to_dict())
                )
                
                # Store in message history
                self.message_history.append(message)
                self.message_stats["sent"] += 1
                
                logger.info(f"📤 MCP message sent: {message_type.value} to {receiver_agent}")
            
            return message_id
            
        except Exception as e:
            self.message_stats["errors"] += 1
            logger.error(f"❌ Failed to send MCP message: {e}")
            raise
    
    async def send_request(
        self,
        receiver_agent: str,
        capability: str,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """Send a request and wait for response"""
        
        correlation_id = str(uuid.uuid4())
        
        try:
            # Send request
            await self.send_message(
                receiver_agent=receiver_agent,
                message_type=MCPMessageType.REQUEST,
                payload={
                    "capability": capability,
                    "parameters": parameters
                },
                context=context,
                correlation_id=correlation_id,
                ttl=timeout
            )
            
            # Wait for response
            response = await self._wait_for_response(correlation_id, timeout)
            return response
            
        except asyncio.TimeoutError:
            self.message_stats["timeouts"] += 1
            logger.error(f"⏰ Request timeout: {capability} to {receiver_agent}")
            raise
        except Exception as e:
            logger.error(f"❌ Request failed: {capability} to {receiver_agent} - {e}")
            raise
    
    async def send_response(
        self,
        receiver_agent: str,
        correlation_id: str,
        result: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ):
        """Send a response to a previous request"""
        
        await self.send_message(
            receiver_agent=receiver_agent,
            message_type=MCPMessageType.RESPONSE,
            payload={"result": result},
            context=context,
            correlation_id=correlation_id
        )
    
    async def send_notification(
        self,
        receiver_agent: str,
        event_type: str,
        data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        priority: MCPPriority = MCPPriority.NORMAL
    ):
        """Send a notification (no response expected)"""
        
        await self.send_message(
            receiver_agent=receiver_agent,
            message_type=MCPMessageType.NOTIFICATION,
            payload={
                "event_type": event_type,
                "data": data
            },
            context=context,
            priority=priority
        )
    
    async def broadcast_message(
        self,
        message_type: MCPMessageType,
        payload: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        priority: MCPPriority = MCPPriority.NORMAL
    ):
        """Broadcast a message to all agents"""
        
        await self.send_message(
            receiver_agent="broadcast",
            message_type=message_type,
            payload=payload,
            context=context,
            priority=priority
        )
    
    async def _announce_capabilities(self):
        """Announce this agent's capabilities to other agents"""
        
        capabilities_data = {
            "agent_id": self.agent_id,
            "capabilities": {name: asdict(cap) for name, cap in self.capabilities.items()},
            "protocol_version": self.protocol_version,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.broadcast_message(
            message_type=MCPMessageType.CAPABILITY_ANNOUNCEMENT,
            payload=capabilities_data,
            priority=MCPPriority.HIGH
        )
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeat messages"""
        
        while self.is_active:
            try:
                await self.send_message(
                    receiver_agent="broadcast",
                    message_type=MCPMessageType.HEARTBEAT,
                    payload={
                        "agent_id": self.agent_id,
                        "status": "active",
                        "capabilities_count": len(self.capabilities),
                        "message_stats": self.message_stats
                    },
                    priority=MCPPriority.LOW
                )
                
                self.last_heartbeat = datetime.now()
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
                
            except Exception as e:
                logger.error(f"❌ Heartbeat failed: {e}")
                await asyncio.sleep(30)
    
    async def _message_listener(self):
        """Listen for incoming MCP messages"""
        
        if not self.redis_client:
            return
        
        while self.is_active:
            try:
                # Listen on agent-specific channel
                channel = self.channels["agent_specific"]
                
                # Non-blocking pop with timeout
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.redis_client.brpop,
                    [channel, self.channels["broadcast"]],
                    1  # 1 second timeout
                )
                
                if result:
                    channel_name, message_data = result
                    message_dict = json.loads(message_data.decode('utf-8'))
                    message = MCPMessage.from_dict(message_dict)
                    
                    # Process the message
                    await self._process_message(message)
                    
            except Exception as e:
                logger.error(f"❌ Message listener error: {e}")
                await asyncio.sleep(1)
    
    async def _process_message(self, message: MCPMessage):
        """Process an incoming MCP message"""
        
        try:
            # Skip messages from self
            if message.sender_agent == self.agent_id:
                return
            
            # Check TTL
            if message.ttl:
                message_time = datetime.fromisoformat(message.timestamp)
                if datetime.now() - message_time > timedelta(seconds=message.ttl):
                    logger.warning(f"⏰ Message expired: {message.message_id}")
                    return
            
            # Update stats
            self.message_stats["received"] += 1
            
            # Store in history
            self.message_history.append(message)
            
            # Route to appropriate handler
            handler = self.message_handlers.get(message.message_type)
            if handler:
                await handler(message)
            else:
                logger.warning(f"⚠️ No handler for message type: {message.message_type}")
            
        except Exception as e:
            logger.error(f"❌ Failed to process message {message.message_id}: {e}")
            
            # Send error response if it was a request
            if message.message_type == MCPMessageType.REQUEST and message.correlation_id:
                await self.send_message(
                    receiver_agent=message.sender_agent,
                    message_type=MCPMessageType.ERROR,
                    payload={"error": str(e)},
                    correlation_id=message.correlation_id
                )
    
    async def _handle_heartbeat(self, message: MCPMessage):
        """Handle heartbeat messages"""
        agent_id = message.payload.get("agent_id")
        if agent_id and agent_id != self.agent_id:
            self.discovered_agents[agent_id] = {
                "last_heartbeat": message.timestamp,
                "status": message.payload.get("status", "unknown"),
                "capabilities_count": message.payload.get("capabilities_count", 0),
                "message_stats": message.payload.get("message_stats", {})
            }
    
    async def _handle_capability_announcement(self, message: MCPMessage):
        """Handle capability announcements from other agents"""
        agent_id = message.payload.get("agent_id")
        capabilities = message.payload.get("capabilities", {})
        
        if agent_id and agent_id != self.agent_id:
            self.discovered_agents[agent_id] = {
                "capabilities": capabilities,
                "protocol_version": message.payload.get("protocol_version"),
                "last_announcement": message.timestamp
            }
            
            logger.info(f"📢 Capabilities discovered for agent {agent_id}: {len(capabilities)} capabilities")
    
    async def _handle_request(self, message: MCPMessage):
        """Handle incoming requests"""
        capability = message.payload.get("capability")
        parameters = message.payload.get("parameters", {})
        
        if capability in self.message_handlers:
            try:
                # Execute the capability handler
                handler = self.message_handlers[capability]
                result = await handler(parameters, message.context)
                
                # Send response
                await self.send_response(
                    receiver_agent=message.sender_agent,
                    correlation_id=message.correlation_id,
                    result=result,
                    context=message.context
                )
                
            except Exception as e:
                # Send error response
                await self.send_message(
                    receiver_agent=message.sender_agent,
                    message_type=MCPMessageType.ERROR,
                    payload={"error": str(e)},
                    correlation_id=message.correlation_id
                )
        else:
            # Unknown capability
            await self.send_message(
                receiver_agent=message.sender_agent,
                message_type=MCPMessageType.ERROR,
                payload={"error": f"Unknown capability: {capability}"},
                correlation_id=message.correlation_id
            )
    
    async def _handle_response(self, message: MCPMessage):
        """Handle response messages"""
        if message.correlation_id in self.response_callbacks:
            callback = self.response_callbacks[message.correlation_id]
            await callback(message.payload.get("result", {}))
            del self.response_callbacks[message.correlation_id]
    
    async def _handle_error(self, message: MCPMessage):
        """Handle error messages"""
        error = message.payload.get("error", "Unknown error")
        logger.error(f"❌ Received error from {message.sender_agent}: {error}")
        
        if message.correlation_id in self.response_callbacks:
            callback = self.response_callbacks[message.correlation_id]
            await callback({"error": error})
            del self.response_callbacks[message.correlation_id]
    
    async def _handle_shutdown(self, message: MCPMessage):
        """Handle shutdown notifications"""
        agent_id = message.sender_agent
        logger.info(f"🛑 Agent shutdown notification received: {agent_id}")
        
        # Remove from discovered agents
        if agent_id in self.discovered_agents:
            del self.discovered_agents[agent_id]
    
    async def _wait_for_response(self, correlation_id: str, timeout: int) -> Dict[str, Any]:
        """Wait for a response with the given correlation ID"""
        
        response_future = asyncio.Future()
        
        async def response_callback(result):
            if not response_future.done():
                response_future.set_result(result)
        
        self.response_callbacks[correlation_id] = response_callback
        
        try:
            result = await asyncio.wait_for(response_future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            # Clean up callback
            if correlation_id in self.response_callbacks:
                del self.response_callbacks[correlation_id]
            raise
    
    async def _send_shutdown_notification(self):
        """Send shutdown notification to other agents"""
        
        await self.broadcast_message(
            message_type=MCPMessageType.SHUTDOWN,
            payload={
                "agent_id": self.agent_id,
                "shutdown_time": datetime.now().isoformat(),
                "reason": "graceful_shutdown"
            },
            priority=MCPPriority.HIGH
        )
    
    def get_discovered_agents(self) -> Dict[str, Dict[str, Any]]:
        """Get list of discovered agents and their capabilities"""
        return self.discovered_agents.copy()
    
    def get_message_stats(self) -> Dict[str, Any]:
        """Get message statistics"""
        return {
            "agent_id": self.agent_id,
            "message_stats": self.message_stats,
            "capabilities_count": len(self.capabilities),
            "discovered_agents_count": len(self.discovered_agents),
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "is_active": self.is_active,
            "message_history_size": len(self.message_history)
        }
    
    def get_capability_info(self, capability_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific capability"""
        if capability_name in self.capabilities:
            return asdict(self.capabilities[capability_name])
        return None
    
    def clear_message_history(self, keep_last: int = 100):
        """Clear message history, keeping only the most recent messages"""
        if len(self.message_history) > keep_last:
            self.message_history = self.message_history[-keep_last:]
            logger.info(f"🧹 Message history cleared, kept last {keep_last} messages")

logger.info("🔄 MCP Handler loaded successfully")