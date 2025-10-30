"""
Master Coordination Agent - Central orchestrator for all AI agents
Manages inter-agent communication, system coordination, and high-level decision making
"""

import asyncio
import logging
import json
from typing import Optional, Dict, Any, List
from datetime import datetime
import redis
import chromadb

logger = logging.getLogger(__name__)

class MasterCoordinationAgent:
    """
    Central AI agent that orchestrates communication between all specialized agents
    and makes high-level strategic decisions using LangChain framework
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None, chroma_client: Optional[chromadb.Client] = None):
        self.redis_client = redis_client
        self.chroma_client = chroma_client
        self.agent_id = "master_coordination_agent"
        self.active_agents: Dict[str, Dict[str, Any]] = {}
        self.system_status = "initializing"
        self.initialized = False
        
        # MCP communication channels
        self.mcp_channels = {
            'route_optimization': 'mcp:route_optimization',
            'alert_management': 'mcp:alert_management',
            'analytics': 'mcp:analytics',
            'bin_simulation': 'mcp:bin_simulation',
            'master_coordination': 'mcp:master_coordination'
        }
        
        logger.info(f"🤖 Master Coordination Agent created with ID: {self.agent_id}")
    
    async def initialize(self) -> bool:
        """Initialize the Master Coordination Agent and establish connections"""
        try:
            logger.info("🚀 Initializing Master Coordination Agent...")
            
            # Initialize vector database collections if available
            if self.chroma_client:
                await self._initialize_vector_collections()
            
            # Setup MCP communication if Redis is available
            if self.redis_client:
                await self._setup_mcp_communication()
            
            # Initialize agent registry
            await self._initialize_agent_registry()
            
            # Initialize and connect all specialized agents
            await self._initialize_specialized_agents()
            
            # Start inter-agent communication monitoring
            await self._start_agent_monitoring()
            
            # Set system status
            self.system_status = "active"
            self.initialized = True
            
            logger.info("✅ Master Coordination Agent initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Master Coordination Agent: {e}")
            self.system_status = "error"
            return False
    
    async def _initialize_vector_collections(self):
        """Initialize vector database collections for contextual memory"""
        try:
            # Get or create agent conversation collection
            try:
                self.conversation_collection = self.chroma_client.get_collection("agent_conversations")
            except:
                self.conversation_collection = self.chroma_client.create_collection(
                    name="agent_conversations",
                    metadata={"description": "Master agent conversation history and context"}
                )
            
            logger.info("📚 Vector database collections initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize vector collections: {e}")
    
    async def _setup_mcp_communication(self):
        """Setup Model Context Protocol communication channels"""
        try:
            # Subscribe to master coordination channel
            if self.redis_client:
                # Clear any existing messages
                await asyncio.get_event_loop().run_in_executor(
                    None, 
                    self.redis_client.delete, 
                    self.mcp_channels['master_coordination']
                )
                
                # Publish initialization message
                init_message = {
                    "agent_id": self.agent_id,
                    "message_type": "initialization",
                    "timestamp": datetime.now().isoformat(),
                    "status": "active",
                    "capabilities": [
                        "agent_coordination",
                        "task_distribution", 
                        "failure_handling",
                        "system_monitoring"
                    ]
                }
                
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.redis_client.lpush,
                    self.mcp_channels['master_coordination'],
                    json.dumps(init_message)
                )
            
            logger.info("🔄 MCP communication channels setup complete")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup MCP communication: {e}")
    
    async def _initialize_agent_registry(self):
        """Initialize registry of available agents"""
        try:
            # Define expected agents and their capabilities
            expected_agents = {
                "route_optimization": {
                    "status": "not_loaded",
                    "capabilities": ["route_planning", "vrp_solving", "optimization"],
                    "last_heartbeat": None
                },
                "alert_management": {
                    "status": "not_loaded", 
                    "capabilities": ["overflow_prediction", "alert_generation", "escalation"],
                    "last_heartbeat": None
                },
                "analytics": {
                    "status": "not_loaded",
                    "capabilities": ["data_analysis", "reporting", "insights"],
                    "last_heartbeat": None
                },
                "bin_simulation": {
                    "status": "not_loaded",
                    "capabilities": ["data_simulation", "iot_emulation", "pattern_generation"],
                    "last_heartbeat": None
                }
            }
            
            self.active_agents = expected_agents
            logger.info(f"📋 Agent registry initialized with {len(expected_agents)} expected agents")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize agent registry: {e}")
    
    async def register_agent(self, agent_id: str, capabilities: List[str]) -> bool:
        """Register a new agent with the master coordinator"""
        try:
            if agent_id in self.active_agents:
                self.active_agents[agent_id].update({
                    "status": "active",
                    "capabilities": capabilities,
                    "last_heartbeat": datetime.now().isoformat()
                })
                
                logger.info(f"✅ Agent registered: {agent_id} with capabilities: {capabilities}")
                return True
            else:
                logger.warning(f"⚠️ Unknown agent attempted registration: {agent_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to register agent {agent_id}: {e}")
            return False
    
    async def send_message_to_agent(self, target_agent: str, message: Dict[str, Any]) -> bool:
        """Send message to specific agent via MCP protocol"""
        try:
            if not self.redis_client:
                logger.warning("⚠️ Redis not available - cannot send inter-agent message")
                return False
            
            if target_agent not in self.mcp_channels:
                logger.error(f"❌ Unknown target agent: {target_agent}")
                return False
            
            # Prepare MCP message
            mcp_message = {
                "sender_agent": self.agent_id,
                "receiver_agent": target_agent,
                "message_id": f"msg_{datetime.now().timestamp()}",
                "timestamp": datetime.now().isoformat(),
                "message_type": message.get("type", "request"),
                "payload": message
            }
            
            # Send via Redis
            channel = self.mcp_channels[target_agent]
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.redis_client.lpush,
                channel,
                json.dumps(mcp_message)
            )
            
            logger.info(f"📤 Message sent to {target_agent}: {message.get('type', 'request')}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send message to {target_agent}: {e}")
            return False
    
    async def handle_agent_failure(self, failed_agent: str, error_info: Dict[str, Any]):
        """Handle agent failure with graceful degradation"""
        try:
            logger.warning(f"⚠️ Agent failure detected: {failed_agent}")
            
            if failed_agent in self.active_agents:
                self.active_agents[failed_agent]["status"] = "failed"
                self.active_agents[failed_agent]["error"] = error_info
            
            # Implement graceful degradation logic
            await self._redistribute_tasks(failed_agent)
            
            # Log failure for analysis
            if self.chroma_client:
                await self._log_failure_context(failed_agent, error_info)
            
            logger.info(f"🔄 Failure handling completed for {failed_agent}")
            
        except Exception as e:
            logger.error(f"❌ Failed to handle agent failure: {e}")
    
    async def _redistribute_tasks(self, failed_agent: str):
        """Redistribute tasks from failed agent to available agents"""
        try:
            # This is a placeholder for task redistribution logic
            # In a full implementation, this would:
            # 1. Identify pending tasks from failed agent
            # 2. Find capable alternative agents
            # 3. Reassign tasks with appropriate context
            
            logger.info(f"🔄 Task redistribution initiated for failed agent: {failed_agent}")
            
            # For now, just log the event
            redistribution_event = {
                "event_type": "task_redistribution",
                "failed_agent": failed_agent,
                "timestamp": datetime.now().isoformat(),
                "status": "initiated"
            }
            
            if self.redis_client:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.redis_client.lpush,
                    "system_events",
                    json.dumps(redistribution_event)
                )
            
        except Exception as e:
            logger.error(f"❌ Failed to redistribute tasks: {e}")
    
    async def _log_failure_context(self, failed_agent: str, error_info: Dict[str, Any]):
        """Log failure context to vector database for learning"""
        try:
            failure_context = {
                "agent": failed_agent,
                "error": str(error_info),
                "timestamp": datetime.now().isoformat(),
                "system_state": self.get_system_status()
            }
            
            # Store in vector database for future analysis
            # This would be expanded with proper embeddings in full implementation
            logger.info(f"📝 Failure context logged for {failed_agent}")
            
        except Exception as e:
            logger.error(f"❌ Failed to log failure context: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and agent health"""
        return {
            "master_agent_status": self.system_status,
            "initialized": self.initialized,
            "active_agents": self.active_agents,
            "redis_available": self.redis_client is not None,
            "vector_db_available": self.chroma_client is not None,
            "timestamp": datetime.now().isoformat()
        }
    
    async def process_natural_language_query(self, query: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Process natural language queries from dashboard users"""
        try:
            # Placeholder for natural language processing
            # In full implementation, this would use LangChain for:
            # 1. Intent recognition
            # 2. Entity extraction  
            # 3. Context retrieval from vector database
            # 4. Response generation
            
            response = {
                "query": query,
                "response": "Natural language processing not yet implemented. This will be enhanced with LangChain integration.",
                "suggestions": [
                    "Check bin status in your ward",
                    "View recent alerts",
                    "Generate analytics report"
                ],
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"💬 Natural language query processed: {query[:50]}...")
            return response
            
        except Exception as e:
            logger.error(f"❌ Failed to process natural language query: {e}")
            return {
                "query": query,
                "error": "Failed to process query",
                "timestamp": datetime.now().isoformat()
            }
    
    async def _initialize_specialized_agents(self):
        """Initialize and connect all specialized agents"""
        try:
            from .route_optimizer import RouteOptimizationAgent
            from .alert_manager import AlertManagementAgent
            from .analytics_agent import AnalyticsAgent
            from .bin_simulator import BinSimulatorAgent
            from .mcp_handler import MCPHandler
            from .vector_db import VectorDatabaseManager
            from .langgraph_workflows import LangGraphWorkflowEngine
            
            # Initialize shared components
            mcp_handler = MCPHandler(
                agent_id="master_coordination",
                redis_client=self.redis_client
            ) if self.redis_client else None
            
            vector_db = VectorDatabaseManager() if self.chroma_client else None
            if vector_db:
                await vector_db.initialize()
            
            workflow_engine = LangGraphWorkflowEngine()
            
            # Initialize Route Optimization Agent
            self.route_optimizer = RouteOptimizationAgent(
                redis_client=self.redis_client,
                vector_db=vector_db,
                mcp_handler=mcp_handler,
                workflow_engine=workflow_engine
            )
            await self.route_optimizer.initialize_agent()
            
            # Initialize Alert Management Agent
            self.alert_manager = AlertManagementAgent(
                redis_client=self.redis_client,
                vector_db=vector_db,
                mcp_handler=mcp_handler
            )
            await self.alert_manager.initialize_agent()
            
            # Initialize Analytics Agent
            self.analytics_agent = AnalyticsAgent(
                redis_client=self.redis_client,
                vector_db=vector_db,
                mcp_handler=mcp_handler
            )
            await self.analytics_agent.initialize_agent()
            
            # Initialize Bin Simulator Agent
            self.bin_simulator = BinSimulatorAgent(
                redis_client=self.redis_client,
                vector_db=vector_db,
                mcp_handler=mcp_handler
            )
            
            # Configure bin simulator with default settings
            config = {
                "existing_bins": [],  # Will be populated from markers.json if available
                "simulation_speed": 1.0
            }
            await self.bin_simulator.initialize_simulation(config)
            
            # Start bin simulation
            await self.bin_simulator.start_simulation(speed_multiplier=1.0)
            
            # Update agent registry with actual agents
            self.active_agents.update({
                "route_optimization": {
                    "status": "active",
                    "capabilities": ["route_planning", "vrp_solving", "optimization"],
                    "last_heartbeat": datetime.now().isoformat(),
                    "agent_instance": self.route_optimizer
                },
                "alert_management": {
                    "status": "active",
                    "capabilities": ["overflow_prediction", "alert_generation", "escalation"],
                    "last_heartbeat": datetime.now().isoformat(),
                    "agent_instance": self.alert_manager
                },
                "analytics": {
                    "status": "active",
                    "capabilities": ["data_analysis", "reporting", "insights"],
                    "last_heartbeat": datetime.now().isoformat(),
                    "agent_instance": self.analytics_agent
                },
                "bin_simulation": {
                    "status": "active",
                    "capabilities": ["data_simulation", "iot_emulation", "pattern_generation"],
                    "last_heartbeat": datetime.now().isoformat(),
                    "agent_instance": self.bin_simulator
                }
            })
            
            # Start alert monitoring
            await self.alert_manager.start_monitoring()
            
            logger.info("✅ All specialized agents initialized and connected")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize specialized agents: {e}")
            raise
    
    async def _start_agent_monitoring(self):
        """Start monitoring inter-agent communication and health"""
        try:
            if self.redis_client:
                # Start background task to monitor agent communication
                asyncio.create_task(self._monitor_agent_communication())
                
                # Start background task to coordinate data flow
                asyncio.create_task(self._coordinate_data_flow())
                
                logger.info("🔍 Agent monitoring and coordination started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start agent monitoring: {e}")
    
    async def _monitor_agent_communication(self):
        """Monitor communication between agents"""
        while self.system_status == "active":
            try:
                # Check agent health via heartbeat
                for agent_id, agent_info in self.active_agents.items():
                    if "agent_instance" in agent_info:
                        agent = agent_info["agent_instance"]
                        
                        # Update heartbeat
                        agent_info["last_heartbeat"] = datetime.now().isoformat()
                        
                        # Check if agent has any status updates
                        if hasattr(agent, 'get_agent_status'):
                            status = agent.get_agent_status()
                            agent_info["status"] = "active" if status.get("is_initialized", False) else "error"
                
                # Sleep for monitoring interval
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"❌ Error in agent communication monitoring: {e}")
                await asyncio.sleep(30)
    
    async def _coordinate_data_flow(self):
        """Coordinate data flow between agents"""
        while self.system_status == "active":
            try:
                # Get bin data from simulator
                if hasattr(self, 'bin_simulator') and self.bin_simulator:
                    bin_data = await self.bin_simulator.get_bin_data()
                    
                    if bin_data:
                        # Send bin data to alert manager for monitoring
                        if hasattr(self, 'alert_manager') and self.alert_manager:
                            # Process bins for potential alerts
                            for bin_info in bin_data[:10]:  # Process first 10 bins to avoid overload
                                await self.alert_manager._process_bin_for_alerts(bin_info)
                        
                        # Publish data via Redis for dashboard updates
                        if self.redis_client:
                            message = {
                                "message_type": "coordinated_bin_data",
                                "timestamp": datetime.now().isoformat(),
                                "source": "master_coordination",
                                "data": bin_data[:50],  # Limit data size
                                "total_bins": len(bin_data),
                                "critical_bins": len([b for b in bin_data if b.get("status") == "critical"]),
                                "warning_bins": len([b for b in bin_data if b.get("status") == "warning"])
                            }
                            
                            await asyncio.get_event_loop().run_in_executor(
                                None,
                                self.redis_client.lpush,
                                "coordinated_data_stream",
                                json.dumps(message)
                            )
                
                # Sleep for coordination interval
                await asyncio.sleep(60)  # Coordinate every minute
                
            except Exception as e:
                logger.error(f"❌ Error in data flow coordination: {e}")
                await asyncio.sleep(60)
    
    async def request_route_optimization(self, ward_id: int, constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Request route optimization from the Route Optimization Agent"""
        try:
            if not hasattr(self, 'route_optimizer') or not self.route_optimizer:
                return {"error": "Route Optimization Agent not available"}
            
            # Get current bin data
            bin_data = []
            if hasattr(self, 'bin_simulator') and self.bin_simulator:
                all_bins = await self.bin_simulator.get_bin_data([ward_id])
                # Filter bins that need collection
                bin_data = [b for b in all_bins if b.get("current_fill", 0) > 70]
            
            # Create vehicle data (simplified)
            vehicles = [
                {
                    "vehicle_id": f"vehicle_{i+1}",
                    "capacity": 2000.0,
                    "current_location": (19.0760, 72.8777),
                    "max_distance": 100.0,
                    "cost_per_km": 15.0
                }
                for i in range(2)  # 2 vehicles per ward
            ]
            
            # Import the data classes
            from .route_optimizer import VehicleInfo, BinLocation
            
            # Request optimization
            result = await self.route_optimizer.optimize_routes(
                ward_id=ward_id,
                available_vehicles=[VehicleInfo(**v) for v in vehicles],
                bin_locations=[BinLocation(
                    bin_id=b["id"],
                    latitude=b["latitude"],
                    longitude=b["longitude"],
                    fill_level=b["current_fill"],
                    priority=5 if b.get("status") == "critical" else 3,
                    estimated_collection_time=5,
                    bin_type=b.get("bin_type", "residential"),
                    capacity=b.get("capacity", 240)
                ) for b in bin_data],
                constraints=constraints or {}
            )
            
            logger.info(f"🚛 Route optimization completed for ward {ward_id}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Route optimization request failed: {e}")
            return {"error": str(e)}
    
    async def get_active_alerts(self, ward_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get active alerts from Alert Management Agent"""
        try:
            if not hasattr(self, 'alert_manager') or not self.alert_manager:
                return []
            
            alerts = self.alert_manager.get_active_alerts(ward_id)
            logger.info(f"📋 Retrieved {len(alerts)} active alerts")
            return alerts
            
        except Exception as e:
            logger.error(f"❌ Failed to get active alerts: {e}")
            return []
    
    async def generate_analytics_report(self, analysis_type: str = "comprehensive", ward_id: Optional[int] = None) -> Dict[str, Any]:
        """Generate analytics report from Analytics Agent"""
        try:
            if not hasattr(self, 'analytics_agent') or not self.analytics_agent:
                return {"error": "Analytics Agent not available"}
            
            result = await self.analytics_agent.analyze_waste_data(
                data_sources=["bin_data", "route_data"],
                analysis_type=analysis_type,
                user_context={"ward_id": ward_id} if ward_id else {}
            )
            
            logger.info(f"📊 Analytics report generated: {analysis_type}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Analytics report generation failed: {e}")
            return {"error": str(e)}
    
    async def get_system_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive system data for dashboard"""
        try:
            # Get bin data
            bin_data = []
            if hasattr(self, 'bin_simulator') and self.bin_simulator:
                bin_data = await self.bin_simulator.get_bin_data()
            
            # Get active alerts
            alerts = await self.get_active_alerts()
            
            # Get system statistics
            system_stats = {
                "total_bins": len(bin_data),
                "critical_bins": len([b for b in bin_data if b.get("status") == "critical"]),
                "warning_bins": len([b for b in bin_data if b.get("status") == "warning"]),
                "normal_bins": len([b for b in bin_data if b.get("status") == "normal"]),
                "active_alerts": len(alerts),
                "system_status": self.system_status,
                "agents_active": len([a for a in self.active_agents.values() if a.get("status") == "active"])
            }
            
            return {
                "bin_data": bin_data[:100],  # Limit for performance
                "alerts": alerts,
                "system_stats": system_stats,
                "agent_status": self.get_system_status(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get system dashboard data: {e}")
            return {"error": str(e)}

    async def shutdown(self):
        """Gracefully shutdown the Master Coordination Agent"""
        try:
            logger.info("🛑 Shutting down Master Coordination Agent...")
            
            # Stop specialized agents
            if hasattr(self, 'bin_simulator') and self.bin_simulator:
                await self.bin_simulator.stop_simulation()
            
            if hasattr(self, 'alert_manager') and self.alert_manager:
                await self.alert_manager.stop_monitoring()
            
            # Notify all agents of shutdown
            if self.redis_client:
                shutdown_message = {
                    "sender_agent": self.agent_id,
                    "message_type": "shutdown",
                    "timestamp": datetime.now().isoformat()
                }
                
                for agent_type, channel in self.mcp_channels.items():
                    if agent_type != "master_coordination":
                        try:
                            await asyncio.get_event_loop().run_in_executor(
                                None,
                                self.redis_client.lpush,
                                channel,
                                json.dumps(shutdown_message)
                            )
                        except Exception as e:
                            logger.error(f"❌ Failed to notify {agent_type} of shutdown: {e}")
            
            # Update status
            self.system_status = "shutdown"
            self.initialized = False
            
            logger.info("✅ Master Coordination Agent shutdown completed")
            
        except Exception as e:
            logger.error(f"❌ Error during Master Coordination Agent shutdown: {e}")