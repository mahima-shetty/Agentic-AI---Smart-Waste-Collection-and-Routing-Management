"""
LangChain Base Framework - Unified agent classes and framework integration
Provides base classes and utilities for all AI agents using LangChain
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from abc import ABC, abstractmethod

try:
    from langchain.agents import create_openai_functions_agent, AgentExecutor
except ImportError:
    try:
        from langchain_core.agents import create_openai_functions_agent, AgentExecutor
    except ImportError:
        # Fallback for newer versions
        create_openai_functions_agent = None
        AgentExecutor = None

try:
    from langchain_core.agents import AgentAction, AgentFinish
except ImportError:
    from langchain.schema import AgentAction, AgentFinish
# Simplified memory implementation for now
class ConversationBufferMemory:
    def __init__(self, memory_key="chat_history", return_messages=True):
        self.memory_key = memory_key
        self.return_messages = return_messages
        self.chat_memory = SimpleMemory()
    
    def clear(self):
        self.chat_memory.messages = []

class SimpleMemory:
    def __init__(self):
        self.messages = []
    
    def add_user_message(self, message):
        self.messages.append({"type": "human", "content": message})
    
    def add_ai_message(self, message):
        self.messages.append({"type": "ai", "content": message})
# Simplified implementations for core functionality
class BaseTool:
    def __init__(self, name, description, func, **kwargs):
        self.name = name
        self.description = description
        self.func = func

class BaseCallbackHandler:
    pass

class ChatOpenAI:
    def __init__(self, model="gpt-3.5-turbo", temperature=0.1, max_tokens=1000):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

class HumanMessage:
    def __init__(self, content):
        self.content = content

class AIMessage:
    def __init__(self, content):
        self.content = content

class SystemMessage:
    def __init__(self, content):
        self.content = content

logger = logging.getLogger(__name__)

class WasteManagementCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for waste management agents"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.execution_log = []
    
    def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        """Called when agent takes an action"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": self.agent_id,
            "action": action.tool,
            "input": action.tool_input,
            "log": action.log
        }
        self.execution_log.append(log_entry)
        logger.info(f"🔧 Agent {self.agent_id} executing: {action.tool}")
    
    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
        """Called when agent finishes execution"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": self.agent_id,
            "result": finish.return_values,
            "log": finish.log
        }
        self.execution_log.append(log_entry)
        logger.info(f"✅ Agent {self.agent_id} completed execution")
    
    def get_execution_log(self) -> List[Dict[str, Any]]:
        """Get the execution log for this agent"""
        return self.execution_log.copy()

class BaseLangChainAgent(ABC):
    """
    Base class for all LangChain-powered waste management agents
    Provides common functionality and interface for specialized agents
    """
    
    def __init__(
        self, 
        agent_id: str, 
        agent_type: str,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.1,
        max_tokens: int = 1000
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize LangChain components
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Memory for conversation history
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Custom callback handler
        self.callback_handler = WasteManagementCallbackHandler(agent_id)
        
        # Tools and agent executor (to be set by subclasses)
        self.tools: List[BaseTool] = []
        self.agent_executor: Optional[AgentExecutor] = None
        
        # Agent state
        self.is_initialized = False
        self.current_task = None
        self.performance_metrics = {
            "tasks_completed": 0,
            "errors_encountered": 0,
            "average_response_time": 0.0,
            "last_activity": None
        }
        
        logger.info(f"🤖 Base LangChain agent created: {agent_id} ({agent_type})")
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent type"""
        pass
    
    @abstractmethod
    def get_tools(self) -> List[BaseTool]:
        """Get the tools available to this agent"""
        pass
    
    def initialize_agent(self) -> bool:
        """Initialize the LangChain agent with tools and prompt"""
        try:
            # Get tools from subclass
            self.tools = self.get_tools()
            
            # For now, use a simplified initialization without complex agent framework
            # This allows the bin simulator to work without full LangChain agent setup
            self.is_initialized = True
            logger.info(f"✅ Base agent initialized: {self.agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize agent {self.agent_id}: {e}")
            return False
    
    async def process_request(self, request: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a request using the agent"""
        if not self.is_initialized:
            return {
                "error": "Agent not initialized",
                "agent_id": self.agent_id,
                "timestamp": datetime.now().isoformat()
            }
        
        start_time = datetime.now()
        
        try:
            # Set current task
            self.current_task = request
            
            # For now, return a simple response
            # This can be enhanced later with full LangChain integration
            result = f"Agent {self.agent_id} processed request: {request}"
            
            # Update performance metrics
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            self._update_performance_metrics(response_time, success=True)
            
            return {
                "result": result,
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "response_time": response_time,
                "timestamp": end_time.isoformat(),
                "execution_log": self.callback_handler.get_execution_log()
            }
            
        except Exception as e:
            # Update performance metrics for error
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            self._update_performance_metrics(response_time, success=False)
            
            logger.error(f"❌ Agent {self.agent_id} failed to process request: {e}")
            return {
                "error": str(e),
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "response_time": response_time,
                "timestamp": end_time.isoformat()
            }
        finally:
            self.current_task = None
    
    def _update_performance_metrics(self, response_time: float, success: bool):
        """Update agent performance metrics"""
        if success:
            self.performance_metrics["tasks_completed"] += 1
        else:
            self.performance_metrics["errors_encountered"] += 1
        
        # Update average response time
        total_tasks = (self.performance_metrics["tasks_completed"] + 
                      self.performance_metrics["errors_encountered"])
        
        if total_tasks > 1:
            current_avg = self.performance_metrics["average_response_time"]
            self.performance_metrics["average_response_time"] = (
                (current_avg * (total_tasks - 1) + response_time) / total_tasks
            )
        else:
            self.performance_metrics["average_response_time"] = response_time
        
        self.performance_metrics["last_activity"] = datetime.now().isoformat()
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status and metrics"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "is_initialized": self.is_initialized,
            "current_task": self.current_task,
            "performance_metrics": self.performance_metrics,
            "memory_size": len(self.memory.chat_memory.messages) if self.memory else 0,
            "available_tools": [tool.name for tool in self.tools],
            "timestamp": datetime.now().isoformat()
        }
    
    def clear_memory(self):
        """Clear the agent's conversation memory"""
        if self.memory:
            self.memory.clear()
            logger.info(f"🧹 Cleared memory for agent {self.agent_id}")
    
    def add_memory(self, human_message: str, ai_message: str):
        """Add a message pair to the agent's memory"""
        if self.memory:
            self.memory.chat_memory.add_user_message(human_message)
            self.memory.chat_memory.add_ai_message(ai_message)
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history as a list of dictionaries"""
        if not self.memory:
            return []
        
        history = []
        for message in self.memory.chat_memory.messages:
            if isinstance(message, HumanMessage):
                history.append({"type": "human", "content": message.content})
            elif isinstance(message, AIMessage):
                history.append({"type": "ai", "content": message.content})
            elif isinstance(message, SystemMessage):
                history.append({"type": "system", "content": message.content})
        
        return history

class WasteManagementTool(BaseTool):
    """Base class for waste management specific tools"""
    
    def __init__(self, name: str, description: str, func: Callable, **kwargs):
        super().__init__(name=name, description=description, func=func, **kwargs)
        self.usage_count = 0
        self.last_used = None
    
    def _run(self, *args, **kwargs):
        """Execute the tool and track usage"""
        self.usage_count += 1
        self.last_used = datetime.now().isoformat()
        
        try:
            result = self.func(*args, **kwargs)
            logger.info(f"🔧 Tool {self.name} executed successfully")
            return result
        except Exception as e:
            logger.error(f"❌ Tool {self.name} failed: {e}")
            raise
    
    async def _arun(self, *args, **kwargs):
        """Async version of tool execution"""
        return self._run(*args, **kwargs)
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for this tool"""
        return {
            "name": self.name,
            "usage_count": self.usage_count,
            "last_used": self.last_used
        }

class AgentPromptTemplates:
    """Collection of prompt templates for different agent types"""
    
    ROUTE_OPTIMIZATION_SYSTEM = """
    You are a Route Optimization Agent for a Smart Waste Management System.
    Your role is to optimize waste collection routes using advanced algorithms and machine learning.
    
    Key responsibilities:
    - Analyze bin fill levels and predict collection needs
    - Calculate optimal routes using Vehicle Routing Problem (VRP) algorithms
    - Consider traffic patterns, vehicle capacity, and time constraints
    - Learn from historical collection data to improve future routing
    - Coordinate with other agents for emergency route adjustments
    
    Always provide clear, actionable route recommendations with reasoning.
    Focus on minimizing collection time, fuel costs, and ensuring all bins are serviced efficiently.
    """
    
    ALERT_MANAGEMENT_SYSTEM = """
    You are an Alert Management Agent for a Smart Waste Management System.
    Your role is to monitor bin status and generate intelligent alerts for overflow prevention.
    
    Key responsibilities:
    - Monitor bin fill levels and predict overflow events 2-4 hours in advance
    - Generate priority-based alerts with clear severity levels
    - Cluster nearby overflow risks for coordinated response
    - Escalate critical situations to supervisory staff
    - Learn from alert response patterns to improve accuracy
    
    Always provide clear, actionable alerts with specific recommendations.
    Focus on preventing environmental hazards and citizen complaints through proactive monitoring.
    """
    
    ANALYTICS_SYSTEM = """
    You are an Analytics Agent for a Smart Waste Management System.
    Your role is to analyze waste management data and provide intelligent insights.
    
    Key responsibilities:
    - Process collection data and identify patterns and trends
    - Generate dynamic visualizations and comprehensive reports
    - Calculate cost optimizations and efficiency metrics
    - Discover hidden patterns using machine learning models
    - Provide personalized insights based on user behavior
    
    Always provide clear, data-driven insights with actionable recommendations.
    Focus on helping operators make informed decisions for waste management optimization.
    """
    
    BIN_SIMULATION_SYSTEM = """
    You are a Bin Simulator Agent for a Smart Waste Management System.
    Your role is to generate realistic IoT sensor data for demonstration and testing.
    
    Key responsibilities:
    - Generate realistic fill level data with temporal patterns
    - Simulate various bin types and collection scenarios
    - Create festival/event-based fill rate variations
    - Provide consistent data streams for agent training
    - Support multiple ward configurations
    
    Always generate realistic, consistent data that reflects real-world waste collection patterns.
    Focus on providing high-quality simulation data for system demonstration and agent training.
    """

def create_waste_management_tool(
    name: str, 
    description: str, 
    func: Callable,
    **kwargs
) -> WasteManagementTool:
    """Factory function to create waste management tools"""
    return WasteManagementTool(
        name=name,
        description=description,
        func=func,
        **kwargs
    )

# Utility functions for common agent operations
def format_agent_response(
    agent_id: str,
    result: Any,
    context: Optional[Dict[str, Any]] = None,
    recommendations: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Format a standardized agent response"""
    response = {
        "agent_id": agent_id,
        "result": result,
        "timestamp": datetime.now().isoformat()
    }
    
    if context:
        response["context"] = context
    
    if recommendations:
        response["recommendations"] = recommendations
    
    return response

def validate_agent_input(input_data: Dict[str, Any], required_fields: List[str]) -> bool:
    """Validate that required fields are present in agent input"""
    for field in required_fields:
        if field not in input_data:
            logger.error(f"❌ Missing required field: {field}")
            return False
    return True

logger.info("📚 LangChain base framework loaded successfully")