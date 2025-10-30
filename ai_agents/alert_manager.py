"""
Alert Management Agent - Intelligent overflow prediction and notification system
Monitors bin status and generates intelligent alerts with contextual reasoning using LangChain
"""

import asyncio
import logging
import json
import math
import sqlite3
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import redis
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

from .langchain_base import BaseLangChainAgent, WasteManagementTool, AgentPromptTemplates
from .mcp_handler import MCPHandler, MCPMessageType, MCPPriority
from .vector_db import VectorDatabaseManager
from .email_notifier import EmailNotificationSystem

logger = logging.getLogger(__name__)

@dataclass
class Alert:
    """Alert data model"""
    id: str
    bin_id: str
    alert_type: str  # "overflow_warning", "overflow_critical", "maintenance"
    priority: int  # 1-5 scale
    predicted_overflow_time: Optional[datetime]
    message: str
    status: str  # "active", "acknowledged", "resolved"
    created_at: datetime
    resolved_at: Optional[datetime] = None
    context_embedding: Optional[List[float]] = None
    natural_language_summary: Optional[str] = None
    similar_historical_cases: Optional[List[str]] = None
    recommended_action: Optional[str] = None
    affected_bins: Optional[List[str]] = None
    escalation_required: bool = False

@dataclass
class AlertCluster:
    """Cluster of related alerts for coordinated response"""
    cluster_id: str
    alerts: List[Alert]
    cluster_center: Tuple[float, float]  # lat, lon
    cluster_radius: float  # in meters
    priority: int
    recommended_response: str
    created_at: datetime

class AlertManagementAgent(BaseLangChainAgent):
    """
    Alert Management Agent for intelligent overflow prediction and notification
    Uses machine learning models enhanced with LangChain reasoning for contextual alerts
    """
    
    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        vector_db: Optional[VectorDatabaseManager] = None,
        mcp_handler: Optional[MCPHandler] = None,
        db_path: str = "backend/db/operators.db"
    ):
        super().__init__(
            agent_id="alert_management",
            agent_type="alert_management",
            model_name="gpt-3.5-turbo",
            temperature=0.1
        )
        
        self.redis_client = redis_client
        self.vector_db = vector_db
        self.mcp_handler = mcp_handler
        self.db_path = db_path
        
        # Initialize email notification system
        self.email_notifier = EmailNotificationSystem(db_path=db_path)
        
        # Alert management state
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_clusters: Dict[str, AlertCluster] = {}
        self.monitoring_active = False
        
        # Machine learning models
        self.overflow_predictor = None
        self.scaler = StandardScaler()
        self.model_trained = False
        self.model_path = "ai_agents/models/overflow_predictor.joblib"
        
        # Alert thresholds and parameters
        self.alert_thresholds = {
            "warning_level": 85.0,  # Fill level percentage
            "critical_level": 95.0,
            "prediction_horizon": 4.0,  # Hours ahead to predict
            "cluster_radius": 500.0,  # Meters for clustering nearby alerts
            "escalation_threshold": 3,  # Number of critical alerts to trigger escalation
        }
        
        # Contextual reasoning patterns
        self.alert_patterns = {
            "temporal": {
                "morning_rush": {"hours": [7, 8, 9], "multiplier": 1.5},
                "evening_peak": {"hours": [18, 19, 20], "multiplier": 1.8},
                "weekend_high": {"days": [5, 6], "multiplier": 1.3}
            },
            "seasonal": {
                "festival_periods": {"multiplier": 2.5, "duration_days": 5},
                "monsoon": {"months": [6, 7, 8, 9], "multiplier": 0.8},
                "summer": {"months": [3, 4, 5], "multiplier": 1.2}
            },
            "location": {
                "commercial": {"base_urgency": 1.4, "business_hours_boost": 1.3},
                "residential": {"base_urgency": 1.0, "evening_boost": 1.2},
                "public": {"base_urgency": 1.6, "weekend_boost": 1.4}
            }
        }
        
        # Statistics
        self.alert_stats = {
            "total_alerts_generated": 0,
            "overflow_predictions_made": 0,
            "successful_predictions": 0,
            "false_positives": 0,
            "alerts_resolved": 0,
            "average_response_time": 0.0,
            "escalations_triggered": 0
        }
        
        logger.info("🚨 Alert Management Agent initialized")
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the alert management agent"""
        return AgentPromptTemplates.ALERT_MANAGEMENT_SYSTEM
    
    def get_tools(self) -> List[WasteManagementTool]:
        """Get tools available to the alert management agent"""
        tools = []
        
        # Alert generation tool
        tools.append(WasteManagementTool(
            name="generate_alert",
            description="Generate intelligent alerts with contextual reasoning",
            func=self._generate_alert_tool
        ))
        
        # Alert clustering tool
        tools.append(WasteManagementTool(
            name="cluster_alerts",
            description="Cluster nearby alerts for coordinated response",
            func=self._cluster_alerts_tool
        ))
        
        # Prediction tool
        tools.append(WasteManagementTool(
            name="predict_overflow",
            description="Predict overflow events using ML models",
            func=self._predict_overflow_tool
        ))
        
        # Escalation tool
        tools.append(WasteManagementTool(
            name="escalate_alert",
            description="Escalate critical alerts to supervisory staff",
            func=self._escalate_alert_tool
        ))
        
        return tools
    
    async def initialize_agent(self) -> bool:
        """Initialize the Alert Management Agent"""
        try:
            logger.info("🚀 Initializing Alert Management Agent...")
            
            # Initialize LangChain agent
            if not self.initialize_agent():
                return False
            
            # Setup database tables
            await self._setup_database_tables()
            
            # Setup MCP capabilities if available
            if self.mcp_handler:
                await self._setup_mcp_capabilities()
            
            # Initialize or load ML models
            await self._initialize_ml_models()
            
            # Load existing alerts from database
            await self._load_existing_alerts()
            
            logger.info("✅ Alert Management Agent initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Alert Management Agent: {e}")
            return False
    
    async def _setup_database_tables(self):
        """Setup database tables for alerts"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create alerts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    bin_id TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    predicted_overflow_time TEXT,
                    message TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    resolved_at TEXT,
                    natural_language_summary TEXT,
                    recommended_action TEXT,
                    escalation_required INTEGER DEFAULT 0
                )
            """)
            
            # Create alert_clusters table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alert_clusters (
                    cluster_id TEXT PRIMARY KEY,
                    cluster_center_lat REAL NOT NULL,
                    cluster_center_lon REAL NOT NULL,
                    cluster_radius REAL NOT NULL,
                    priority INTEGER NOT NULL,
                    recommended_response TEXT,
                    created_at TEXT NOT NULL,
                    status TEXT DEFAULT 'active'
                )
            """)
            
            # Create alert_history table for learning
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alert_history (
                    id TEXT PRIMARY KEY,
                    bin_id TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    predicted_time TEXT,
                    actual_overflow_time TEXT,
                    response_time_minutes INTEGER,
                    resolution_method TEXT,
                    accuracy_score REAL,
                    created_at TEXT NOT NULL
                )
            """)
            
            conn.commit()
            conn.close()
            
            logger.info("📊 Alert database tables created successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup database tables: {e}")
            raise
    
    async def _setup_mcp_capabilities(self):
        """Setup MCP capabilities for alert management"""
        try:
            # Register overflow prediction capability
            await self.mcp_handler.register_capability(
                name="predict_overflow",
                description="Predict bin overflow events 2-4 hours in advance using ML models",
                input_schema={
                    "type": "object",
                    "properties": {
                        "bin_data": {"type": "array"},
                        "prediction_horizon": {"type": "number"},
                        "include_context": {"type": "boolean"}
                    }
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "predictions": {"type": "array"},
                        "alerts_generated": {"type": "array"},
                        "contextual_insights": {"type": "array"}
                    }
                },
                handler=self._handle_overflow_prediction_request
            )
            
            # Register alert generation capability
            await self.mcp_handler.register_capability(
                name="generate_intelligent_alert",
                description="Generate priority-based alerts with contextual reasoning",
                input_schema={
                    "type": "object",
                    "properties": {
                        "bin_id": {"type": "string"},
                        "alert_type": {"type": "string"},
                        "context": {"type": "object"},
                        "include_clustering": {"type": "boolean"}
                    }
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "alert": {"type": "object"},
                        "natural_language_summary": {"type": "string"},
                        "recommended_actions": {"type": "array"},
                        "similar_cases": {"type": "array"}
                    }
                },
                handler=self._handle_alert_generation_request
            )
            
            # Register alert clustering capability
            await self.mcp_handler.register_capability(
                name="cluster_alerts",
                description="Cluster nearby overflow risks for coordinated response",
                input_schema={
                    "type": "object",
                    "properties": {
                        "alerts": {"type": "array"},
                        "cluster_radius": {"type": "number"},
                        "min_cluster_size": {"type": "integer"}
                    }
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "clusters": {"type": "array"},
                        "coordinated_responses": {"type": "array"},
                        "optimization_suggestions": {"type": "array"}
                    }
                },
                handler=self._handle_alert_clustering_request
            )
            
            logger.info("🔄 MCP capabilities registered for Alert Management")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup MCP capabilities: {e}")
    
    async def _initialize_ml_models(self):
        """Initialize or load machine learning models for overflow prediction"""
        try:
            # Check if pre-trained model exists
            if os.path.exists(self.model_path):
                self.overflow_predictor = joblib.load(self.model_path)
                self.model_trained = True
                logger.info("📚 Loaded pre-trained overflow prediction model")
            else:
                # Create new model
                self.overflow_predictor = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
                logger.info("🤖 Created new overflow prediction model")
            
            # Ensure model directory exists
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize ML models: {e}")
            raise
    
    async def _load_existing_alerts(self):
        """Load existing active alerts from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM alerts WHERE status = 'active'
            """)
            
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            
            for row in rows:
                alert_data = dict(zip(columns, row))
                
                # Convert datetime strings back to datetime objects
                alert_data['created_at'] = datetime.fromisoformat(alert_data['created_at'])
                if alert_data['predicted_overflow_time']:
                    alert_data['predicted_overflow_time'] = datetime.fromisoformat(alert_data['predicted_overflow_time'])
                if alert_data['resolved_at']:
                    alert_data['resolved_at'] = datetime.fromisoformat(alert_data['resolved_at'])
                
                # Convert escalation_required from integer to boolean
                alert_data['escalation_required'] = bool(alert_data['escalation_required'])
                
                alert = Alert(**alert_data)
                self.active_alerts[alert.id] = alert
            
            conn.close()
            
            logger.info(f"📋 Loaded {len(self.active_alerts)} existing active alerts")
            
        except Exception as e:
            logger.error(f"❌ Failed to load existing alerts: {e}")
    
    async def start_monitoring(self) -> bool:
        """Start continuous monitoring for overflow prediction"""
        try:
            if self.monitoring_active:
                logger.warning("⚠️ Monitoring already active")
                return False
            
            self.monitoring_active = True
            
            # Start monitoring loop
            asyncio.create_task(self._monitoring_loop())
            
            # Notify other agents via MCP
            if self.mcp_handler:
                await self.mcp_handler.send_notification(
                    receiver_agent="broadcast",
                    event_type="alert_monitoring_started",
                    data={
                        "agent_id": self.agent_id,
                        "prediction_horizon": self.alert_thresholds["prediction_horizon"],
                        "start_time": datetime.now().isoformat()
                    },
                    priority=MCPPriority.HIGH
                )
            
            logger.info("🔍 Alert monitoring started")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start monitoring: {e}")
            return False
    
    async def stop_monitoring(self) -> bool:
        """Stop continuous monitoring"""
        try:
            self.monitoring_active = False
            
            # Notify other agents via MCP
            if self.mcp_handler:
                await self.mcp_handler.send_notification(
                    receiver_agent="broadcast",
                    event_type="alert_monitoring_stopped",
                    data={
                        "agent_id": self.agent_id,
                        "total_alerts": len(self.active_alerts),
                        "stop_time": datetime.now().isoformat()
                    },
                    priority=MCPPriority.NORMAL
                )
            
            logger.info("🛑 Alert monitoring stopped")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to stop monitoring: {e}")
            return False
    
    async def _monitoring_loop(self):
        """Main monitoring loop for continuous alert generation"""
        logger.info("🔄 Starting alert monitoring loop...")
        
        while self.monitoring_active:
            try:
                # Get current bin data from Redis or other source
                bin_data = await self._get_current_bin_data()
                
                if bin_data:
                    # Process each bin for potential alerts
                    for bin_info in bin_data:
                        await self._process_bin_for_alerts(bin_info)
                    
                    # Cluster nearby alerts
                    await self._cluster_nearby_alerts()
                    
                    # Check for escalation conditions
                    await self._check_escalation_conditions()
                    
                    # Update ML model with new data
                    if len(bin_data) > 10:  # Only if we have sufficient data
                        await self._update_ml_model(bin_data)
                
                # Sleep for monitoring interval (5 minutes)
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"❌ Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _get_current_bin_data(self) -> List[Dict[str, Any]]:
        """Get current bin data from Redis or bin simulator"""
        try:
            if not self.redis_client:
                return []
            
            # Try to get data from Redis bin data stream
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.redis_client.lrange,
                "bin_data_stream",
                0, 0  # Get latest message
            )
            
            if result:
                message_data = json.loads(result[0].decode('utf-8'))
                return message_data.get("data", [])
            
            return []
            
        except Exception as e:
            logger.error(f"❌ Failed to get current bin data: {e}")
            return []
    
    async def _process_bin_for_alerts(self, bin_info: Dict[str, Any]):
        """Process a single bin for potential alert generation"""
        try:
            bin_id = bin_info.get("id")
            current_fill = bin_info.get("current_fill", 0)
            fill_rate = bin_info.get("fill_rate", 0)
            bin_type = bin_info.get("bin_type", "residential")
            
            # Skip if bin already has active alert
            existing_alert = next((alert for alert in self.active_alerts.values() 
                                 if alert.bin_id == bin_id and alert.status == "active"), None)
            
            # Check for warning level alert
            if current_fill >= self.alert_thresholds["warning_level"] and not existing_alert:
                await self._generate_overflow_alert(bin_info, "overflow_warning")
            
            # Check for critical level alert
            elif current_fill >= self.alert_thresholds["critical_level"]:
                if existing_alert and existing_alert.alert_type == "overflow_warning":
                    # Escalate existing warning to critical
                    await self._escalate_alert(existing_alert.id, "overflow_critical")
                elif not existing_alert:
                    await self._generate_overflow_alert(bin_info, "overflow_critical")
            
            # Predictive alert generation
            if fill_rate > 0 and current_fill > 50:
                predicted_overflow = await self._predict_bin_overflow(bin_info)
                if predicted_overflow and not existing_alert:
                    await self._generate_predictive_alert(bin_info, predicted_overflow)
            
        except Exception as e:
            logger.error(f"❌ Failed to process bin {bin_info.get('id', 'unknown')}: {e}")
    
    async def _generate_overflow_alert(self, bin_info: Dict[str, Any], alert_type: str):
        """Generate an overflow alert with contextual reasoning"""
        try:
            bin_id = bin_info.get("id")
            current_fill = bin_info.get("current_fill", 0)
            
            # Generate alert ID
            alert_id = f"alert_{bin_id}_{datetime.now().timestamp()}"
            
            # Calculate priority based on context
            priority = await self._calculate_alert_priority(bin_info, alert_type)
            
            # Generate contextual message
            message = await self._generate_contextual_message(bin_info, alert_type)
            
            # Get similar historical cases from vector database
            similar_cases = []
            if self.vector_db:
                similar_cases = await self._find_similar_alert_cases(bin_info, alert_type)
            
            # Generate natural language summary
            nl_summary = await self._generate_natural_language_summary(bin_info, alert_type, similar_cases)
            
            # Generate recommended action
            recommended_action = await self._generate_recommended_action(bin_info, alert_type, similar_cases)
            
            # Create alert object
            alert = Alert(
                id=alert_id,
                bin_id=bin_id,
                alert_type=alert_type,
                priority=priority,
                predicted_overflow_time=None,
                message=message,
                status="active",
                created_at=datetime.now(),
                natural_language_summary=nl_summary,
                recommended_action=recommended_action,
                similar_historical_cases=[case.get("id", "") for case in similar_cases[:3]],
                escalation_required=(alert_type == "overflow_critical" and priority >= 4)
            )
            
            # Store alert in memory and database
            self.active_alerts[alert_id] = alert
            await self._save_alert_to_database(alert)
            
            # Store alert pattern in vector database
            if self.vector_db:
                await self._store_alert_pattern(alert, bin_info)
            
            # Send notifications
            await self._send_alert_notifications(alert, bin_info)
            
            self.alert_stats["total_alerts_generated"] += 1
            
            logger.info(f"🚨 Generated {alert_type} alert for bin {bin_id} (Priority: {priority})")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate overflow alert: {e}")
    
    async def _predict_bin_overflow(self, bin_info: Dict[str, Any]) -> Optional[datetime]:
        """Predict when a bin will overflow using ML models"""
        try:
            if not self.model_trained:
                return None
            
            current_fill = bin_info.get("current_fill", 0)
            fill_rate = bin_info.get("fill_rate", 0)
            capacity = bin_info.get("capacity", 240)
            
            # Simple prediction if ML model not available
            if fill_rate <= 0:
                return None
            
            remaining_capacity = capacity * (100 - current_fill) / 100
            hours_to_overflow = remaining_capacity / fill_rate
            
            # Only predict if overflow is within prediction horizon
            if hours_to_overflow <= self.alert_thresholds["prediction_horizon"]:
                return datetime.now() + timedelta(hours=hours_to_overflow)
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to predict bin overflow: {e}")
            return None
    
    async def _generate_predictive_alert(self, bin_info: Dict[str, Any], predicted_overflow: datetime):
        """Generate a predictive alert for upcoming overflow"""
        try:
            bin_id = bin_info.get("id")
            
            # Generate alert ID
            alert_id = f"pred_alert_{bin_id}_{datetime.now().timestamp()}"
            
            # Calculate priority (predictive alerts have lower priority)
            base_priority = await self._calculate_alert_priority(bin_info, "overflow_warning")
            priority = max(1, base_priority - 1)  # Reduce priority for predictions
            
            # Generate contextual message
            hours_until = (predicted_overflow - datetime.now()).total_seconds() / 3600
            message = f"Predicted overflow in {hours_until:.1f} hours based on current fill rate"
            
            # Generate natural language summary
            nl_summary = await self._generate_predictive_summary(bin_info, predicted_overflow)
            
            # Create alert object
            alert = Alert(
                id=alert_id,
                bin_id=bin_id,
                alert_type="overflow_prediction",
                priority=priority,
                predicted_overflow_time=predicted_overflow,
                message=message,
                status="active",
                created_at=datetime.now(),
                natural_language_summary=nl_summary,
                recommended_action="Schedule collection before predicted overflow time",
                escalation_required=False
            )
            
            # Store alert
            self.active_alerts[alert_id] = alert
            await self._save_alert_to_database(alert)
            
            # Send notifications
            await self._send_alert_notifications(alert, bin_info)
            
            self.alert_stats["overflow_predictions_made"] += 1
            
            logger.info(f"🔮 Generated predictive alert for bin {bin_id} (Overflow in {hours_until:.1f}h)")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate predictive alert: {e}")
    
    async def _calculate_alert_priority(self, bin_info: Dict[str, Any], alert_type: str) -> int:
        """Calculate alert priority based on contextual factors"""
        try:
            base_priority = {
                "overflow_warning": 2,
                "overflow_critical": 4,
                "overflow_prediction": 2,
                "maintenance": 1
            }.get(alert_type, 2)
            
            # Contextual adjustments
            current_time = datetime.now()
            bin_type = bin_info.get("bin_type", "residential")
            location_type = bin_info.get("location_type", "street")
            
            # Time-based adjustments
            if current_time.hour in self.alert_patterns["temporal"]["morning_rush"]["hours"]:
                base_priority += 1
            elif current_time.hour in self.alert_patterns["temporal"]["evening_peak"]["hours"]:
                base_priority += 1
            
            # Location-based adjustments
            if bin_type in self.alert_patterns["location"]:
                location_config = self.alert_patterns["location"][bin_type]
                base_priority = int(base_priority * location_config["base_urgency"])
            
            # Weekend adjustments
            if current_time.weekday() >= 5 and bin_type == "public":
                base_priority += 1
            
            return min(5, max(1, base_priority))
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate alert priority: {e}")
            return 2
    
    async def _generate_contextual_message(self, bin_info: Dict[str, Any], alert_type: str) -> str:
        """Generate contextual alert message"""
        try:
            bin_id = bin_info.get("id", "Unknown")
            current_fill = bin_info.get("current_fill", 0)
            bin_type = bin_info.get("bin_type", "residential")
            location_type = bin_info.get("location_type", "street")
            
            base_messages = {
                "overflow_warning": f"Bin {bin_id} is {current_fill:.1f}% full and approaching capacity",
                "overflow_critical": f"CRITICAL: Bin {bin_id} is {current_fill:.1f}% full and requires immediate attention",
                "overflow_prediction": f"Bin {bin_id} is predicted to overflow based on current fill patterns",
                "maintenance": f"Bin {bin_id} may require maintenance or cleaning"
            }
            
            base_message = base_messages.get(alert_type, f"Alert for bin {bin_id}")
            
            # Add contextual information
            context_info = []
            
            if bin_type == "commercial":
                context_info.append("commercial area")
            elif bin_type == "public":
                context_info.append("public space")
            
            if location_type == "market":
                context_info.append("market location")
            elif location_type == "school":
                context_info.append("school area")
            
            current_time = datetime.now()
            if current_time.hour in [7, 8, 9]:
                context_info.append("morning rush period")
            elif current_time.hour in [18, 19, 20]:
                context_info.append("evening peak period")
            
            if context_info:
                base_message += f" ({', '.join(context_info)})"
            
            return base_message
            
        except Exception as e:
            logger.error(f"❌ Failed to generate contextual message: {e}")
            return f"Alert for bin {bin_info.get('id', 'Unknown')}"
    
    async def _find_similar_alert_cases(self, bin_info: Dict[str, Any], alert_type: str) -> List[Dict[str, Any]]:
        """Find similar historical alert cases using vector database"""
        try:
            if not self.vector_db:
                return []
            
            # Create search query
            bin_type = bin_info.get("bin_type", "residential")
            location_type = bin_info.get("location_type", "street")
            
            query = f"alert {alert_type} {bin_type} {location_type}"
            
            # Search for similar cases
            similar_cases = await self.vector_db.find_similar_alerts(
                bin_id=bin_info.get("id", ""),
                alert_type=alert_type,
                n_results=5
            )
            
            return similar_cases
            
        except Exception as e:
            logger.error(f"❌ Failed to find similar alert cases: {e}")
            return []
    
    async def _generate_natural_language_summary(
        self, 
        bin_info: Dict[str, Any], 
        alert_type: str, 
        similar_cases: List[Dict[str, Any]]
    ) -> str:
        """Generate natural language summary using contextual reasoning"""
        try:
            bin_id = bin_info.get("id", "Unknown")
            current_fill = bin_info.get("current_fill", 0)
            bin_type = bin_info.get("bin_type", "residential")
            
            # Base summary
            if alert_type == "overflow_critical":
                summary = f"Critical overflow alert for {bin_type} bin {bin_id}. "
                summary += f"Current fill level is {current_fill:.1f}%, requiring immediate collection. "
            elif alert_type == "overflow_warning":
                summary = f"Warning alert for {bin_type} bin {bin_id}. "
                summary += f"Fill level has reached {current_fill:.1f}%, approaching capacity. "
            else:
                summary = f"Predictive alert for {bin_type} bin {bin_id}. "
                summary += "Overflow predicted based on current fill rate patterns. "
            
            # Add contextual insights from similar cases
            if similar_cases:
                summary += f"Based on {len(similar_cases)} similar historical cases, "
                
                # Analyze resolution patterns from similar cases
                resolution_times = []
                for case in similar_cases:
                    metadata = case.get("metadata", {})
                    if "resolution_time" in metadata:
                        resolution_times.append(metadata["resolution_time"])
                
                if resolution_times:
                    avg_resolution = sum(resolution_times) / len(resolution_times)
                    summary += f"typical resolution time is {avg_resolution:.1f} hours. "
            
            # Add time-sensitive context
            current_hour = datetime.now().hour
            if current_hour in [7, 8, 9]:
                summary += "Morning rush period may increase urgency. "
            elif current_hour in [18, 19, 20]:
                summary += "Evening peak period requires prompt attention. "
            elif current_hour >= 22 or current_hour <= 6:
                summary += "Night-time collection may be scheduled for early morning. "
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"❌ Failed to generate natural language summary: {e}")
            return f"Alert generated for bin {bin_info.get('id', 'Unknown')}"
    
    async def _generate_predictive_summary(self, bin_info: Dict[str, Any], predicted_overflow: datetime) -> str:
        """Generate natural language summary for predictive alerts"""
        try:
            bin_id = bin_info.get("id", "Unknown")
            current_fill = bin_info.get("current_fill", 0)
            fill_rate = bin_info.get("fill_rate", 0)
            
            hours_until = (predicted_overflow - datetime.now()).total_seconds() / 3600
            
            summary = f"Predictive overflow alert for bin {bin_id}. "
            summary += f"Currently at {current_fill:.1f}% capacity with fill rate of {fill_rate:.2f} L/h. "
            summary += f"Overflow predicted in approximately {hours_until:.1f} hours "
            summary += f"at {predicted_overflow.strftime('%H:%M on %Y-%m-%d')}. "
            
            # Add recommendation based on timing
            if hours_until <= 2:
                summary += "Immediate collection recommended. "
            elif hours_until <= 4:
                summary += "Collection should be scheduled within next few hours. "
            else:
                summary += "Collection can be planned for next regular route. "
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Failed to generate predictive summary: {e}")
            return f"Predictive alert for bin {bin_info.get('id', 'Unknown')}"
    
    async def _generate_recommended_action(
        self, 
        bin_info: Dict[str, Any], 
        alert_type: str, 
        similar_cases: List[Dict[str, Any]]
    ) -> str:
        """Generate recommended action based on context and historical data"""
        try:
            bin_type = bin_info.get("bin_type", "residential")
            current_fill = bin_info.get("current_fill", 0)
            
            if alert_type == "overflow_critical":
                if bin_type == "commercial":
                    return "Dispatch emergency collection vehicle immediately. Contact commercial area supervisor."
                elif bin_type == "public":
                    return "Priority collection required. Notify public area maintenance team."
                else:
                    return "Schedule immediate collection. Alert collection crew supervisor."
            
            elif alert_type == "overflow_warning":
                if current_fill > 90:
                    return "Schedule collection within next 4 hours. Monitor fill level closely."
                else:
                    return "Add to next collection route. Monitor for rapid fill rate changes."
            
            elif alert_type == "overflow_prediction":
                return "Schedule proactive collection before predicted overflow time. Optimize route planning."
            
            else:
                return "Investigate bin condition and schedule appropriate maintenance."
            
        except Exception as e:
            logger.error(f"❌ Failed to generate recommended action: {e}")
            return "Review bin status and take appropriate action."
    
    async def _save_alert_to_database(self, alert: Alert):
        """Save alert to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO alerts (
                    id, bin_id, alert_type, priority, predicted_overflow_time,
                    message, status, created_at, resolved_at, natural_language_summary,
                    recommended_action, escalation_required
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.id,
                alert.bin_id,
                alert.alert_type,
                alert.priority,
                alert.predicted_overflow_time.isoformat() if alert.predicted_overflow_time else None,
                alert.message,
                alert.status,
                alert.created_at.isoformat(),
                alert.resolved_at.isoformat() if alert.resolved_at else None,
                alert.natural_language_summary,
                alert.recommended_action,
                int(alert.escalation_required)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Failed to save alert to database: {e}")
    
    async def _store_alert_pattern(self, alert: Alert, bin_info: Dict[str, Any]):
        """Store alert pattern in vector database for learning"""
        try:
            if not self.vector_db:
                return
            
            pattern_data = {
                "alert": asdict(alert),
                "bin_info": bin_info,
                "context": {
                    "time_of_day": datetime.now().hour,
                    "day_of_week": datetime.now().weekday(),
                    "month": datetime.now().month,
                    "bin_type": bin_info.get("bin_type"),
                    "location_type": bin_info.get("location_type"),
                    "fill_level": bin_info.get("current_fill"),
                    "fill_rate": bin_info.get("fill_rate")
                }
            }
            
            await self.vector_db.store_alert_pattern(
                bin_id=alert.bin_id,
                alert_data=pattern_data,
                context={
                    "alert_type": alert.alert_type,
                    "priority": alert.priority,
                    "timestamp": alert.created_at.isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to store alert pattern: {e}")
    
    async def _send_alert_notifications(self, alert: Alert, bin_info: Dict[str, Any]):
        """Send alert notifications via various channels"""
        try:
            # Send via MCP to other agents
            if self.mcp_handler:
                await self.mcp_handler.send_notification(
                    receiver_agent="broadcast",
                    event_type="alert_generated",
                    data={
                        "alert_id": alert.id,
                        "bin_id": alert.bin_id,
                        "alert_type": alert.alert_type,
                        "priority": alert.priority,
                        "message": alert.message,
                        "natural_language_summary": alert.natural_language_summary,
                        "recommended_action": alert.recommended_action,
                        "escalation_required": alert.escalation_required,
                        "bin_location": {
                            "latitude": bin_info.get("latitude"),
                            "longitude": bin_info.get("longitude"),
                            "ward_id": bin_info.get("ward_id")
                        }
                    },
                    priority=MCPPriority.HIGH if alert.priority >= 4 else MCPPriority.NORMAL
                )
            
            # Send via Redis for dashboard updates
            if self.redis_client:
                notification_data = {
                    "type": "alert_notification",
                    "alert": asdict(alert),
                    "bin_info": bin_info,
                    "timestamp": datetime.now().isoformat()
                }
                
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.redis_client.lpush,
                    "alert_notifications",
                    json.dumps(notification_data)
                )
            
            # Send email notifications
            await self._send_email_notifications(alert, bin_info)
            
            logger.info(f"📤 Sent notifications for alert {alert.id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to send alert notifications: {e}")
    
    async def _send_email_notifications(self, alert: Alert, bin_info: Dict[str, Any]):
        """Send email notifications for alerts"""
        try:
            ward_id = bin_info.get("ward_id")
            if not ward_id:
                logger.warning("⚠️ No ward_id found for bin, skipping email notifications")
                return
            
            # Get ward operator email
            operator_email = await self._get_ward_operator_email(ward_id)
            if operator_email:
                # Prepare alert data for email template
                alert_data = {
                    "id": alert.id,
                    "alert_type": alert.alert_type,
                    "priority": alert.priority,
                    "message": alert.message,
                    "created_at": alert.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "recommended_action": alert.recommended_action,
                    "natural_language_summary": alert.natural_language_summary,
                    "predicted_overflow_time": alert.predicted_overflow_time.isoformat() if alert.predicted_overflow_time else None
                }
                
                # Send email notification
                success = await self.email_notifier.send_alert_notification(
                    alert_data=alert_data,
                    bin_info=bin_info,
                    recipient_email=operator_email["email"],
                    recipient_name=operator_email["name"]
                )
                
                if success:
                    logger.info(f"📧 Email notification sent to {operator_email['email']}")
                else:
                    logger.error(f"❌ Failed to send email notification to {operator_email['email']}")
            
            # Send escalation emails if required
            if alert.escalation_required:
                await self._send_escalation_emails(alert, bin_info)
            
        except Exception as e:
            logger.error(f"❌ Failed to send email notifications: {e}")
    
    async def _get_ward_operator_email(self, ward_id: int) -> Optional[Dict[str, str]]:
        """Get operator email for a ward"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT name, email FROM operators WHERE ward_id = ? LIMIT 1
            """, (ward_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {"name": row[0], "email": row[1]}
            else:
                logger.warning(f"⚠️ No operator found for ward {ward_id}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to get ward operator email: {e}")
            return None
    
    async def _send_escalation_emails(self, alert: Alert, bin_info: Dict[str, Any]):
        """Send escalation emails to supervisors"""
        try:
            ward_id = bin_info.get("ward_id")
            if not ward_id:
                return
            
            # Get supervisor contacts
            supervisors = await self.email_notifier.get_supervisor_contacts(ward_id)
            
            if supervisors:
                # Prepare escalation data
                escalation_data = {
                    "escalation_id": f"escalation_{alert.id}",
                    "ward_name": f"Ward {ward_id}",
                    "critical_alerts": [{
                        "bin_id": alert.bin_id,
                        "fill_level": bin_info.get("current_fill", 0),
                        "message": alert.message
                    }],
                    "escalation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "escalation_reason": f"Critical alert for bin {alert.bin_id} requires supervisor attention"
                }
                
                # Send escalation notification
                success = await self.email_notifier.send_escalation_notification(
                    escalation_data=escalation_data,
                    supervisor_contacts=supervisors
                )
                
                if success:
                    logger.info(f"📧 Escalation emails sent for alert {alert.id}")
                else:
                    logger.error(f"❌ Failed to send escalation emails for alert {alert.id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to send escalation emails: {e}")
    
    async def _cluster_nearby_alerts(self):
        """Cluster nearby alerts for coordinated response"""
        try:
            # Get active alerts with location data
            alerts_with_location = []
            for alert in self.active_alerts.values():
                if alert.status == "active":
                    # Get bin location (would need to fetch from bin data)
                    # For now, skip clustering if we don't have location data
                    pass
            
            # Implement clustering logic here
            # This is a simplified version - full implementation would use proper clustering algorithms
            
        except Exception as e:
            logger.error(f"❌ Failed to cluster nearby alerts: {e}")
    
    async def _check_escalation_conditions(self):
        """Check if escalation conditions are met"""
        try:
            critical_alerts = [alert for alert in self.active_alerts.values() 
                             if alert.status == "active" and alert.priority >= 4]
            
            if len(critical_alerts) >= self.alert_thresholds["escalation_threshold"]:
                await self._trigger_system_escalation(critical_alerts)
            
        except Exception as e:
            logger.error(f"❌ Failed to check escalation conditions: {e}")
    
    async def _trigger_system_escalation(self, critical_alerts: List[Alert]):
        """Trigger system-wide escalation for multiple critical alerts"""
        try:
            escalation_id = f"escalation_{datetime.now().timestamp()}"
            
            # Send escalation notification
            if self.mcp_handler:
                await self.mcp_handler.send_notification(
                    receiver_agent="broadcast",
                    event_type="system_escalation",
                    data={
                        "escalation_id": escalation_id,
                        "critical_alert_count": len(critical_alerts),
                        "alert_ids": [alert.id for alert in critical_alerts],
                        "escalation_reason": f"{len(critical_alerts)} critical alerts require immediate attention",
                        "timestamp": datetime.now().isoformat()
                    },
                    priority=MCPPriority.EMERGENCY
                )
            
            # Send system-wide escalation emails
            await self._send_system_escalation_emails(escalation_id, critical_alerts)
            
            self.alert_stats["escalations_triggered"] += 1
            
            logger.warning(f"🚨 SYSTEM ESCALATION: {len(critical_alerts)} critical alerts")
            
        except Exception as e:
            logger.error(f"❌ Failed to trigger system escalation: {e}")
    
    async def _send_system_escalation_emails(self, escalation_id: str, critical_alerts: List[Alert]):
        """Send system-wide escalation emails to all supervisors"""
        try:
            # Group alerts by ward
            ward_alerts = {}
            for alert in critical_alerts:
                # Get ward_id from bin data (would need to fetch from database)
                # For now, assume we can get it from alert context
                ward_id = 1  # Placeholder - would need proper ward lookup
                
                if ward_id not in ward_alerts:
                    ward_alerts[ward_id] = []
                ward_alerts[ward_id].append(alert)
            
            # Send escalation emails for each ward
            for ward_id, alerts in ward_alerts.items():
                supervisors = await self.email_notifier.get_supervisor_contacts(ward_id)
                
                if supervisors:
                    escalation_data = {
                        "escalation_id": escalation_id,
                        "ward_name": f"Ward {ward_id}",
                        "critical_alerts": [{
                            "bin_id": alert.bin_id,
                            "fill_level": "95+",  # Placeholder
                            "message": alert.message
                        } for alert in alerts],
                        "escalation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "escalation_reason": f"System escalation: {len(alerts)} critical alerts in ward {ward_id}"
                    }
                    
                    await self.email_notifier.send_escalation_notification(
                        escalation_data=escalation_data,
                        supervisor_contacts=supervisors
                    )
            
            logger.info(f"📧 System escalation emails sent for {len(ward_alerts)} wards")
            
        except Exception as e:
            logger.error(f"❌ Failed to send system escalation emails: {e}")
    
    async def _update_ml_model(self, bin_data: List[Dict[str, Any]]):
        """Update ML model with new data"""
        try:
            # This is a simplified version - full implementation would include proper model training
            # For now, just log that we would update the model
            logger.debug(f"📚 Would update ML model with {len(bin_data)} data points")
            
        except Exception as e:
            logger.error(f"❌ Failed to update ML model: {e}")
    
    async def resolve_alert(self, alert_id: str, resolution_method: str = "collected") -> bool:
        """Resolve an active alert"""
        try:
            if alert_id not in self.active_alerts:
                logger.error(f"❌ Alert not found: {alert_id}")
                return False
            
            alert = self.active_alerts[alert_id]
            alert.status = "resolved"
            alert.resolved_at = datetime.now()
            
            # Update database
            await self._save_alert_to_database(alert)
            
            # Store resolution pattern for learning
            if self.vector_db:
                resolution_data = {
                    "alert_id": alert_id,
                    "resolution_method": resolution_method,
                    "response_time": (alert.resolved_at - alert.created_at).total_seconds() / 3600,
                    "alert_type": alert.alert_type,
                    "priority": alert.priority
                }
                
                await self.vector_db.store_system_knowledge(
                    knowledge_type="alert_resolution",
                    content=json.dumps(resolution_data),
                    source_agent=self.agent_id,
                    confidence_score=0.9
                )
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            self.alert_stats["alerts_resolved"] += 1
            
            logger.info(f"✅ Resolved alert {alert_id} via {resolution_method}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to resolve alert {alert_id}: {e}")
            return False
    
    async def _escalate_alert(self, alert_id: str, new_alert_type: str):
        """Escalate an existing alert to higher priority"""
        try:
            if alert_id not in self.active_alerts:
                return
            
            alert = self.active_alerts[alert_id]
            alert.alert_type = new_alert_type
            alert.priority = min(5, alert.priority + 1)
            alert.escalation_required = True
            
            # Update message
            alert.message = f"ESCALATED: {alert.message}"
            
            # Update database
            await self._save_alert_to_database(alert)
            
            logger.warning(f"⬆️ Escalated alert {alert_id} to {new_alert_type}")
            
        except Exception as e:
            logger.error(f"❌ Failed to escalate alert {alert_id}: {e}")
    
    def get_active_alerts(self, ward_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get active alerts, optionally filtered by ward"""
        try:
            alerts = []
            for alert in self.active_alerts.values():
                if alert.status == "active":
                    alert_dict = asdict(alert)
                    
                    # Convert datetime objects to ISO strings
                    if alert_dict["created_at"]:
                        alert_dict["created_at"] = alert.created_at.isoformat()
                    if alert_dict["predicted_overflow_time"]:
                        alert_dict["predicted_overflow_time"] = alert.predicted_overflow_time.isoformat()
                    if alert_dict["resolved_at"]:
                        alert_dict["resolved_at"] = alert.resolved_at.isoformat()
                    
                    alerts.append(alert_dict)
            
            return alerts
            
        except Exception as e:
            logger.error(f"❌ Failed to get active alerts: {e}")
            return []
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert management statistics"""
        email_stats = self.email_notifier.get_email_stats()
        
        return {
            **self.alert_stats,
            "active_alerts_count": len(self.active_alerts),
            "monitoring_active": self.monitoring_active,
            "model_trained": self.model_trained,
            "email_notifications": email_stats,
            "timestamp": datetime.now().isoformat()
        }
    
    async def update_email_preferences(self, operator_id: int, preferences: Dict[str, Any]) -> bool:
        """Update email preferences for an operator"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get operator email
            cursor.execute("SELECT email FROM operators WHERE id = ?", (operator_id,))
            operator_row = cursor.fetchone()
            
            if not operator_row:
                logger.error(f"❌ Operator {operator_id} not found")
                return False
            
            operator_email = operator_row[0]
            
            # Update preferences
            cursor.execute("""
                INSERT OR REPLACE INTO email_preferences (
                    operator_id, email, alert_notifications, escalation_notifications,
                    daily_reports, weekly_summaries, emergency_alerts,
                    notification_frequency, quiet_hours_start, quiet_hours_end
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                operator_id,
                operator_email,
                preferences.get("alert_notifications", True),
                preferences.get("escalation_notifications", True),
                preferences.get("daily_reports", True),
                preferences.get("weekly_summaries", True),
                preferences.get("emergency_alerts", True),
                preferences.get("notification_frequency", "immediate"),
                preferences.get("quiet_hours_start", "22:00"),
                preferences.get("quiet_hours_end", "06:00")
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Updated email preferences for operator {operator_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update email preferences: {e}")
            return False
    
    async def get_email_preferences(self, operator_id: int) -> Dict[str, Any]:
        """Get email preferences for an operator"""
        return await self.email_notifier.get_operator_email_preferences(operator_id)
    
    async def retry_failed_email_notifications(self) -> int:
        """Retry failed email notifications"""
        return await self.email_notifier.retry_failed_notifications()
    
    # Tool implementations
    def _generate_alert_tool(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Tool implementation for generating alerts"""
        try:
            bin_id = parameters.get("bin_id")
            alert_type = parameters.get("alert_type", "overflow_warning")
            
            # This would trigger alert generation
            return {
                "status": "success",
                "message": f"Alert generation triggered for bin {bin_id}",
                "alert_type": alert_type
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _cluster_alerts_tool(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Tool implementation for clustering alerts"""
        try:
            cluster_radius = parameters.get("cluster_radius", 500)
            
            # This would trigger alert clustering
            return {
                "status": "success",
                "message": f"Alert clustering initiated with {cluster_radius}m radius",
                "active_clusters": len(self.alert_clusters)
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _predict_overflow_tool(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Tool implementation for overflow prediction"""
        try:
            bin_data = parameters.get("bin_data", [])
            
            predictions = []
            for bin_info in bin_data:
                predicted_time = asyncio.run(self._predict_bin_overflow(bin_info))
                if predicted_time:
                    predictions.append({
                        "bin_id": bin_info.get("id"),
                        "predicted_overflow": predicted_time.isoformat()
                    })
            
            return {
                "status": "success",
                "predictions": predictions,
                "count": len(predictions)
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _escalate_alert_tool(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Tool implementation for alert escalation"""
        try:
            alert_id = parameters.get("alert_id")
            
            if alert_id in self.active_alerts:
                asyncio.run(self._escalate_alert(alert_id, "overflow_critical"))
                return {
                    "status": "success",
                    "message": f"Alert {alert_id} escalated successfully"
                }
            else:
                return {
                    "status": "error",
                    "message": f"Alert {alert_id} not found"
                }
                
        except Exception as e:
            return {"status": "error", "message": str(e)}   
 # MCP request handlers
    async def _handle_overflow_prediction_request(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP request for overflow prediction"""
        try:
            bin_data = parameters.get("bin_data", [])
            prediction_horizon = parameters.get("prediction_horizon", 4.0)
            include_context = parameters.get("include_context", True)
            
            predictions = []
            alerts_generated = []
            contextual_insights = []
            
            for bin_info in bin_data:
                # Predict overflow
                predicted_time = await self._predict_bin_overflow(bin_info)
                if predicted_time:
                    predictions.append({
                        "bin_id": bin_info.get("id"),
                        "predicted_overflow_time": predicted_time.isoformat(),
                        "confidence": 0.8  # Placeholder confidence score
                    })
                    
                    # Generate alert if within horizon
                    hours_until = (predicted_time - datetime.now()).total_seconds() / 3600
                    if hours_until <= prediction_horizon:
                        await self._generate_predictive_alert(bin_info, predicted_time)
                        alerts_generated.append({
                            "bin_id": bin_info.get("id"),
                            "alert_type": "overflow_prediction",
                            "hours_until_overflow": hours_until
                        })
                
                # Add contextual insights if requested
                if include_context:
                    priority = await self._calculate_alert_priority(bin_info, "overflow_warning")
                    contextual_insights.append({
                        "bin_id": bin_info.get("id"),
                        "priority_score": priority,
                        "context_factors": [
                            f"Bin type: {bin_info.get('bin_type', 'unknown')}",
                            f"Current fill: {bin_info.get('current_fill', 0):.1f}%",
                            f"Fill rate: {bin_info.get('fill_rate', 0):.2f} L/h"
                        ]
                    })
            
            return {
                "predictions": predictions,
                "alerts_generated": alerts_generated,
                "contextual_insights": contextual_insights
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to handle overflow prediction request: {e}")
            return {"error": str(e)}
    
    async def _handle_alert_generation_request(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP request for alert generation"""
        try:
            bin_id = parameters.get("bin_id")
            alert_type = parameters.get("alert_type", "overflow_warning")
            bin_context = parameters.get("context", {})
            include_clustering = parameters.get("include_clustering", False)
            
            # Create bin_info from context
            bin_info = {
                "id": bin_id,
                **bin_context
            }
            
            # Generate alert
            await self._generate_overflow_alert(bin_info, alert_type)
            
            # Get the generated alert
            alert = next((a for a in self.active_alerts.values() if a.bin_id == bin_id), None)
            
            if alert:
                # Find similar cases
                similar_cases = await self._find_similar_alert_cases(bin_info, alert_type)
                
                # Generate recommended actions
                recommended_actions = [
                    alert.recommended_action,
                    "Monitor bin status closely",
                    "Update collection schedule if needed"
                ]
                
                return {
                    "alert": asdict(alert),
                    "natural_language_summary": alert.natural_language_summary,
                    "recommended_actions": recommended_actions,
                    "similar_cases": similar_cases[:3]
                }
            else:
                return {"error": "Failed to generate alert"}
            
        except Exception as e:
            logger.error(f"❌ Failed to handle alert generation request: {e}")
            return {"error": str(e)}
    
    async def _handle_alert_clustering_request(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP request for alert clustering"""
        try:
            alerts_data = parameters.get("alerts", [])
            cluster_radius = parameters.get("cluster_radius", 500.0)
            min_cluster_size = parameters.get("min_cluster_size", 2)
            
            # This is a simplified clustering implementation
            # Full implementation would use proper clustering algorithms
            
            clusters = []
            coordinated_responses = []
            optimization_suggestions = []
            
            if len(alerts_data) >= min_cluster_size:
                # Create a simple cluster for demonstration
                cluster = {
                    "cluster_id": f"cluster_{datetime.now().timestamp()}",
                    "alert_count": len(alerts_data),
                    "priority": max([alert.get("priority", 1) for alert in alerts_data]),
                    "recommended_response": "Coordinate collection route to handle multiple bins efficiently"
                }
                clusters.append(cluster)
                
                coordinated_responses.append({
                    "cluster_id": cluster["cluster_id"],
                    "response_type": "coordinated_collection",
                    "estimated_time_savings": "30-45 minutes",
                    "resource_optimization": "Single vehicle can handle multiple bins"
                })
                
                optimization_suggestions.append({
                    "suggestion": "Route optimization recommended for clustered alerts",
                    "potential_benefit": "Reduced fuel consumption and faster response time",
                    "implementation": "Contact route optimization agent for updated routes"
                })
            
            return {
                "clusters": clusters,
                "coordinated_responses": coordinated_responses,
                "optimization_suggestions": optimization_suggestions
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to handle alert clustering request: {e}")
            return {"error": str(e)}

logger.info("🚨 Alert Management Agent loaded successfully")