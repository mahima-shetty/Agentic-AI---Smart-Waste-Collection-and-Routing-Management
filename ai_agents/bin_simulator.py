"""
Bin Simulator Agent - Realistic IoT sensor simulation for demonstration and testing
Generates realistic fill level data with temporal patterns and festival variations
"""

import asyncio
import logging
import json
import random
import math
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import redis
import chromadb

from .langchain_base import BaseLangChainAgent, WasteManagementTool, AgentPromptTemplates
from .mcp_handler import MCPHandler, MCPMessageType, MCPPriority
from .vector_db import VectorDatabaseManager

logger = logging.getLogger(__name__)

@dataclass
class BinConfiguration:
    """Configuration for a simulated bin"""
    bin_id: str
    ward_id: int
    latitude: float
    longitude: float
    capacity: float  # in liters
    bin_type: str  # "residential", "commercial", "industrial", "public"
    base_fill_rate: float  # liters per hour under normal conditions
    location_type: str  # "street", "market", "residential", "office", "school"
    seasonal_factor: float  # multiplier for seasonal variations
    festival_sensitivity: float  # how much festivals affect this bin (0-2.0)

@dataclass
class BinState:
    """Current state of a simulated bin"""
    bin_id: str
    current_fill: float  # percentage (0-100)
    fill_rate: float  # current fill rate in liters per hour
    last_collection: datetime
    last_update: datetime
    status: str  # "normal", "warning", "critical", "maintenance"
    temperature: float  # affects decomposition rate
    humidity: float  # affects waste volume
    predicted_overflow: Optional[datetime] = None

class BinSimulatorAgent(BaseLangChainAgent):
    """
    Bin Simulator Agent for realistic IoT sensor simulation
    Generates realistic fill level data with temporal patterns and festival variations
    """
    
    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        vector_db: Optional[VectorDatabaseManager] = None,
        mcp_handler: Optional[MCPHandler] = None
    ):
        super().__init__(
            agent_id="bin_simulator",
            agent_type="bin_simulation",
            model_name="gpt-3.5-turbo",
            temperature=0.1
        )
        
        self.redis_client = redis_client
        self.vector_db = vector_db
        self.mcp_handler = mcp_handler
        
        # Simulation state
        self.bin_configurations: Dict[str, BinConfiguration] = {}
        self.bin_states: Dict[str, BinState] = {}
        self.simulation_running = False
        self.simulation_speed = 1.0  # 1.0 = real time, 10.0 = 10x faster
        
        # Temporal patterns
        self.time_patterns = {
            "residential": {
                "morning_peak": (7, 9, 1.5),  # (start_hour, end_hour, multiplier)
                "evening_peak": (18, 21, 1.8),
                "night_low": (23, 6, 0.3)
            },
            "commercial": {
                "business_hours": (9, 18, 2.0),
                "lunch_peak": (12, 14, 2.5),
                "night_low": (20, 8, 0.2)
            },
            "public": {
                "morning_rush": (8, 10, 1.8),
                "afternoon_rush": (16, 19, 2.2),
                "weekend_high": (10, 16, 1.6)
            }
        }
        
        # Festival and event patterns
        self.festival_patterns = {
            "ganesh_chaturthi": {"duration_days": 11, "multiplier": 3.0, "month": 8},
            "diwali": {"duration_days": 5, "multiplier": 2.5, "month": 10},
            "navratri": {"duration_days": 9, "multiplier": 2.2, "month": 9},
            "holi": {"duration_days": 2, "multiplier": 2.8, "month": 3},
            "eid": {"duration_days": 3, "multiplier": 2.0, "month": 5},
            "christmas": {"duration_days": 3, "multiplier": 1.8, "month": 12}
        }
        
        # Weather impact patterns
        self.weather_patterns = {
            "monsoon": {"months": [6, 7, 8, 9], "fill_rate_multiplier": 0.7, "humidity_increase": 20},
            "summer": {"months": [3, 4, 5], "fill_rate_multiplier": 1.2, "temperature_increase": 5},
            "winter": {"months": [12, 1, 2], "fill_rate_multiplier": 0.9, "temperature_decrease": 3}
        }
        
        # Statistics
        self.simulation_stats = {
            "total_bins": 0,
            "active_bins": 0,
            "data_points_generated": 0,
            "overflows_predicted": 0,
            "collections_simulated": 0,
            "simulation_start_time": None
        }
        
        logger.info("🗑️ Bin Simulator Agent initialized")
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the bin simulator agent"""
        return AgentPromptTemplates.BIN_SIMULATION_SYSTEM
    
    def get_tools(self) -> List[WasteManagementTool]:
        """Get tools available to the bin simulator agent"""
        tools = []
        
        # Bin configuration tool
        tools.append(WasteManagementTool(
            name="configure_bins",
            description="Configure bins for simulation with realistic parameters",
            func=self._configure_bins_tool
        ))
        
        # Simulation control tool
        tools.append(WasteManagementTool(
            name="control_simulation",
            description="Start, stop, or modify simulation parameters",
            func=self._control_simulation_tool
        ))
        
        # Data generation tool
        tools.append(WasteManagementTool(
            name="generate_bin_data",
            description="Generate realistic bin fill level data",
            func=self._generate_bin_data_tool
        ))
        
        # Pattern analysis tool
        tools.append(WasteManagementTool(
            name="analyze_patterns",
            description="Analyze and store bin behavior patterns",
            func=self._analyze_patterns_tool
        ))
        
        return tools
    
    async def initialize_simulation(self, config: Dict[str, Any]) -> bool:
        """Initialize the bin simulation with configuration"""
        try:
            logger.info("🚀 Initializing Bin Simulator Agent...")
            
            # Initialize LangChain agent
            if not self.initialize_agent():
                return False
            
            # Setup MCP capabilities if available
            if self.mcp_handler:
                await self._setup_mcp_capabilities()
            
            # Load or create bin configurations
            await self._initialize_bin_configurations(config)
            
            # Initialize bin states
            await self._initialize_bin_states()
            
            # Store initial patterns in vector database
            if self.vector_db:
                await self._store_initial_patterns()
            
            self.simulation_stats["simulation_start_time"] = datetime.now().isoformat()
            
            logger.info("✅ Bin Simulator Agent initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Bin Simulator Agent: {e}")
            return False
    
    async def _setup_mcp_capabilities(self):
        """Setup MCP capabilities for bin simulation"""
        try:
            # Register bin data generation capability
            await self.mcp_handler.register_capability(
                name="generate_bin_data",
                description="Generate realistic bin fill level data with temporal patterns",
                input_schema={
                    "type": "object",
                    "properties": {
                        "ward_ids": {"type": "array", "items": {"type": "integer"}},
                        "duration_hours": {"type": "number"},
                        "include_festivals": {"type": "boolean"}
                    }
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "bin_data": {"type": "array"},
                        "timestamp": {"type": "string"},
                        "patterns_detected": {"type": "array"}
                    }
                },
                handler=self._handle_generate_bin_data_request
            )
            
            # Register simulation control capability
            await self.mcp_handler.register_capability(
                name="control_simulation",
                description="Control bin simulation parameters and state",
                input_schema={
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["start", "stop", "pause", "configure"]},
                        "parameters": {"type": "object"}
                    }
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "message": {"type": "string"},
                        "simulation_stats": {"type": "object"}
                    }
                },
                handler=self._handle_simulation_control_request
            )
            
            logger.info("🔄 MCP capabilities registered for Bin Simulator")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup MCP capabilities: {e}")
    
    async def _initialize_bin_configurations(self, config: Dict[str, Any]):
        """Initialize bin configurations for multiple wards"""
        try:
            # Load existing bin data from markers.json if available
            existing_bins = config.get("existing_bins", [])
            
            # Create configurations for Mumbai's 24 wards
            ward_configs = self._generate_ward_configurations()
            
            for ward_id in range(1, 25):  # Mumbai has 24 wards
                ward_config = ward_configs.get(ward_id, {})
                bins_per_ward = ward_config.get("bins_per_ward", 4)
                
                # Generate bins for this ward
                for bin_index in range(bins_per_ward):
                    bin_id = f"BIN_{ward_id:02d}_{bin_index:03d}"
                    
                    # Use existing bin data if available, otherwise generate
                    existing_bin = next((b for b in existing_bins if b.get("name") == f"Dustbin {bin_index + 1}"), None)
                    
                    if existing_bin:
                        lat, lon = existing_bin["latitude"], existing_bin["longitude"]
                    else:
                        # Generate realistic coordinates within ward boundaries
                        lat, lon = self._generate_ward_coordinates(ward_id)
                    
                    # Determine bin type and characteristics
                    bin_type, location_type = self._determine_bin_characteristics(ward_id, bin_index)
                    
                    bin_config = BinConfiguration(
                        bin_id=bin_id,
                        ward_id=ward_id,
                        latitude=lat,
                        longitude=lon,
                        capacity=self._get_bin_capacity(bin_type),
                        bin_type=bin_type,
                        base_fill_rate=self._get_base_fill_rate(bin_type, location_type),
                        location_type=location_type,
                        seasonal_factor=random.uniform(0.8, 1.2),
                        festival_sensitivity=self._get_festival_sensitivity(location_type)
                    )
                    
                    self.bin_configurations[bin_id] = bin_config
            
            self.simulation_stats["total_bins"] = len(self.bin_configurations)
            logger.info(f"📍 Configured {len(self.bin_configurations)} bins across 24 wards")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize bin configurations: {e}")
            raise
    
    def _generate_ward_configurations(self) -> Dict[int, Dict[str, Any]]:
        """Generate realistic ward configurations for Mumbai"""
        ward_configs = {}
        
        # Different ward types with different characteristics (optimized for ~100 total bins)
        ward_types = {
            "commercial": {"bins_per_ward": 5, "wards": [1, 2, 3, 4, 5]},
            "residential_dense": {"bins_per_ward": 4, "wards": [6, 7, 8, 9, 10, 11]},
            "residential_medium": {"bins_per_ward": 4, "wards": [12, 13, 14, 15, 16, 17]},
            "mixed": {"bins_per_ward": 4, "wards": [18, 19, 20, 21]},
            "industrial": {"bins_per_ward": 3, "wards": [22, 23, 24]}
        }
        
        for ward_type, config in ward_types.items():
            for ward_id in config["wards"]:
                ward_configs[ward_id] = {
                    "bins_per_ward": config["bins_per_ward"],
                    "ward_type": ward_type,
                    "population_density": self._get_population_density(ward_type),
                    "commercial_ratio": self._get_commercial_ratio(ward_type)
                }
        
        return ward_configs
    
    def _generate_ward_coordinates(self, ward_id: int) -> Tuple[float, float]:
        """Generate realistic coordinates within Mumbai ward boundaries"""
        # Mumbai approximate boundaries
        mumbai_bounds = {
            "lat_min": 18.90, "lat_max": 19.30,
            "lon_min": 72.77, "lon_max": 73.00
        }
        
        # Create ward-specific areas (simplified grid approach)
        rows, cols = 6, 4  # 6x4 grid for 24 wards
        ward_index = ward_id - 1
        row = ward_index // cols
        col = ward_index % cols
        
        # Calculate ward boundaries
        lat_range = (mumbai_bounds["lat_max"] - mumbai_bounds["lat_min"]) / rows
        lon_range = (mumbai_bounds["lon_max"] - mumbai_bounds["lon_min"]) / cols
        
        ward_lat_min = mumbai_bounds["lat_min"] + row * lat_range
        ward_lat_max = ward_lat_min + lat_range
        ward_lon_min = mumbai_bounds["lon_min"] + col * lon_range
        ward_lon_max = ward_lon_min + lon_range
        
        # Generate random coordinates within ward
        latitude = random.uniform(ward_lat_min, ward_lat_max)
        longitude = random.uniform(ward_lon_min, ward_lon_max)
        
        return latitude, longitude
    
    def _determine_bin_characteristics(self, ward_id: int, bin_index: int) -> Tuple[str, str]:
        """Determine bin type and location type based on ward and position"""
        # Get ward type
        ward_configs = self._generate_ward_configurations()
        ward_config = ward_configs.get(ward_id, {"ward_type": "mixed"})
        ward_type = ward_config["ward_type"]
        
        # Determine bin type based on ward type and random distribution
        if ward_type == "commercial":
            bin_types = ["commercial", "public", "residential"]
            weights = [0.6, 0.3, 0.1]
        elif ward_type == "industrial":
            bin_types = ["industrial", "commercial", "public"]
            weights = [0.5, 0.3, 0.2]
        elif ward_type == "residential_dense":
            bin_types = ["residential", "public", "commercial"]
            weights = [0.7, 0.2, 0.1]
        elif ward_type == "residential_medium":
            bin_types = ["residential", "public", "commercial"]
            weights = [0.8, 0.15, 0.05]
        else:  # mixed
            bin_types = ["residential", "commercial", "public", "industrial"]
            weights = [0.4, 0.3, 0.2, 0.1]
        
        bin_type = random.choices(bin_types, weights=weights)[0]
        
        # Determine location type based on bin type
        location_types = {
            "residential": ["residential", "street"],
            "commercial": ["market", "office", "street"],
            "public": ["street", "park", "station"],
            "industrial": ["factory", "warehouse", "office"]
        }
        
        location_type = random.choice(location_types[bin_type])
        
        return bin_type, location_type
    
    def _get_bin_capacity(self, bin_type: str) -> float:
        """Get bin capacity in liters based on type"""
        capacities = {
            "residential": random.uniform(120, 240),  # Standard household bins
            "commercial": random.uniform(240, 1100),  # Commercial dumpsters
            "public": random.uniform(80, 200),        # Street bins
            "industrial": random.uniform(1100, 2200)  # Large industrial containers
        }
        return capacities.get(bin_type, 240)
    
    def _get_base_fill_rate(self, bin_type: str, location_type: str) -> float:
        """Get base fill rate in liters per hour"""
        base_rates = {
            "residential": {"residential": 0.8, "street": 1.2},
            "commercial": {"market": 4.5, "office": 2.8, "street": 3.2},
            "public": {"street": 2.0, "park": 1.5, "station": 3.5},
            "industrial": {"factory": 8.0, "warehouse": 3.5, "office": 2.0}
        }
        
        return base_rates.get(bin_type, {}).get(location_type, 2.0)
    
    def _get_festival_sensitivity(self, location_type: str) -> float:
        """Get festival sensitivity multiplier"""
        sensitivities = {
            "residential": 1.8,
            "market": 2.5,
            "street": 2.0,
            "park": 1.5,
            "office": 1.2,
            "factory": 1.1,
            "warehouse": 1.0,
            "station": 2.2
        }
        return sensitivities.get(location_type, 1.5)
    
    def _get_population_density(self, ward_type: str) -> str:
        """Get population density category"""
        densities = {
            "commercial": "high",
            "residential_dense": "very_high",
            "residential_medium": "medium",
            "mixed": "high",
            "industrial": "low"
        }
        return densities.get(ward_type, "medium")
    
    def _get_commercial_ratio(self, ward_type: str) -> float:
        """Get commercial activity ratio"""
        ratios = {
            "commercial": 0.8,
            "residential_dense": 0.2,
            "residential_medium": 0.1,
            "mixed": 0.5,
            "industrial": 0.6
        }
        return ratios.get(ward_type, 0.3)
    
    async def _initialize_bin_states(self):
        """Initialize current states for all configured bins"""
        try:
            current_time = datetime.now()
            
            for bin_id, config in self.bin_configurations.items():
                # Initialize with random but realistic starting conditions
                initial_fill = random.uniform(10, 40)  # Start partially filled
                
                bin_state = BinState(
                    bin_id=bin_id,
                    current_fill=initial_fill,
                    fill_rate=config.base_fill_rate,
                    last_collection=current_time - timedelta(hours=random.uniform(2, 24)),
                    last_update=current_time,
                    status="normal",
                    temperature=random.uniform(20, 35),  # Celsius
                    humidity=random.uniform(40, 80)      # Percentage
                )
                
                self.bin_states[bin_id] = bin_state
            
            self.simulation_stats["active_bins"] = len(self.bin_states)
            logger.info(f"🔄 Initialized states for {len(self.bin_states)} bins")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize bin states: {e}")
            raise
    
    async def _store_initial_patterns(self):
        """Store initial bin behavior patterns in vector database"""
        try:
            if not self.vector_db:
                return
            
            # Store bin configuration patterns
            for bin_id, config in self.bin_configurations.items():
                pattern_data = {
                    "bin_id": bin_id,
                    "configuration": asdict(config),
                    "pattern_type": "configuration",
                    "temporal_characteristics": self._analyze_temporal_characteristics(config),
                    "expected_behavior": self._predict_bin_behavior(config)
                }
                
                await self.vector_db.store_document(
                    collection_name="bin_behaviors",
                    document_id=f"config_{bin_id}",
                    content=json.dumps(pattern_data, indent=2),
                    metadata={
                        "bin_id": bin_id,
                        "ward_id": config.ward_id,
                        "pattern_type": "configuration",
                        "seasonal_factor": config.seasonal_factor,
                        "timestamp": datetime.now().isoformat()
                    }
                )
            
            logger.info("📚 Initial bin patterns stored in vector database")
            
        except Exception as e:
            logger.error(f"❌ Failed to store initial patterns: {e}")
    
    def _analyze_temporal_characteristics(self, config: BinConfiguration) -> Dict[str, Any]:
        """Analyze temporal characteristics of a bin configuration"""
        characteristics = {
            "primary_pattern": config.bin_type,
            "location_influence": config.location_type,
            "peak_hours": [],
            "low_hours": [],
            "seasonal_variation": config.seasonal_factor,
            "festival_impact": config.festival_sensitivity
        }
        
        # Determine peak and low hours based on bin type
        if config.bin_type in self.time_patterns:
            patterns = self.time_patterns[config.bin_type]
            for pattern_name, (start, end, multiplier) in patterns.items():
                if multiplier > 1.5:
                    characteristics["peak_hours"].append({"start": start, "end": end, "intensity": multiplier})
                elif multiplier < 0.5:
                    characteristics["low_hours"].append({"start": start, "end": end, "intensity": multiplier})
        
        return characteristics
    
    def _predict_bin_behavior(self, config: BinConfiguration) -> Dict[str, Any]:
        """Predict expected behavior patterns for a bin"""
        behavior = {
            "average_daily_fill": config.base_fill_rate * 24,
            "typical_collection_frequency": "daily" if config.base_fill_rate > 3 else "every_2_days",
            "overflow_risk": "high" if config.base_fill_rate > 4 else "medium" if config.base_fill_rate > 2 else "low",
            "weather_sensitivity": "high" if config.location_type in ["street", "park"] else "medium",
            "festival_impact_level": "high" if config.festival_sensitivity > 2 else "medium" if config.festival_sensitivity > 1.5 else "low"
        }
        
        return behavior
    
    async def start_simulation(self, speed_multiplier: float = 1.0) -> bool:
        """Start the bin simulation with specified speed"""
        try:
            if self.simulation_running:
                logger.warning("⚠️ Simulation already running")
                return False
            
            self.simulation_running = True
            self.simulation_speed = speed_multiplier
            
            # Start simulation loop
            asyncio.create_task(self._simulation_loop())
            
            # Notify other agents via MCP
            if self.mcp_handler:
                await self.mcp_handler.send_notification(
                    receiver_agent="broadcast",
                    event_type="simulation_started",
                    data={
                        "total_bins": len(self.bin_configurations),
                        "speed_multiplier": speed_multiplier,
                        "start_time": datetime.now().isoformat()
                    },
                    priority=MCPPriority.HIGH
                )
            
            logger.info(f"🚀 Bin simulation started with {speed_multiplier}x speed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start simulation: {e}")
            return False
    
    async def stop_simulation(self) -> bool:
        """Stop the bin simulation"""
        try:
            self.simulation_running = False
            
            # Notify other agents via MCP
            if self.mcp_handler:
                await self.mcp_handler.send_notification(
                    receiver_agent="broadcast",
                    event_type="simulation_stopped",
                    data={
                        "total_data_points": self.simulation_stats["data_points_generated"],
                        "stop_time": datetime.now().isoformat()
                    },
                    priority=MCPPriority.NORMAL
                )
            
            logger.info("🛑 Bin simulation stopped")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to stop simulation: {e}")
            return False
    
    async def _simulation_loop(self):
        """Main simulation loop that updates bin states"""
        logger.info("🔄 Starting simulation loop...")
        
        while self.simulation_running:
            try:
                current_time = datetime.now()
                
                # Update all bin states
                updated_bins = []
                for bin_id, bin_state in self.bin_states.items():
                    config = self.bin_configurations[bin_id]
                    
                    # Calculate time delta since last update
                    time_delta = (current_time - bin_state.last_update).total_seconds() / 3600  # hours
                    time_delta *= self.simulation_speed  # Apply speed multiplier
                    
                    # Update bin state
                    updated_state = await self._update_bin_state(config, bin_state, time_delta, current_time)
                    self.bin_states[bin_id] = updated_state
                    
                    # Collect updated bin data
                    bin_data = self._bin_state_to_dict(config, updated_state)
                    updated_bins.append(bin_data)
                
                # Publish updated data via Redis if available
                if self.redis_client and updated_bins:
                    await self._publish_bin_data(updated_bins)
                
                # Store patterns in vector database periodically
                if len(updated_bins) > 0 and self.simulation_stats["data_points_generated"] % 100 == 0:
                    await self._store_simulation_patterns(updated_bins)
                
                self.simulation_stats["data_points_generated"] += len(updated_bins)
                
                # Sleep based on simulation speed (update every 5 minutes in real time)
                sleep_time = (5 * 60) / self.simulation_speed  # 5 minutes adjusted for speed
                await asyncio.sleep(min(sleep_time, 60))  # Cap at 1 minute max sleep
                
            except Exception as e:
                logger.error(f"❌ Error in simulation loop: {e}")
                await asyncio.sleep(10)  # Wait before retrying
    
    async def _update_bin_state(
        self, 
        config: BinConfiguration, 
        state: BinState, 
        time_delta: float, 
        current_time: datetime
    ) -> BinState:
        """Update a single bin's state based on temporal patterns"""
        
        # Calculate current fill rate based on various factors
        current_fill_rate = self._calculate_current_fill_rate(config, current_time)
        
        # Update fill level
        fill_increase = current_fill_rate * time_delta
        new_fill_level = min(state.current_fill + (fill_increase / config.capacity) * 100, 100)
        
        # Determine status based on fill level
        if new_fill_level >= 95:
            status = "critical"
        elif new_fill_level >= 85:
            status = "warning"
        elif new_fill_level < 10:
            status = "maintenance"  # Might need cleaning or repair
        else:
            status = "normal"
        
        # Predict overflow time if approaching capacity
        predicted_overflow = None
        if new_fill_level > 70 and current_fill_rate > 0:
            remaining_capacity = config.capacity * (100 - new_fill_level) / 100
            hours_to_overflow = remaining_capacity / current_fill_rate
            predicted_overflow = current_time + timedelta(hours=hours_to_overflow)
        
        # Update environmental factors
        temperature = self._calculate_temperature(current_time)
        humidity = self._calculate_humidity(current_time)
        
        # Check if collection should occur
        if new_fill_level >= 90 or (current_time - state.last_collection).days >= 2:
            # Simulate collection
            new_fill_level = random.uniform(5, 15)  # Some residual waste
            last_collection = current_time
            self.simulation_stats["collections_simulated"] += 1
        else:
            last_collection = state.last_collection
        
        return BinState(
            bin_id=state.bin_id,
            current_fill=new_fill_level,
            fill_rate=current_fill_rate,
            last_collection=last_collection,
            last_update=current_time,
            status=status,
            temperature=temperature,
            humidity=humidity,
            predicted_overflow=predicted_overflow
        )
    
    def _calculate_current_fill_rate(self, config: BinConfiguration, current_time: datetime) -> float:
        """Calculate current fill rate based on temporal patterns and events"""
        base_rate = config.base_fill_rate
        
        # Apply time-of-day patterns
        hour = current_time.hour
        day_of_week = current_time.weekday()  # 0 = Monday, 6 = Sunday
        
        time_multiplier = self._get_time_multiplier(config.bin_type, hour, day_of_week)
        
        # Apply seasonal factors
        seasonal_multiplier = self._get_seasonal_multiplier(current_time.month)
        
        # Apply festival factors
        festival_multiplier = self._get_festival_multiplier(current_time, config.festival_sensitivity)
        
        # Apply weather factors
        weather_multiplier = self._get_weather_multiplier(current_time.month)
        
        # Apply random variation (±20%)
        random_multiplier = random.uniform(0.8, 1.2)
        
        # Calculate final fill rate
        final_rate = (base_rate * 
                     time_multiplier * 
                     seasonal_multiplier * 
                     festival_multiplier * 
                     weather_multiplier * 
                     random_multiplier * 
                     config.seasonal_factor)
        
        return max(0, final_rate)  # Ensure non-negative
    
    def _get_time_multiplier(self, bin_type: str, hour: int, day_of_week: int) -> float:
        """Get time-based multiplier for fill rate"""
        if bin_type not in self.time_patterns:
            return 1.0
        
        patterns = self.time_patterns[bin_type]
        multiplier = 1.0
        
        # Check each pattern
        for pattern_name, (start_hour, end_hour, pattern_multiplier) in patterns.items():
            if start_hour <= end_hour:
                # Normal time range (e.g., 9-18)
                if start_hour <= hour <= end_hour:
                    multiplier = max(multiplier, pattern_multiplier)
            else:
                # Overnight range (e.g., 23-6)
                if hour >= start_hour or hour <= end_hour:
                    multiplier = max(multiplier, pattern_multiplier)
        
        # Weekend adjustments
        if day_of_week >= 5:  # Saturday or Sunday
            if bin_type == "commercial":
                multiplier *= 0.6  # Less commercial activity
            elif bin_type == "public":
                multiplier *= 1.3  # More public activity
            elif bin_type == "residential":
                multiplier *= 1.1  # Slightly more residential activity
        
        return multiplier
    
    def _get_seasonal_multiplier(self, month: int) -> float:
        """Get seasonal multiplier based on month"""
        # Mumbai seasonal patterns
        if month in [6, 7, 8, 9]:  # Monsoon
            return 0.9  # Less outdoor activity
        elif month in [10, 11, 12, 1, 2]:  # Post-monsoon/Winter
            return 1.1  # More outdoor activity
        elif month in [3, 4, 5]:  # Summer
            return 0.95  # Slightly less due to heat
        else:
            return 1.0
    
    def _get_festival_multiplier(self, current_time: datetime, sensitivity: float) -> float:
        """Get festival-based multiplier"""
        month = current_time.month
        day = current_time.day
        
        # Check if current time falls within any festival period
        for festival, details in self.festival_patterns.items():
            if month == details["month"]:
                # Simplified festival detection (would be more sophisticated in real implementation)
                festival_probability = 0.1  # 10% chance any given day in festival month
                if random.random() < festival_probability:
                    return 1.0 + (details["multiplier"] - 1.0) * sensitivity
        
        return 1.0
    
    def _get_weather_multiplier(self, month: int) -> float:
        """Get weather-based multiplier"""
        for weather_type, details in self.weather_patterns.items():
            if month in details["months"]:
                return details["fill_rate_multiplier"]
        return 1.0
    
    def _calculate_temperature(self, current_time: datetime) -> float:
        """Calculate realistic temperature based on time and season"""
        month = current_time.month
        hour = current_time.hour
        
        # Base temperature by month (Mumbai)
        monthly_temps = {
            1: 25, 2: 27, 3: 30, 4: 33, 5: 35, 6: 32,
            7: 29, 8: 28, 9: 29, 10: 31, 11: 29, 12: 26
        }
        
        base_temp = monthly_temps.get(month, 30)
        
        # Daily variation
        if 6 <= hour <= 18:  # Daytime
            temp_variation = math.sin((hour - 6) * math.pi / 12) * 5
        else:  # Nighttime
            temp_variation = -3
        
        return base_temp + temp_variation + random.uniform(-2, 2)
    
    def _calculate_humidity(self, current_time: datetime) -> float:
        """Calculate realistic humidity based on time and season"""
        month = current_time.month
        hour = current_time.hour
        
        # Base humidity by month (Mumbai)
        if month in [6, 7, 8, 9]:  # Monsoon
            base_humidity = 85
        elif month in [10, 11, 12, 1, 2]:  # Post-monsoon/Winter
            base_humidity = 65
        else:  # Summer
            base_humidity = 70
        
        # Daily variation (higher at night)
        if 6 <= hour <= 18:  # Daytime
            humidity_variation = -10
        else:  # Nighttime
            humidity_variation = 5
        
        return max(30, min(95, base_humidity + humidity_variation + random.uniform(-5, 5)))
    
    def _bin_state_to_dict(self, config: BinConfiguration, state: BinState) -> Dict[str, Any]:
        """Convert bin state to dictionary format"""
        return {
            "id": state.bin_id,
            "name": f"Bin {state.bin_id}",
            "ward_id": config.ward_id,
            "latitude": config.latitude,
            "longitude": config.longitude,
            "capacity": config.capacity,
            "current_fill": round(state.current_fill, 2),
            "fill_level": round(state.current_fill, 2),  # Compatibility with existing code
            "fill_rate": round(state.fill_rate, 3),
            "bin_type": config.bin_type,
            "location_type": config.location_type,
            "status": state.status,
            "last_collection": state.last_collection.isoformat(),
            "last_update": state.last_update.isoformat(),
            "temperature": round(state.temperature, 1),
            "humidity": round(state.humidity, 1),
            "predicted_overflow": state.predicted_overflow.isoformat() if state.predicted_overflow else None,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _publish_bin_data(self, bin_data: List[Dict[str, Any]]):
        """Publish bin data via Redis for other agents"""
        try:
            if not self.redis_client:
                return
            
            # Publish to general bin data channel
            message = {
                "message_type": "bin_data_update",
                "timestamp": datetime.now().isoformat(),
                "source": "bin_simulator",
                "data": bin_data,
                "total_bins": len(bin_data)
            }
            
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.redis_client.lpush,
                "bin_data_stream",
                json.dumps(message)
            )
            
            # Also publish via MCP if available
            if self.mcp_handler:
                await self.mcp_handler.send_notification(
                    receiver_agent="broadcast",
                    event_type="bin_data_update",
                    data={
                        "bin_count": len(bin_data),
                        "timestamp": datetime.now().isoformat(),
                        "critical_bins": len([b for b in bin_data if b["status"] == "critical"]),
                        "warning_bins": len([b for b in bin_data if b["status"] == "warning"])
                    },
                    priority=MCPPriority.NORMAL
                )
            
        except Exception as e:
            logger.error(f"❌ Failed to publish bin data: {e}")
    
    async def _store_simulation_patterns(self, bin_data: List[Dict[str, Any]]):
        """Store simulation patterns in vector database"""
        try:
            if not self.vector_db:
                return
            
            # Analyze current patterns
            pattern_analysis = self._analyze_current_patterns(bin_data)
            
            # Store pattern analysis
            await self.vector_db.store_document(
                collection_name="bin_behaviors",
                document_id=f"pattern_{datetime.now().timestamp()}",
                content=json.dumps(pattern_analysis, indent=2),
                metadata={
                    "pattern_type": "simulation_analysis",
                    "bin_count": len(bin_data),
                    "timestamp": datetime.now().isoformat(),
                    "critical_count": pattern_analysis.get("critical_bins", 0),
                    "warning_count": pattern_analysis.get("warning_bins", 0)
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to store simulation patterns: {e}")
    
    def _analyze_current_patterns(self, bin_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze current bin data patterns"""
        if not bin_data:
            return {}
        
        # Calculate statistics
        fill_levels = [b["current_fill"] for b in bin_data]
        fill_rates = [b["fill_rate"] for b in bin_data]
        
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "total_bins": len(bin_data),
            "average_fill_level": sum(fill_levels) / len(fill_levels),
            "max_fill_level": max(fill_levels),
            "min_fill_level": min(fill_levels),
            "average_fill_rate": sum(fill_rates) / len(fill_rates),
            "critical_bins": len([b for b in bin_data if b["status"] == "critical"]),
            "warning_bins": len([b for b in bin_data if b["status"] == "warning"]),
            "normal_bins": len([b for b in bin_data if b["status"] == "normal"]),
            "bins_by_ward": {},
            "bins_by_type": {},
            "predicted_overflows": len([b for b in bin_data if b["predicted_overflow"]])
        }
        
        # Analyze by ward
        for bin_data_item in bin_data:
            ward_id = bin_data_item["ward_id"]
            if ward_id not in analysis["bins_by_ward"]:
                analysis["bins_by_ward"][ward_id] = {"count": 0, "avg_fill": 0, "critical": 0}
            
            analysis["bins_by_ward"][ward_id]["count"] += 1
            analysis["bins_by_ward"][ward_id]["avg_fill"] += bin_data_item["current_fill"]
            if bin_data_item["status"] == "critical":
                analysis["bins_by_ward"][ward_id]["critical"] += 1
        
        # Calculate averages
        for ward_id in analysis["bins_by_ward"]:
            count = analysis["bins_by_ward"][ward_id]["count"]
            analysis["bins_by_ward"][ward_id]["avg_fill"] /= count
        
        # Analyze by type
        for bin_data_item in bin_data:
            bin_type = bin_data_item["bin_type"]
            if bin_type not in analysis["bins_by_type"]:
                analysis["bins_by_type"][bin_type] = {"count": 0, "avg_fill": 0}
            
            analysis["bins_by_type"][bin_type]["count"] += 1
            analysis["bins_by_type"][bin_type]["avg_fill"] += bin_data_item["current_fill"]
        
        # Calculate averages
        for bin_type in analysis["bins_by_type"]:
            count = analysis["bins_by_type"][bin_type]["count"]
            analysis["bins_by_type"][bin_type]["avg_fill"] /= count
        
        return analysis
    
    async def get_bin_data(self, ward_ids: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """Get current bin data, optionally filtered by ward IDs"""
        try:
            bin_data = []
            
            for bin_id, state in self.bin_states.items():
                config = self.bin_configurations[bin_id]
                
                # Filter by ward if specified
                if ward_ids and config.ward_id not in ward_ids:
                    continue
                
                bin_dict = self._bin_state_to_dict(config, state)
                bin_data.append(bin_dict)
            
            return bin_data
            
        except Exception as e:
            logger.error(f"❌ Failed to get bin data: {e}")
            return []
    
    async def simulate_collection(self, bin_id: str) -> bool:
        """Simulate collection for a specific bin"""
        try:
            if bin_id not in self.bin_states:
                logger.error(f"❌ Bin not found: {bin_id}")
                return False
            
            # Reset bin to post-collection state
            state = self.bin_states[bin_id]
            state.current_fill = random.uniform(5, 15)  # Some residual waste
            state.last_collection = datetime.now()
            state.status = "normal"
            state.predicted_overflow = None
            
            self.simulation_stats["collections_simulated"] += 1
            
            logger.info(f"🚛 Simulated collection for bin {bin_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to simulate collection for {bin_id}: {e}")
            return False
    
    def get_simulation_stats(self) -> Dict[str, Any]:
        """Get current simulation statistics"""
        return {
            **self.simulation_stats,
            "simulation_running": self.simulation_running,
            "simulation_speed": self.simulation_speed,
            "current_time": datetime.now().isoformat(),
            "bins_by_status": {
                "normal": len([s for s in self.bin_states.values() if s.status == "normal"]),
                "warning": len([s for s in self.bin_states.values() if s.status == "warning"]),
                "critical": len([s for s in self.bin_states.values() if s.status == "critical"]),
                "maintenance": len([s for s in self.bin_states.values() if s.status == "maintenance"])
            }
        }
    
    # Tool implementations
    def _configure_bins_tool(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Tool implementation for configuring bins"""
        try:
            ward_ids = parameters.get("ward_ids", list(range(1, 25)))
            bins_per_ward = parameters.get("bins_per_ward", 50)
            
            # This would trigger reconfiguration
            return {
                "status": "success",
                "message": f"Bin configuration updated for wards {ward_ids}",
                "total_bins": len(ward_ids) * bins_per_ward
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _control_simulation_tool(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Tool implementation for controlling simulation"""
        try:
            action = parameters.get("action", "status")
            
            if action == "start":
                speed = parameters.get("speed", 1.0)
                asyncio.create_task(self.start_simulation(speed))
                return {"status": "success", "message": f"Simulation started with {speed}x speed"}
            
            elif action == "stop":
                asyncio.create_task(self.stop_simulation())
                return {"status": "success", "message": "Simulation stopped"}
            
            elif action == "status":
                return {"status": "success", "data": self.get_simulation_stats()}
            
            else:
                return {"status": "error", "message": f"Unknown action: {action}"}
                
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _generate_bin_data_tool(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Tool implementation for generating bin data"""
        try:
            ward_ids = parameters.get("ward_ids")
            
            # Get current bin data
            bin_data = asyncio.run(self.get_bin_data(ward_ids))
            
            return {
                "status": "success",
                "bin_data": bin_data,
                "count": len(bin_data),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _analyze_patterns_tool(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Tool implementation for analyzing patterns"""
        try:
            # Get current bin data and analyze patterns
            bin_data = asyncio.run(self.get_bin_data())
            analysis = self._analyze_current_patterns(bin_data)
            
            return {
                "status": "success",
                "analysis": analysis,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    # MCP request handlers
    async def _handle_generate_bin_data_request(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP request for generating bin data"""
        try:
            ward_ids = parameters.get("ward_ids")
            duration_hours = parameters.get("duration_hours", 1)
            include_festivals = parameters.get("include_festivals", True)
            
            # Get current bin data
            bin_data = await self.get_bin_data(ward_ids)
            
            # Analyze patterns
            patterns = self._analyze_current_patterns(bin_data)
            
            return {
                "bin_data": bin_data,
                "timestamp": datetime.now().isoformat(),
                "patterns_detected": [
                    f"Average fill level: {patterns.get('average_fill_level', 0):.1f}%",
                    f"Critical bins: {patterns.get('critical_bins', 0)}",
                    f"Predicted overflows: {patterns.get('predicted_overflows', 0)}"
                ]
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to handle generate bin data request: {e}")
            return {"error": str(e)}
    
    async def _handle_simulation_control_request(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP request for simulation control"""
        try:
            action = parameters.get("action", "status")
            sim_parameters = parameters.get("parameters", {})
            
            if action == "start":
                speed = sim_parameters.get("speed", 1.0)
                success = await self.start_simulation(speed)
                return {
                    "status": "started" if success else "failed",
                    "message": f"Simulation {'started' if success else 'failed to start'} with {speed}x speed",
                    "simulation_stats": self.get_simulation_stats()
                }
            
            elif action == "stop":
                success = await self.stop_simulation()
                return {
                    "status": "stopped" if success else "failed",
                    "message": f"Simulation {'stopped' if success else 'failed to stop'}",
                    "simulation_stats": self.get_simulation_stats()
                }
            
            else:
                return {
                    "status": "running" if self.simulation_running else "stopped",
                    "message": "Simulation status retrieved",
                    "simulation_stats": self.get_simulation_stats()
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to handle simulation control request: {e}")
            return {"error": str(e)}

logger.info("🗑️ Bin Simulator Agent loaded successfully")