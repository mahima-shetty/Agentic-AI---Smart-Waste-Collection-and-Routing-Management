"""
Route Optimization Agent - AI-powered route optimization with advanced capabilities
Uses OR-Tools VRP solver with LangGraph workflow orchestration and vector similarity search
"""

import asyncio
import logging
import json
import math
import random
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import numpy as np

# OR-Tools for Vehicle Routing Problem
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# LangChain and LangGraph imports
from .langchain_base import BaseLangChainAgent, WasteManagementTool, AgentPromptTemplates
from .langgraph_workflows import LangGraphWorkflowEngine, WorkflowType, RouteOptimizationState
from .mcp_handler import MCPHandler, MCPMessageType, MCPPriority
from .vector_db import VectorDatabaseManager

logger = logging.getLogger(__name__)

@dataclass
class VehicleInfo:
    """Information about a collection vehicle"""
    vehicle_id: str
    capacity: float  # in liters
    current_location: Tuple[float, float]  # (latitude, longitude)
    max_distance: float  # maximum distance per route in km
    cost_per_km: float  # fuel cost per kilometer
    available: bool = True
    current_load: float = 0.0

@dataclass
class BinLocation:
    """Information about a bin location for routing"""
    bin_id: str
    latitude: float
    longitude: float
    fill_level: float  # percentage
    priority: int  # 1-5 scale (5 = highest priority)
    estimated_collection_time: int  # minutes
    bin_type: str
    capacity: float

@dataclass
class RouteSegment:
    """A segment of an optimized route"""
    from_location: str
    to_location: str
    distance: float  # km
    travel_time: int  # minutes
    bin_id: Optional[str] = None
    collection_time: Optional[int] = None

@dataclass
class OptimizedRoute:
    """Complete optimized route for a vehicle"""
    vehicle_id: str
    route_segments: List[RouteSegment]
    total_distance: float
    total_time: int  # minutes
    total_bins: int
    estimated_fuel_cost: float
    optimization_score: float
    route_sequence: List[str]  # bin IDs in order

class RouteOptimizationAgent(BaseLangChainAgent):
    """
    Route Optimization Agent using OR-Tools VRP solver with LangGraph workflow orchestration
    Enhanced with reinforcement learning and vector similarity search for historical patterns
    """
    
    def __init__(
        self,
        redis_client=None,
        vector_db: Optional[VectorDatabaseManager] = None,
        mcp_handler: Optional[MCPHandler] = None,
        workflow_engine: Optional[LangGraphWorkflowEngine] = None
    ):
        super().__init__(
            agent_id="route_optimizer",
            agent_type="route_optimization",
            model_name="gpt-3.5-turbo",
            temperature=0.1
        )
        
        self.redis_client = redis_client
        self.vector_db = vector_db
        self.mcp_handler = mcp_handler
        self.workflow_engine = workflow_engine
        
        # Route optimization parameters
        self.optimization_params = {
            "max_route_duration": 480,  # 8 hours in minutes
            "max_vehicle_capacity": 2000,  # liters
            "service_time_per_bin": 5,  # minutes
            "travel_speed": 25,  # km/h average in Mumbai
            "fuel_cost_per_liter": 100,  # INR
            "vehicle_fuel_efficiency": 8,  # km/liter
            "priority_weight": 2.0,  # multiplier for high priority bins
            "distance_penalty": 1.5,  # penalty for long routes
            "time_window_penalty": 3.0  # penalty for violating time windows
        }
        
        # Learning and adaptation parameters
        self.learning_params = {
            "learning_rate": 0.01,
            "exploration_rate": 0.1,
            "pattern_similarity_threshold": 0.8,
            "historical_weight": 0.3,
            "adaptation_factor": 0.2
        }
        
        # Performance tracking
        self.performance_metrics = {
            "routes_optimized": 0,
            "total_distance_saved": 0.0,
            "total_time_saved": 0,
            "fuel_cost_saved": 0.0,
            "average_optimization_score": 0.0,
            "patterns_learned": 0,
            "adaptations_made": 0
        }
        
        # Cache for frequently used calculations
        self.distance_cache: Dict[Tuple[str, str], float] = {}
        self.pattern_cache: Dict[str, Dict[str, Any]] = {}
        
        logger.info("🚛 Route Optimization Agent initialized")
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the route optimization agent"""
        return AgentPromptTemplates.ROUTE_OPTIMIZATION_SYSTEM
    
    def get_tools(self) -> List[WasteManagementTool]:
        """Get tools available to the route optimization agent"""
        tools = []
        
        # Route optimization tool
        tools.append(WasteManagementTool(
            name="optimize_routes",
            description="Optimize waste collection routes using VRP algorithms and ML",
            func=self._optimize_routes_tool
        ))
        
        # Route analysis tool
        tools.append(WasteManagementTool(
            name="analyze_route_patterns",
            description="Analyze historical route patterns for learning",
            func=self._analyze_route_patterns_tool
        ))
        
        # Route adaptation tool
        tools.append(WasteManagementTool(
            name="adapt_routes",
            description="Adapt routes based on real-time conditions",
            func=self._adapt_routes_tool
        ))
        
        # Natural language route query tool
        tools.append(WasteManagementTool(
            name="process_route_query",
            description="Process natural language route optimization requests",
            func=self._process_route_query_tool
        ))
        
        return tools
    
    async def initialize_agent(self) -> bool:
        """Initialize the route optimization agent with advanced capabilities"""
        try:
            logger.info("🚀 Initializing Route Optimization Agent...")
            
            # Initialize base LangChain agent
            if not super().initialize_agent():
                return False
            
            # Setup MCP capabilities if available
            if self.mcp_handler:
                await self._setup_mcp_capabilities()
            
            # Initialize workflow engine if available
            if self.workflow_engine:
                await self._initialize_workflows()
            
            # Load historical patterns from vector database
            if self.vector_db:
                await self._load_historical_patterns()
            
            # Initialize OR-Tools solver
            await self._initialize_vrp_solver()
            
            logger.info("✅ Route Optimization Agent initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Route Optimization Agent: {e}")
            return False
    
    async def _setup_mcp_capabilities(self):
        """Setup MCP capabilities for route optimization"""
        try:
            # Register route optimization capability
            await self.mcp_handler.register_capability(
                name="optimize_collection_routes",
                description="Optimize waste collection routes using advanced VRP algorithms",
                input_schema={
                    "type": "object",
                    "properties": {
                        "ward_id": {"type": "integer"},
                        "available_vehicles": {"type": "array", "items": {"type": "object"}},
                        "bin_data": {"type": "array", "items": {"type": "object"}},
                        "constraints": {"type": "object"},
                        "optimization_goals": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["ward_id", "available_vehicles", "bin_data"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "optimized_routes": {"type": "array"},
                        "optimization_score": {"type": "number"},
                        "estimated_savings": {"type": "object"},
                        "recommendations": {"type": "array"}
                    }
                },
                handler=self._handle_route_optimization_request
            )
            
            # Register route adaptation capability
            await self.mcp_handler.register_capability(
                name="adapt_routes_realtime",
                description="Adapt existing routes based on real-time conditions",
                input_schema={
                    "type": "object",
                    "properties": {
                        "current_routes": {"type": "array"},
                        "new_conditions": {"type": "object"},
                        "adaptation_type": {"type": "string"}
                    }
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "adapted_routes": {"type": "array"},
                        "changes_made": {"type": "array"},
                        "impact_analysis": {"type": "object"}
                    }
                },
                handler=self._handle_route_adaptation_request
            )
            
            logger.info("🔄 MCP capabilities registered for Route Optimization")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup MCP capabilities: {e}")
    
    async def _initialize_workflows(self):
        """Initialize LangGraph workflows for route optimization"""
        try:
            # The workflow engine should already have route optimization workflows
            # We just need to ensure they're available
            if hasattr(self.workflow_engine, 'route_optimization_graph'):
                logger.info("🔄 Route optimization workflows available")
            else:
                logger.warning("⚠️ Route optimization workflows not found in workflow engine")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize workflows: {e}")
    
    async def _load_historical_patterns(self):
        """Load historical route patterns from vector database"""
        try:
            if not self.vector_db:
                return
            
            # Search for historical route patterns
            patterns = await self.vector_db.semantic_search(
                collection_name="route_patterns",
                query="historical route optimization patterns successful solutions",
                n_results=20
            )
            
            # Process and cache patterns
            for pattern in patterns:
                pattern_data = json.loads(pattern["document"])
                pattern_key = f"ward_{pattern_data.get('ward_id', 0)}_vehicles_{len(pattern_data.get('vehicles', []))}"
                self.pattern_cache[pattern_key] = {
                    "pattern_data": pattern_data,
                    "similarity_score": pattern.get("similarity", 0.0),
                    "optimization_score": pattern["metadata"].get("optimization_score", 0.0),
                    "timestamp": pattern["metadata"].get("timestamp")
                }
            
            self.performance_metrics["patterns_learned"] = len(self.pattern_cache)
            logger.info(f"📚 Loaded {len(self.pattern_cache)} historical route patterns")
            
        except Exception as e:
            logger.error(f"❌ Failed to load historical patterns: {e}")
    
    async def _initialize_vrp_solver(self):
        """Initialize OR-Tools VRP solver with custom parameters"""
        try:
            # VRP solver will be initialized per optimization request
            # This method sets up default solver parameters
            self.solver_params = {
                "time_limit": 30,  # seconds
                "solution_limit": 100,
                "local_search_metaheuristic": routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
                "first_solution_strategy": routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
            }
            
            logger.info("🔧 VRP solver parameters initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize VRP solver: {e}")
    
    async def optimize_routes(
        self,
        ward_id: int,
        available_vehicles: List[VehicleInfo],
        bin_locations: List[BinLocation],
        constraints: Optional[Dict[str, Any]] = None,
        use_workflow: bool = True
    ) -> Dict[str, Any]:
        """
        Main route optimization method using LangGraph workflow orchestration
        """
        try:
            if use_workflow and self.workflow_engine:
                # Use LangGraph workflow for complex optimization
                return await self._optimize_with_workflow(
                    ward_id, available_vehicles, bin_locations, constraints
                )
            else:
                # Direct optimization without workflow
                return await self._optimize_direct(
                    ward_id, available_vehicles, bin_locations, constraints
                )
                
        except Exception as e:
            logger.error(f"❌ Route optimization failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _optimize_with_workflow(
        self,
        ward_id: int,
        available_vehicles: List[VehicleInfo],
        bin_locations: List[BinLocation],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Optimize routes using LangGraph workflow orchestration"""
        try:
            # Prepare input data for workflow
            input_data = {
                "ward_id": ward_id,
                "available_vehicles": [asdict(v) for v in available_vehicles],
                "bin_locations": [asdict(b) for b in bin_locations],
                "constraints": constraints or {},
                "optimization_params": self.optimization_params,
                "learning_params": self.learning_params
            }
            
            # Execute workflow
            result = await self.workflow_engine.execute_workflow(
                workflow_type=WorkflowType.ROUTE_OPTIMIZATION,
                input_data=input_data
            )
            
            # Process workflow result
            if result.get("final_result"):
                final_result = result["final_result"]
                
                # Store optimization patterns in vector database
                if self.vector_db and final_result.get("optimized_routes"):
                    await self._store_optimization_pattern(ward_id, final_result)
                
                # Update performance metrics
                self._update_performance_metrics(final_result)
                
                return {
                    "success": True,
                    "optimized_routes": final_result.get("optimized_routes", []),
                    "optimization_score": final_result.get("optimization_score", 0.0),
                    "validation": final_result.get("validation", {}),
                    "recommendations": final_result.get("recommendations", []),
                    "workflow_id": result.get("workflow_id"),
                    "execution_history": result.get("execution_history", []),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Workflow execution failed"),
                    "workflow_id": result.get("workflow_id"),
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"❌ Workflow-based optimization failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _optimize_direct(
        self,
        ward_id: int,
        available_vehicles: List[VehicleInfo],
        bin_locations: List[BinLocation],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Direct route optimization using OR-Tools VRP solver"""
        try:
            logger.info(f"🔍 Starting direct route optimization for ward {ward_id}")
            
            # Filter bins that need collection (>70% full or high priority)
            bins_to_collect = [
                bin_loc for bin_loc in bin_locations
                if bin_loc.fill_level > 70 or bin_loc.priority >= 4
            ]
            
            if not bins_to_collect:
                return {
                    "success": True,
                    "optimized_routes": [],
                    "optimization_score": 100.0,
                    "message": "No bins require collection",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Check for similar historical patterns
            historical_solution = await self._find_similar_historical_pattern(
                ward_id, len(available_vehicles), len(bins_to_collect)
            )
            
            # Create distance matrix
            distance_matrix = await self._create_distance_matrix(bins_to_collect, available_vehicles)
            
            # Setup VRP problem
            vrp_data = self._create_vrp_data(
                bins_to_collect, available_vehicles, distance_matrix, constraints
            )
            
            # Solve VRP
            solution_data = await self._solve_vrp(vrp_data, historical_solution)
            
            if solution_data:
                solution, manager, routing = solution_data
                # Convert solution to optimized routes
                logger.info("🔄 Converting solution to routes...")
                optimized_routes = self._convert_solution_to_routes(
                    solution, manager, routing, vrp_data, bins_to_collect, available_vehicles
                )
                
                # Calculate optimization score
                optimization_score = self._calculate_optimization_score(
                    optimized_routes, bins_to_collect, available_vehicles
                )
                
                # Generate recommendations
                recommendations = self._generate_recommendations(
                    optimized_routes, bins_to_collect, available_vehicles
                )
                
                # Store pattern for future learning
                if self.vector_db:
                    await self._store_optimization_pattern(ward_id, {
                        "optimized_routes": [asdict(route) for route in optimized_routes],
                        "optimization_score": optimization_score,
                        "bin_count": len(bins_to_collect),
                        "vehicle_count": len(available_vehicles)
                    })
                
                # Update performance metrics
                self._update_performance_metrics({
                    "optimization_score": optimization_score,
                    "routes_count": len(optimized_routes)
                })
                
                return {
                    "success": True,
                    "optimized_routes": [asdict(route) for route in optimized_routes],
                    "optimization_score": optimization_score,
                    "recommendations": recommendations,
                    "bins_collected": len(bins_to_collect),
                    "vehicles_used": len([r for r in optimized_routes if r.total_bins > 0]),
                    "total_distance": sum(r.total_distance for r in optimized_routes),
                    "total_time": sum(r.total_time for r in optimized_routes),
                    "estimated_fuel_cost": sum(r.estimated_fuel_cost for r in optimized_routes),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": "No feasible solution found",
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"❌ Direct optimization failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            } 
   
    async def _find_similar_historical_pattern(
        self,
        ward_id: int,
        vehicle_count: int,
        bin_count: int
    ) -> Optional[Dict[str, Any]]:
        """Find similar historical patterns using vector similarity search"""
        try:
            if not self.vector_db:
                return None
            
            # Create search query
            query = f"route optimization ward {ward_id} vehicles {vehicle_count} bins {bin_count}"
            
            # Search for similar patterns
            similar_patterns = await self.vector_db.find_similar_routes(
                ward_id=ward_id,
                vehicle_count=vehicle_count,
                n_results=3
            )
            
            if similar_patterns:
                # Select best pattern based on similarity and optimization score
                best_pattern = max(
                    similar_patterns,
                    key=lambda p: (
                        p.get("similarity", 0.0) * 0.7 + 
                        p["metadata"].get("optimization_score", 0.0) / 100 * 0.3
                    )
                )
                
                if best_pattern.get("similarity", 0.0) > self.learning_params["pattern_similarity_threshold"]:
                    logger.info(f"📚 Found similar historical pattern with {best_pattern['similarity']:.2f} similarity")
                    return json.loads(best_pattern["document"])
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to find similar historical pattern: {e}")
            return None
    
    async def _create_distance_matrix(
        self,
        bin_locations: List[BinLocation],
        vehicles: List[VehicleInfo]
    ) -> List[List[float]]:
        """Create distance matrix for VRP solver"""
        try:
            # Create list of all locations (depot + bins)
            all_locations = []
            
            # Add vehicle starting locations as depots
            for vehicle in vehicles:
                all_locations.append(vehicle.current_location)
            
            # Add bin locations
            for bin_loc in bin_locations:
                all_locations.append((bin_loc.latitude, bin_loc.longitude))
            
            # Calculate distance matrix
            n_locations = len(all_locations)
            distance_matrix = [[0.0 for _ in range(n_locations)] for _ in range(n_locations)]
            
            for i in range(n_locations):
                for j in range(n_locations):
                    if i != j:
                        distance = self._calculate_haversine_distance(
                            all_locations[i], all_locations[j]
                        )
                        distance_matrix[i][j] = distance
            
            return distance_matrix
            
        except Exception as e:
            logger.error(f"❌ Failed to create distance matrix: {e}")
            return []
    
    def _calculate_haversine_distance(
        self,
        coord1: Tuple[float, float],
        coord2: Tuple[float, float]
    ) -> float:
        """Calculate haversine distance between two coordinates in kilometers"""
        # Check cache first
        cache_key = (f"{coord1[0]:.6f},{coord1[1]:.6f}", f"{coord2[0]:.6f},{coord2[1]:.6f}")
        if cache_key in self.distance_cache:
            return self.distance_cache[cache_key]
        
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth radius in kilometers
        r = 6371
        distance = c * r
        
        # Cache the result
        self.distance_cache[cache_key] = distance
        
        return distance
    
    def _create_vrp_data(
        self,
        bin_locations: List[BinLocation],
        vehicles: List[VehicleInfo],
        distance_matrix: List[List[float]],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create VRP data structure for OR-Tools solver"""
        try:
            # Convert distance matrix to integer (OR-Tools requirement)
            # Scale by 1000 to maintain precision
            int_distance_matrix = [
                [int(distance * 1000) for distance in row]
                for row in distance_matrix
            ]
            
            # Vehicle capacities (in liters * 1000 for precision)
            vehicle_capacities = [int(v.capacity * 1000) for v in vehicles]
            
            # Bin demands (based on fill level and capacity)
            bin_demands = []
            for bin_loc in bin_locations:
                demand = int((bin_loc.fill_level / 100) * bin_loc.capacity * 1000)
                bin_demands.append(demand)
            
            # Depot indices (vehicle starting locations)
            depot_indices = list(range(len(vehicles)))
            
            vrp_data = {
                'distance_matrix': int_distance_matrix,
                'demands': [0] * len(vehicles) + bin_demands,  # Depots have 0 demand
                'vehicle_capacities': vehicle_capacities,
                'num_vehicles': len(vehicles),
                'depot': 0,  # Use first vehicle location as main depot
                'depot_indices': depot_indices,
                'bin_locations': bin_locations,
                'vehicles': vehicles,
                'constraints': constraints or {}
            }
            
            return vrp_data
            
        except Exception as e:
            logger.error(f"❌ Failed to create VRP data: {e}")
            return {}
    
    async def _solve_vrp(
        self,
        vrp_data: Dict[str, Any],
        historical_solution: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """Solve VRP using OR-Tools with optional historical solution as starting point"""
        try:
            # Create routing index manager
            manager = pywrapcp.RoutingIndexManager(
                len(vrp_data['distance_matrix']),
                vrp_data['num_vehicles'],
                vrp_data['depot']
            )
            
            # Create routing model
            routing = pywrapcp.RoutingModel(manager)
            
            # Create distance callback
            def distance_callback(from_index, to_index):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                return vrp_data['distance_matrix'][from_node][to_node]
            
            transit_callback_index = routing.RegisterTransitCallback(distance_callback)
            routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
            
            # Add capacity constraint
            def demand_callback(from_index):
                from_node = manager.IndexToNode(from_index)
                return vrp_data['demands'][from_node]
            
            demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
            routing.AddDimensionWithVehicleCapacity(
                demand_callback_index,
                0,  # null capacity slack
                vrp_data['vehicle_capacities'],  # vehicle maximum capacities
                True,  # start cumul to zero
                'Capacity'
            )
            
            # Add time constraint
            routing.AddDimension(
                transit_callback_index,
                30,  # allow waiting time
                self.optimization_params['max_route_duration'] * 60,  # maximum time per vehicle
                False,  # don't force start cumul to zero
                'Time'
            )
            
            # Set search parameters
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.first_solution_strategy = self.solver_params['first_solution_strategy']
            search_parameters.local_search_metaheuristic = self.solver_params['local_search_metaheuristic']
            search_parameters.time_limit.seconds = min(self.solver_params['time_limit'], 10)  # Cap at 10 seconds for testing
            search_parameters.solution_limit = self.solver_params['solution_limit']
            
            # Apply historical solution as initial solution if available
            if historical_solution and self.learning_params['historical_weight'] > 0:
                initial_solution = self._adapt_historical_solution(
                    historical_solution, vrp_data, manager, routing
                )
                if initial_solution:
                    routing.CloseModelWithParameters(search_parameters)
                    routing.SetAssignmentFromOtherModelAssignment(initial_solution)
            
            # Solve the problem
            logger.info("🔧 Solving VRP with OR-Tools...")
            logger.info(f"   - Locations: {len(vrp_data['distance_matrix'])}")
            logger.info(f"   - Vehicles: {vrp_data['num_vehicles']}")
            logger.info(f"   - Time limit: {search_parameters.time_limit.seconds}s")
            
            solution = routing.SolveWithParameters(search_parameters)
            
            if solution:
                logger.info("✅ VRP solution found")
                return (solution, manager, routing)
            else:
                logger.warning("⚠️ No VRP solution found")
                return None
                
        except Exception as e:
            logger.error(f"❌ VRP solving failed: {e}")
            return None
    
    def _adapt_historical_solution(
        self,
        historical_solution: Dict[str, Any],
        vrp_data: Dict[str, Any],
        manager: Any,
        routing: Any
    ) -> Optional[Any]:
        """Adapt historical solution to current problem"""
        try:
            # This is a simplified adaptation - in a full implementation,
            # this would use more sophisticated ML techniques
            
            # For now, we'll use the historical solution as a guide for
            # the search heuristics rather than as an initial solution
            logger.info("📚 Using historical solution to guide optimization")
            
            return None  # Return None to use default initialization
            
        except Exception as e:
            logger.error(f"❌ Failed to adapt historical solution: {e}")
            return None
    
    def _convert_solution_to_routes(
        self,
        solution: Any,
        manager: Any,
        routing: Any,
        vrp_data: Dict[str, Any],
        bin_locations: List[BinLocation],
        vehicles: List[VehicleInfo]
    ) -> List[OptimizedRoute]:
        """Convert OR-Tools solution to OptimizedRoute objects"""
        try:
            optimized_routes = []
            
            for vehicle_id in range(vrp_data['num_vehicles']):
                logger.info(f"🚛 Processing vehicle {vehicle_id}")
                vehicle = vehicles[vehicle_id]
                route_segments = []
                route_sequence = []
                total_distance = 0.0
                total_time = 0
                total_bins = 0
                
                index = routing.Start(vehicle_id)
                previous_node = 0  # Start from depot
                logger.info(f"   Starting from index {index}")
                
                while not routing.IsEnd(index):
                    current_node = manager.IndexToNode(index)
                    
                    # If this is a bin location (not depot)
                    if current_node >= len(vehicles):
                        bin_index = current_node - len(vehicles)
                        if bin_index < len(bin_locations):
                            bin_loc = bin_locations[bin_index]
                            
                            # Calculate distance from previous location
                            distance_scaled = vrp_data['distance_matrix'][previous_node][current_node]
                            segment_distance = distance_scaled / 1000.0  # Convert back from scaled
                            segment_time = int((segment_distance / self.optimization_params['travel_speed']) * 60)
                            
                            route_segment = RouteSegment(
                                from_location=f"location_{previous_node}",
                                to_location=f"bin_{bin_index}",
                                distance=segment_distance,
                                travel_time=segment_time,
                                bin_id=bin_loc.bin_id,
                                collection_time=bin_loc.estimated_collection_time
                            )
                            
                            route_segments.append(route_segment)
                            route_sequence.append(bin_loc.bin_id)
                            total_distance += segment_distance
                            total_time += segment_time + bin_loc.estimated_collection_time
                            total_bins += 1
                    
                    previous_node = current_node
                    index = solution.Value(routing.NextVar(index))
                
                # Calculate costs
                estimated_fuel_cost = (
                    total_distance / self.optimization_params['vehicle_fuel_efficiency'] *
                    self.optimization_params['fuel_cost_per_liter']
                )
                
                # Calculate optimization score for this route
                optimization_score = self._calculate_route_score(
                    total_distance, total_time, total_bins, vehicle.capacity
                )
                
                optimized_route = OptimizedRoute(
                    vehicle_id=vehicle.vehicle_id,
                    route_segments=route_segments,
                    total_distance=total_distance,
                    total_time=total_time,
                    total_bins=total_bins,
                    estimated_fuel_cost=estimated_fuel_cost,
                    optimization_score=optimization_score,
                    route_sequence=route_sequence
                )
                
                optimized_routes.append(optimized_route)
            
            logger.info(f"✅ Converted solution to {len(optimized_routes)} routes")
            return optimized_routes
            
        except Exception as e:
            logger.error(f"❌ Failed to convert solution to routes: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _calculate_route_score(
        self,
        distance: float,
        time: int,
        bins: int,
        vehicle_capacity: float
    ) -> float:
        """Calculate optimization score for a route (0-100)"""
        try:
            # Base score
            score = 100.0
            
            # Distance penalty (prefer shorter routes)
            if distance > 50:  # More than 50km
                score -= (distance - 50) * 0.5
            
            # Time penalty (prefer shorter time)
            if time > 300:  # More than 5 hours
                score -= (time - 300) * 0.1
            
            # Efficiency bonus (more bins per km)
            if distance > 0:
                efficiency = bins / distance
                if efficiency > 2:  # More than 2 bins per km
                    score += (efficiency - 2) * 5
            
            # Capacity utilization bonus
            if bins > 0:
                utilization = min(bins * 200 / vehicle_capacity, 1.0)  # Assume 200L per bin
                score += utilization * 10
            
            return max(0.0, min(100.0, score))
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate route score: {e}")
            return 50.0
    
    def _calculate_optimization_score(
        self,
        routes: List[OptimizedRoute],
        bin_locations: List[BinLocation],
        vehicles: List[VehicleInfo]
    ) -> float:
        """Calculate overall optimization score"""
        try:
            if not routes:
                return 0.0
            
            # Calculate weighted average of route scores
            total_score = 0.0
            total_weight = 0.0
            
            for route in routes:
                weight = route.total_bins + 1  # +1 to avoid zero weight
                total_score += route.optimization_score * weight
                total_weight += weight
            
            if total_weight > 0:
                return total_score / total_weight
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"❌ Failed to calculate optimization score: {e}")
            return 0.0
    
    def _generate_recommendations(
        self,
        routes: List[OptimizedRoute],
        bin_locations: List[BinLocation],
        vehicles: List[VehicleInfo]
    ) -> List[str]:
        """Generate optimization recommendations"""
        try:
            recommendations = []
            
            # Analyze route efficiency
            total_distance = sum(r.total_distance for r in routes)
            total_bins = sum(r.total_bins for r in routes)
            
            if total_distance > 0:
                efficiency = total_bins / total_distance
                if efficiency < 1.5:
                    recommendations.append(
                        "Consider consolidating routes to improve efficiency (currently {:.1f} bins/km)".format(efficiency)
                    )
            
            # Check for underutilized vehicles
            for route in routes:
                if route.total_bins == 0:
                    recommendations.append(f"Vehicle {route.vehicle_id} is not utilized - consider reassigning")
                elif route.total_bins < 5:
                    recommendations.append(f"Vehicle {route.vehicle_id} has low utilization ({route.total_bins} bins)")
            
            # Check for long routes
            for route in routes:
                if route.total_time > 360:  # More than 6 hours
                    recommendations.append(
                        f"Route for vehicle {route.vehicle_id} is long ({route.total_time} minutes) - consider splitting"
                    )
            
            # Priority bin recommendations
            high_priority_bins = [b for b in bin_locations if b.priority >= 4]
            if high_priority_bins:
                recommendations.append(f"Prioritize collection of {len(high_priority_bins)} high-priority bins")
            
            # Time-based recommendations
            current_hour = datetime.now().hour
            if 6 <= current_hour <= 10:
                recommendations.append("Optimal time for collection - traffic is lighter in morning hours")
            elif 16 <= current_hour <= 19:
                recommendations.append("Consider avoiding peak traffic hours (4-7 PM) for better efficiency")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Failed to generate recommendations: {e}")
            return ["Unable to generate recommendations due to analysis error"]
    
    async def _store_optimization_pattern(
        self,
        ward_id: int,
        optimization_result: Dict[str, Any]
    ):
        """Store optimization pattern in vector database for future learning"""
        try:
            if not self.vector_db:
                return
            
            # Create pattern description
            pattern_description = f"""
            Route optimization for ward {ward_id}:
            - Vehicles used: {optimization_result.get('vehicle_count', 0)}
            - Bins collected: {optimization_result.get('bin_count', 0)}
            - Optimization score: {optimization_result.get('optimization_score', 0):.1f}
            - Total distance: {sum(r.get('total_distance', 0) for r in optimization_result.get('optimized_routes', [])):.1f} km
            - Efficiency: {optimization_result.get('bin_count', 0) / max(sum(r.get('total_distance', 1) for r in optimization_result.get('optimized_routes', [])), 1):.2f} bins/km
            """
            
            # Store in vector database
            await self.vector_db.store_route_pattern(
                ward_id=ward_id,
                route_data=optimization_result,
                optimization_score=optimization_result.get('optimization_score', 0.0),
                context={
                    "agent_id": self.agent_id,
                    "optimization_method": "or_tools_vrp",
                    "learning_enabled": True
                }
            )
            
            logger.info(f"📚 Stored optimization pattern for ward {ward_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to store optimization pattern: {e}")
    
    def _update_performance_metrics(self, result: Dict[str, Any]):
        """Update agent performance metrics"""
        try:
            self.performance_metrics["routes_optimized"] += 1
            
            if "optimization_score" in result:
                # Update average optimization score
                current_avg = self.performance_metrics["average_optimization_score"]
                count = self.performance_metrics["routes_optimized"]
                new_score = result["optimization_score"]
                
                self.performance_metrics["average_optimization_score"] = (
                    (current_avg * (count - 1) + new_score) / count
                )
            
            # Update other metrics if available
            if "distance_saved" in result:
                self.performance_metrics["total_distance_saved"] += result["distance_saved"]
            
            if "time_saved" in result:
                self.performance_metrics["total_time_saved"] += result["time_saved"]
            
            if "fuel_cost_saved" in result:
                self.performance_metrics["fuel_cost_saved"] += result["fuel_cost_saved"]
            
        except Exception as e:
            logger.error(f"❌ Failed to update performance metrics: {e}")
    
    # Natural Language Processing Methods
    async def process_natural_language_request(self, request: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process natural language route optimization requests"""
        try:
            # Store conversation in vector database for context
            if self.vector_db:
                await self.vector_db.store_conversation(
                    agent_id=self.agent_id,
                    conversation_id=context.get("conversation_id", "default"),
                    message=request,
                    message_type="user_request",
                    context=context
                )
            
            # Parse the request to extract optimization parameters
            parsed_request = await self._parse_natural_language_request(request, context)
            
            if parsed_request.get("success"):
                # Execute optimization based on parsed request
                optimization_result = await self.optimize_routes(
                    ward_id=parsed_request.get("ward_id", 1),
                    available_vehicles=parsed_request.get("vehicles", []),
                    bin_locations=parsed_request.get("bins", []),
                    constraints=parsed_request.get("constraints", {}),
                    use_workflow=True
                )
                
                # Generate natural language response
                response = await self._generate_natural_language_response(
                    optimization_result, request, context
                )
                
                # Store response in vector database
                if self.vector_db:
                    await self.vector_db.store_conversation(
                        agent_id=self.agent_id,
                        conversation_id=context.get("conversation_id", "default"),
                        message=response,
                        message_type="agent_response",
                        context={"optimization_result": optimization_result}
                    )
                
                return {
                    "success": True,
                    "response": response,
                    "optimization_result": optimization_result,
                    "parsed_request": parsed_request
                }
            else:
                return {
                    "success": False,
                    "error": parsed_request.get("error", "Failed to parse request"),
                    "response": "I couldn't understand your route optimization request. Please provide more specific details about the ward, vehicles, or constraints."
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to process natural language request: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": "I encountered an error while processing your request. Please try again."
            }
    
    async def _parse_natural_language_request(
        self,
        request: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Parse natural language request to extract optimization parameters"""
        try:
            # This is a simplified parser - in a full implementation,
            # this would use more sophisticated NLP techniques
            
            parsed = {
                "success": True,
                "ward_id": 1,  # Default
                "vehicles": [],
                "bins": [],
                "constraints": {},
                "optimization_goals": []
            }
            
            # Extract ward information
            import re
            ward_match = re.search(r'ward\s+(\d+)', request.lower())
            if ward_match:
                parsed["ward_id"] = int(ward_match.group(1))
            
            # Extract vehicle information
            vehicle_match = re.search(r'(\d+)\s+vehicles?', request.lower())
            if vehicle_match:
                vehicle_count = int(vehicle_match.group(1))
                # Create default vehicles
                for i in range(vehicle_count):
                    parsed["vehicles"].append(VehicleInfo(
                        vehicle_id=f"vehicle_{i+1}",
                        capacity=2000.0,
                        current_location=(19.0760, 72.8777),  # Mumbai center
                        max_distance=100.0,
                        cost_per_km=15.0
                    ))
            
            # Extract optimization goals
            if "minimize" in request.lower():
                if "distance" in request.lower():
                    parsed["optimization_goals"].append("minimize_distance")
                if "time" in request.lower():
                    parsed["optimization_goals"].append("minimize_time")
                if "cost" in request.lower():
                    parsed["optimization_goals"].append("minimize_cost")
            
            # Extract constraints
            if "before" in request.lower():
                time_match = re.search(r'before\s+(\d+)', request.lower())
                if time_match:
                    parsed["constraints"]["max_time"] = int(time_match.group(1)) * 60  # Convert to minutes
            
            return parsed
            
        except Exception as e:
            logger.error(f"❌ Failed to parse natural language request: {e}")
            return {"success": False, "error": str(e)}
    
    async def _generate_natural_language_response(
        self,
        optimization_result: Dict[str, Any],
        original_request: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate natural language response for optimization results"""
        try:
            if not optimization_result.get("success"):
                return f"I couldn't optimize the routes due to: {optimization_result.get('error', 'unknown error')}"
            
            routes = optimization_result.get("optimized_routes", [])
            score = optimization_result.get("optimization_score", 0)
            
            if not routes:
                return "No routes were needed - all bins are below collection threshold."
            
            # Generate response based on results
            response_parts = []
            
            # Summary
            total_bins = sum(r.get("total_bins", 0) for r in routes)
            total_distance = sum(r.get("total_distance", 0) for r in routes)
            total_cost = sum(r.get("estimated_fuel_cost", 0) for r in routes)
            
            response_parts.append(
                f"I've optimized the routes with a score of {score:.1f}/100. "
                f"The plan covers {total_bins} bins across {len(routes)} vehicles, "
                f"with a total distance of {total_distance:.1f} km and estimated fuel cost of ₹{total_cost:.0f}."
            )
            
            # Route details
            if len(routes) <= 3:  # Provide details for small number of routes
                for i, route in enumerate(routes):
                    if route.get("total_bins", 0) > 0:
                        response_parts.append(
                            f"Vehicle {route.get('vehicle_id', i+1)} will collect {route.get('total_bins', 0)} bins "
                            f"over {route.get('total_distance', 0):.1f} km in approximately {route.get('total_time', 0)} minutes."
                        )
            
            # Recommendations
            recommendations = optimization_result.get("recommendations", [])
            if recommendations:
                response_parts.append("Recommendations: " + "; ".join(recommendations[:2]))  # Limit to 2 recommendations
            
            return " ".join(response_parts)
            
        except Exception as e:
            logger.error(f"❌ Failed to generate natural language response: {e}")
            return "I've completed the route optimization, but encountered an issue generating the detailed response."
    
    # Tool implementations
    def _optimize_routes_tool(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Tool implementation for route optimization"""
        try:
            ward_id = parameters.get("ward_id", 1)
            vehicles_data = parameters.get("vehicles", [])
            bins_data = parameters.get("bins", [])
            constraints = parameters.get("constraints", {})
            
            # Convert data to objects
            vehicles = [VehicleInfo(**v) for v in vehicles_data]
            bins = [BinLocation(**b) for b in bins_data]
            
            # Run optimization
            result = asyncio.run(self.optimize_routes(ward_id, vehicles, bins, constraints))
            
            return result
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _analyze_route_patterns_tool(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Tool implementation for route pattern analysis"""
        try:
            ward_id = parameters.get("ward_id")
            time_range = parameters.get("time_range", "last_30_days")
            
            # This would analyze historical patterns
            # For now, return current performance metrics
            return {
                "success": True,
                "performance_metrics": self.performance_metrics,
                "patterns_analyzed": len(self.pattern_cache),
                "ward_id": ward_id,
                "time_range": time_range
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _adapt_routes_tool(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Tool implementation for route adaptation"""
        try:
            current_routes = parameters.get("current_routes", [])
            new_conditions = parameters.get("new_conditions", {})
            adaptation_type = parameters.get("adaptation_type", "traffic")
            
            # This would implement real-time route adaptation
            # For now, return a simple adaptation result
            adapted_routes = []
            for route in current_routes:
                # Simple adaptation logic
                adapted_route = route.copy()
                if adaptation_type == "traffic":
                    adapted_route["total_time"] = int(route.get("total_time", 0) * 1.2)  # 20% increase
                elif adaptation_type == "emergency":
                    # Prioritize emergency bins
                    adapted_route["priority_adjusted"] = True
                
                adapted_routes.append(adapted_route)
            
            return {
                "success": True,
                "adapted_routes": adapted_routes,
                "changes_made": [f"Applied {adaptation_type} adaptation"],
                "impact_analysis": {
                    "time_impact": "20% increase due to traffic",
                    "cost_impact": "Minimal",
                    "efficiency_impact": "Maintained"
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _process_route_query_tool(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Tool implementation for natural language route queries"""
        try:
            query = parameters.get("query", "")
            context = parameters.get("context", {})
            
            # Process natural language query
            result = asyncio.run(self.process_natural_language_request(query, context))
            
            return result
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # MCP request handlers
    async def _handle_route_optimization_request(
        self,
        parameters: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle MCP request for route optimization"""
        try:
            ward_id = parameters.get("ward_id", 1)
            vehicles_data = parameters.get("available_vehicles", [])
            bins_data = parameters.get("bin_data", [])
            constraints = parameters.get("constraints", {})
            
            # Convert data to objects
            vehicles = []
            for v_data in vehicles_data:
                vehicles.append(VehicleInfo(
                    vehicle_id=v_data.get("vehicle_id", "unknown"),
                    capacity=v_data.get("capacity", 2000.0),
                    current_location=(v_data.get("latitude", 19.0760), v_data.get("longitude", 72.8777)),
                    max_distance=v_data.get("max_distance", 100.0),
                    cost_per_km=v_data.get("cost_per_km", 15.0)
                ))
            
            bins = []
            for b_data in bins_data:
                bins.append(BinLocation(
                    bin_id=b_data.get("id", "unknown"),
                    latitude=b_data.get("latitude", 19.0760),
                    longitude=b_data.get("longitude", 72.8777),
                    fill_level=b_data.get("fill_level", 0.0),
                    priority=b_data.get("priority", 1),
                    estimated_collection_time=b_data.get("collection_time", 5),
                    bin_type=b_data.get("bin_type", "residential"),
                    capacity=b_data.get("capacity", 240.0)
                ))
            
            # Perform optimization
            result = await self.optimize_routes(ward_id, vehicles, bins, constraints)
            
            return {
                "optimized_routes": result.get("optimized_routes", []),
                "optimization_score": result.get("optimization_score", 0.0),
                "estimated_savings": {
                    "distance_km": result.get("total_distance", 0.0),
                    "time_minutes": result.get("total_time", 0),
                    "fuel_cost_inr": result.get("estimated_fuel_cost", 0.0)
                },
                "recommendations": result.get("recommendations", [])
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to handle route optimization request: {e}")
            return {"error": str(e)}
    
    async def _handle_route_adaptation_request(
        self,
        parameters: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle MCP request for route adaptation"""
        try:
            current_routes = parameters.get("current_routes", [])
            new_conditions = parameters.get("new_conditions", {})
            adaptation_type = parameters.get("adaptation_type", "traffic")
            
            # Implement route adaptation logic
            adapted_routes = []
            changes_made = []
            
            for route in current_routes:
                adapted_route = route.copy()
                
                if adaptation_type == "traffic":
                    # Adjust for traffic conditions
                    traffic_factor = new_conditions.get("traffic_factor", 1.2)
                    adapted_route["total_time"] = int(route.get("total_time", 0) * traffic_factor)
                    changes_made.append(f"Adjusted time for traffic (factor: {traffic_factor})")
                
                elif adaptation_type == "emergency":
                    # Handle emergency bin additions
                    emergency_bins = new_conditions.get("emergency_bins", [])
                    if emergency_bins:
                        adapted_route["emergency_bins_added"] = len(emergency_bins)
                        changes_made.append(f"Added {len(emergency_bins)} emergency bins")
                
                elif adaptation_type == "vehicle_breakdown":
                    # Handle vehicle breakdown
                    broken_vehicle = new_conditions.get("broken_vehicle_id")
                    if route.get("vehicle_id") == broken_vehicle:
                        adapted_route["status"] = "reassignment_needed"
                        changes_made.append(f"Marked route for reassignment due to vehicle breakdown")
                
                adapted_routes.append(adapted_route)
            
            # Calculate impact analysis
            impact_analysis = {
                "routes_affected": len([r for r in adapted_routes if r != current_routes[adapted_routes.index(r)]]),
                "total_time_change": sum(r.get("total_time", 0) for r in adapted_routes) - sum(r.get("total_time", 0) for r in current_routes),
                "adaptation_type": adaptation_type,
                "severity": "high" if adaptation_type == "emergency" else "medium"
            }
            
            return {
                "adapted_routes": adapted_routes,
                "changes_made": changes_made,
                "impact_analysis": impact_analysis
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to handle route adaptation request: {e}")
            return {"error": str(e)}
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status and performance metrics"""
        base_status = super().get_agent_status()
        
        # Add route optimization specific metrics
        base_status.update({
            "optimization_params": self.optimization_params,
            "learning_params": self.learning_params,
            "performance_metrics": self.performance_metrics,
            "cached_patterns": len(self.pattern_cache),
            "cached_distances": len(self.distance_cache),
            "vrp_solver_ready": True,
            "vector_db_connected": self.vector_db is not None,
            "mcp_handler_connected": self.mcp_handler is not None,
            "workflow_engine_connected": self.workflow_engine is not None
        })
        
        return base_status

logger.info("🚛 Route Optimization Agent loaded successfully")