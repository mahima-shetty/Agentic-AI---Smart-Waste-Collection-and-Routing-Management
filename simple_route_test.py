"""
Simple test for Route Optimization Agent core functionality
"""

import asyncio
import logging
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_agents.route_optimizer import RouteOptimizationAgent, VehicleInfo, BinLocation

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def simple_test():
    """Simple test of route optimization"""
    try:
        logger.info("🧪 Starting simple Route Optimization test...")
        
        # Create Route Optimization Agent (without dependencies)
        route_agent = RouteOptimizationAgent()
        
        # Initialize the agent
        success = await route_agent.initialize_agent()
        if not success:
            logger.error("❌ Failed to initialize Route Optimization Agent")
            return False
        
        logger.info("✅ Route Optimization Agent initialized")
        
        # Create sample data
        vehicles = [
            VehicleInfo(
                vehicle_id="truck_001",
                capacity=2000.0,
                current_location=(19.0760, 72.8777),
                max_distance=100.0,
                cost_per_km=15.0
            )
        ]
        
        bins = [
            BinLocation(
                bin_id="BIN_001",
                latitude=19.0800,
                longitude=72.8800,
                fill_level=85.0,
                priority=4,
                estimated_collection_time=5,
                bin_type="residential",
                capacity=240.0
            ),
            BinLocation(
                bin_id="BIN_002",
                latitude=19.0820,
                longitude=72.8820,
                fill_level=92.0,
                priority=5,
                estimated_collection_time=7,
                bin_type="commercial",
                capacity=480.0
            )
        ]
        
        logger.info(f"✅ Created {len(vehicles)} vehicles and {len(bins)} bins")
        
        # Test route optimization with timeout
        logger.info("🔧 Starting route optimization...")
        result = await asyncio.wait_for(
            route_agent.optimize_routes(
                ward_id=1,
                available_vehicles=vehicles,
                bin_locations=bins,
                use_workflow=False  # Use direct optimization
            ),
            timeout=30.0  # 30 second timeout
        )
        
        if result.get("success"):
            logger.info("✅ Route optimization successful!")
            logger.info(f"   - Optimization score: {result.get('optimization_score', 0):.1f}")
            logger.info(f"   - Routes generated: {len(result.get('optimized_routes', []))}")
            logger.info(f"   - Total distance: {result.get('total_distance', 0):.1f} km")
            logger.info(f"   - Estimated cost: ₹{result.get('estimated_fuel_cost', 0):.0f}")
            
            # Show route details
            for i, route in enumerate(result.get('optimized_routes', [])):
                logger.info(f"   Route {i+1}: {route.get('vehicle_id')} - {route.get('total_bins')} bins")
            
            return True
        else:
            logger.error(f"❌ Route optimization failed: {result.get('error')}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(simple_test())
    if success:
        print("\n✅ Simple route optimization test PASSED")
    else:
        print("\n❌ Simple route optimization test FAILED")