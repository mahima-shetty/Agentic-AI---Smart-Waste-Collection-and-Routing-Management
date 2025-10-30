#!/usr/bin/env python3
"""
Test script for Master Coordination Agent integration
Tests the integration between all agents via Redis and MCP protocols
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project directories to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ai_agents"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_master_agent_integration():
    """Test Master Coordination Agent integration with all specialized agents"""
    
    logger.info("🧪 Starting Master Agent Integration Test...")
    
    try:
        # Import the system initializer
        from app import SystemInitializer
        
        # Initialize system
        system = SystemInitializer()
        
        # Test database initialization
        logger.info("📊 Testing database initialization...")
        db_success = await system.initialize_database()
        logger.info(f"Database: {'✅ Success' if db_success else '❌ Failed'}")
        
        # Test ChromaDB initialization
        logger.info("🗄️ Testing ChromaDB initialization...")
        chroma_success = system.initialize_chromadb()
        logger.info(f"ChromaDB: {'✅ Success' if chroma_success else '❌ Failed'}")
        
        # Test Redis initialization
        logger.info("🔄 Testing Redis initialization...")
        redis_success = await system.initialize_redis()
        logger.info(f"Redis: {'✅ Success' if redis_success else '⚠️ Not available (optional)'}")
        
        # Test Master Agent initialization
        logger.info("🤖 Testing Master Agent initialization...")
        master_success = await system.initialize_master_agent()
        logger.info(f"Master Agent: {'✅ Success' if master_success else '❌ Failed'}")
        
        if master_success and system.master_agent:
            # Test agent integration
            logger.info("🔗 Testing agent integration...")
            
            # Test system status
            status = system.master_agent.get_system_status()
            logger.info(f"System Status: {status['master_agent_status']}")
            logger.info(f"Active Agents: {len([a for a in status['active_agents'].values() if a.get('status') == 'active'])}")
            
            # Test bin data retrieval
            if hasattr(system.master_agent, 'bin_simulator') and system.master_agent.bin_simulator:
                logger.info("🗑️ Testing bin data retrieval...")
                bin_data = await system.master_agent.bin_simulator.get_bin_data()
                logger.info(f"Retrieved {len(bin_data)} bin records")
                
                # Show sample bin data
                if bin_data:
                    sample_bin = bin_data[0]
                    logger.info(f"Sample bin: {sample_bin['id']} - {sample_bin['current_fill']:.1f}% full")
            
            # Test route optimization
            logger.info("🚛 Testing route optimization...")
            route_result = await system.master_agent.request_route_optimization(ward_id=1)
            if route_result.get("success"):
                routes = route_result.get("optimized_routes", [])
                logger.info(f"Generated {len(routes)} optimized routes")
                if routes:
                    sample_route = routes[0]
                    logger.info(f"Sample route: {sample_route.get('vehicle_id')} - {sample_route.get('total_bins')} bins")
            else:
                logger.info(f"Route optimization: {route_result.get('error', 'No error info')}")
            
            # Test alert retrieval
            logger.info("🚨 Testing alert retrieval...")
            alerts = await system.master_agent.get_active_alerts()
            logger.info(f"Active alerts: {len(alerts)}")
            
            # Test analytics
            logger.info("📊 Testing analytics generation...")
            analytics_result = await system.master_agent.generate_analytics_report()
            if analytics_result.get("success"):
                insights = analytics_result.get("insights", [])
                logger.info(f"Generated {len(insights)} insights")
            else:
                logger.info(f"Analytics: {analytics_result.get('error', 'No error info')}")
            
            # Test dashboard data
            logger.info("📋 Testing dashboard data retrieval...")
            dashboard_data = await system.master_agent.get_system_dashboard_data()
            if "error" not in dashboard_data:
                stats = dashboard_data.get("system_stats", {})
                logger.info(f"Dashboard data: {stats.get('total_bins', 0)} bins, {stats.get('active_alerts', 0)} alerts")
            else:
                logger.info(f"Dashboard data: {dashboard_data.get('error')}")
        
        # Test shutdown
        logger.info("🛑 Testing system shutdown...")
        await system.shutdown_sequence()
        
        logger.info("✅ Integration test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_individual_agents():
    """Test individual agent functionality"""
    
    logger.info("🔧 Testing individual agents...")
    
    try:
        # Test Bin Simulator
        logger.info("🗑️ Testing Bin Simulator Agent...")
        from ai_agents.bin_simulator import BinSimulatorAgent
        
        bin_simulator = BinSimulatorAgent()
        config = {"existing_bins": [], "simulation_speed": 1.0}
        
        init_success = await bin_simulator.initialize_simulation(config)
        logger.info(f"Bin Simulator Init: {'✅ Success' if init_success else '❌ Failed'}")
        
        if init_success:
            # Start simulation briefly
            await bin_simulator.start_simulation(speed_multiplier=10.0)  # 10x speed for testing
            await asyncio.sleep(2)  # Let it run for 2 seconds
            
            # Get data
            bin_data = await bin_simulator.get_bin_data()
            logger.info(f"Generated {len(bin_data)} bin records")
            
            # Stop simulation
            await bin_simulator.stop_simulation()
            logger.info("Bin simulation stopped")
        
        # Test Route Optimizer
        logger.info("🚛 Testing Route Optimization Agent...")
        from ai_agents.route_optimizer import RouteOptimizationAgent, VehicleInfo, BinLocation
        
        route_optimizer = RouteOptimizationAgent()
        init_success = await route_optimizer.initialize_agent()
        logger.info(f"Route Optimizer Init: {'✅ Success' if init_success else '❌ Failed'}")
        
        if init_success:
            # Create test data
            vehicles = [VehicleInfo(
                vehicle_id="test_vehicle_1",
                capacity=2000.0,
                current_location=(19.0760, 72.8777),
                max_distance=100.0,
                cost_per_km=15.0
            )]
            
            bins = [BinLocation(
                bin_id=f"test_bin_{i}",
                latitude=19.0760 + i * 0.001,
                longitude=72.8777 + i * 0.001,
                fill_level=80.0 + i * 2,
                priority=3,
                estimated_collection_time=5,
                bin_type="residential",
                capacity=240.0
            ) for i in range(5)]
            
            # Test optimization
            result = await route_optimizer.optimize_routes(
                ward_id=1,
                available_vehicles=vehicles,
                bin_locations=bins
            )
            
            if result.get("success"):
                logger.info(f"Route optimization successful: {result.get('optimization_score', 0):.1f} score")
            else:
                logger.info(f"Route optimization failed: {result.get('error')}")
        
        logger.info("✅ Individual agent tests completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Individual agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    
    logger.info("🚀 Starting Smart Waste Management Integration Tests...")
    
    # Test individual agents first
    individual_success = await test_individual_agents()
    
    # Test full integration
    integration_success = await test_master_agent_integration()
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    logger.info(f"Individual Agents: {'✅ PASS' if individual_success else '❌ FAIL'}")
    logger.info(f"Full Integration: {'✅ PASS' if integration_success else '❌ FAIL'}")
    
    if individual_success and integration_success:
        logger.info("🎉 ALL TESTS PASSED!")
        return 0
    else:
        logger.info("💥 SOME TESTS FAILED!")
        return 1

if __name__ == "__main__":
    # Run the tests
    exit_code = asyncio.run(main())
    sys.exit(exit_code)