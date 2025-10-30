#!/usr/bin/env python3
"""
Test script for Bin Simulator Agent
Tests the basic functionality of the bin simulator
"""

import asyncio
import json
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_bin_simulator():
    """Test the Bin Simulator Agent functionality"""
    try:
        # Import the agent
        from ai_agents.bin_simulator import BinSimulatorAgent
        
        logger.info("🧪 Starting Bin Simulator Agent tests...")
        
        # Create agent instance
        agent = BinSimulatorAgent()
        
        # Test configuration
        config = {
            "existing_bins": [
                {"name": "Dustbin 1", "latitude": 19.168254558644758, "longitude": 72.94358994398651},
                {"name": "Dustbin 2", "latitude": 19.151482656453155, "longitude": 72.93089775714564}
            ]
        }
        
        # Initialize the agent
        logger.info("🔧 Initializing Bin Simulator Agent...")
        success = await agent.initialize_simulation(config)
        
        if not success:
            logger.error("❌ Failed to initialize agent")
            return False
        
        logger.info("✅ Agent initialized successfully")
        
        # Test bin data generation
        logger.info("📊 Testing bin data generation...")
        bin_data = await agent.get_bin_data(ward_ids=[1, 2, 3])
        
        logger.info(f"📍 Generated data for {len(bin_data)} bins")
        
        # Display sample bin data
        if bin_data:
            sample_bin = bin_data[0]
            logger.info(f"📋 Sample bin data:")
            logger.info(f"   - ID: {sample_bin['id']}")
            logger.info(f"   - Ward: {sample_bin['ward_id']}")
            logger.info(f"   - Fill Level: {sample_bin['current_fill']:.1f}%")
            logger.info(f"   - Status: {sample_bin['status']}")
            logger.info(f"   - Type: {sample_bin['bin_type']}")
            logger.info(f"   - Location: {sample_bin['location_type']}")
        
        # Test simulation statistics
        logger.info("📈 Testing simulation statistics...")
        stats = agent.get_simulation_stats()
        logger.info(f"📊 Simulation stats:")
        logger.info(f"   - Total bins: {stats['total_bins']}")
        logger.info(f"   - Active bins: {stats['active_bins']}")
        logger.info(f"   - Normal bins: {stats['bins_by_status']['normal']}")
        logger.info(f"   - Warning bins: {stats['bins_by_status']['warning']}")
        logger.info(f"   - Critical bins: {stats['bins_by_status']['critical']}")
        
        # Test pattern analysis
        logger.info("🔍 Testing pattern analysis...")
        analysis = agent._analyze_current_patterns(bin_data)
        logger.info(f"📈 Pattern analysis:")
        logger.info(f"   - Average fill level: {analysis.get('average_fill_level', 0):.1f}%")
        logger.info(f"   - Critical bins: {analysis.get('critical_bins', 0)}")
        logger.info(f"   - Predicted overflows: {analysis.get('predicted_overflows', 0)}")
        
        # Test ward distribution
        ward_distribution = analysis.get('bins_by_ward', {})
        logger.info(f"🏘️ Bins by ward:")
        for ward_id, ward_data in list(ward_distribution.items())[:5]:  # Show first 5 wards
            logger.info(f"   - Ward {ward_id}: {ward_data['count']} bins, avg fill: {ward_data['avg_fill']:.1f}%")
        
        # Test bin type distribution
        type_distribution = analysis.get('bins_by_type', {})
        logger.info(f"🗑️ Bins by type:")
        for bin_type, type_data in type_distribution.items():
            logger.info(f"   - {bin_type}: {type_data['count']} bins, avg fill: {type_data['avg_fill']:.1f}%")
        
        # Test collection simulation
        if bin_data:
            test_bin_id = bin_data[0]['id']
            logger.info(f"🚛 Testing collection simulation for bin {test_bin_id}...")
            collection_success = await agent.simulate_collection(test_bin_id)
            
            if collection_success:
                logger.info("✅ Collection simulation successful")
                
                # Get updated data
                updated_data = await agent.get_bin_data([bin_data[0]['ward_id']])
                updated_bin = next((b for b in updated_data if b['id'] == test_bin_id), None)
                
                if updated_bin:
                    logger.info(f"📊 Updated bin data after collection:")
                    logger.info(f"   - Fill Level: {updated_bin['current_fill']:.1f}%")
                    logger.info(f"   - Status: {updated_bin['status']}")
            else:
                logger.error("❌ Collection simulation failed")
        
        # Test short simulation run
        logger.info("⏱️ Testing short simulation run...")
        await agent.start_simulation(speed_multiplier=10.0)  # 10x speed for quick test
        
        # Let it run for a few seconds
        await asyncio.sleep(3)
        
        # Stop simulation
        await agent.stop_simulation()
        
        # Check updated stats
        final_stats = agent.get_simulation_stats()
        logger.info(f"📊 Final simulation stats:")
        logger.info(f"   - Data points generated: {final_stats['data_points_generated']}")
        logger.info(f"   - Collections simulated: {final_stats['collections_simulated']}")
        
        logger.info("✅ All tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    logger.info("🚀 Starting Bin Simulator Agent tests...")
    
    success = await test_bin_simulator()
    
    if success:
        logger.info("🎉 All tests passed!")
    else:
        logger.error("💥 Tests failed!")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())