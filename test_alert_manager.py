#!/usr/bin/env python3
"""
Test script for Alert Management Agent
Tests basic functionality and alert generation
"""

import asyncio
import sys
import os
import logging
from datetime import datetime

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_agents.alert_manager import AlertManagementAgent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_alert_manager():
    """Test the Alert Management Agent functionality"""
    
    print("🧪 Testing Alert Management Agent...")
    
    # Initialize the agent
    agent = AlertManagementAgent()
    
    # Test initialization
    print("\n1. Testing agent initialization...")
    success = await agent.initialize_agent()
    if success:
        print("✅ Agent initialized successfully")
    else:
        print("❌ Agent initialization failed")
        return
    
    # Test alert generation
    print("\n2. Testing alert generation...")
    
    # Sample bin data for testing
    test_bin_data = {
        "id": "BIN_01_001",
        "current_fill": 87.5,
        "fill_rate": 2.3,
        "capacity": 240,
        "bin_type": "commercial",
        "location_type": "market",
        "ward_id": 1,
        "latitude": 19.0760,
        "longitude": 72.8777
    }
    
    # Generate a warning alert
    await agent._generate_overflow_alert(test_bin_data, "overflow_warning")
    
    # Check if alert was created
    active_alerts = agent.get_active_alerts()
    if active_alerts:
        print(f"✅ Generated {len(active_alerts)} alert(s)")
        for alert in active_alerts:
            print(f"   - Alert ID: {alert['id']}")
            print(f"   - Type: {alert['alert_type']}")
            print(f"   - Priority: {alert['priority']}")
            print(f"   - Message: {alert['message']}")
            print(f"   - Summary: {alert['natural_language_summary']}")
            print(f"   - Action: {alert['recommended_action']}")
    else:
        print("❌ No alerts generated")
    
    # Test overflow prediction
    print("\n3. Testing overflow prediction...")
    
    predicted_overflow = await agent._predict_bin_overflow(test_bin_data)
    if predicted_overflow:
        print(f"✅ Predicted overflow at: {predicted_overflow}")
    else:
        print("ℹ️ No overflow predicted within horizon")
    
    # Test priority calculation
    print("\n4. Testing priority calculation...")
    
    priority = await agent._calculate_alert_priority(test_bin_data, "overflow_critical")
    print(f"✅ Calculated priority: {priority}")
    
    # Test contextual message generation
    print("\n5. Testing contextual message generation...")
    
    message = await agent._generate_contextual_message(test_bin_data, "overflow_warning")
    print(f"✅ Generated message: {message}")
    
    # Test natural language summary
    print("\n6. Testing natural language summary...")
    
    summary = await agent._generate_natural_language_summary(test_bin_data, "overflow_warning", [])
    print(f"✅ Generated summary: {summary}")
    
    # Test recommended action
    print("\n7. Testing recommended action generation...")
    
    action = await agent._generate_recommended_action(test_bin_data, "overflow_warning", [])
    print(f"✅ Generated action: {action}")
    
    # Test alert statistics
    print("\n8. Testing alert statistics...")
    
    stats = agent.get_alert_stats()
    print(f"✅ Alert statistics:")
    print(f"   - Total alerts generated: {stats['total_alerts_generated']}")
    print(f"   - Active alerts: {stats['active_alerts_count']}")
    print(f"   - Monitoring active: {stats['monitoring_active']}")
    print(f"   - Model trained: {stats['model_trained']}")
    
    # Test alert resolution
    print("\n9. Testing alert resolution...")
    
    if active_alerts:
        alert_id = active_alerts[0]['id']
        resolved = await agent.resolve_alert(alert_id, "collected")
        if resolved:
            print(f"✅ Resolved alert {alert_id}")
        else:
            print(f"❌ Failed to resolve alert {alert_id}")
    
    # Test tools
    print("\n10. Testing agent tools...")
    
    tools = agent.get_tools()
    print(f"✅ Agent has {len(tools)} tools:")
    for tool in tools:
        print(f"   - {tool.name}: {tool.description}")
    
    print("\n🎉 Alert Management Agent testing completed!")

if __name__ == "__main__":
    asyncio.run(test_alert_manager())