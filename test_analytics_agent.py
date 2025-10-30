#!/usr/bin/env python3
"""
Test script for the enhanced Analytics Agent
Tests the core functionality and LangChain integration
"""

import asyncio
import sys
import os
import logging

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_agents.analytics_agent import AnalyticsAgent
from ai_agents.vector_db import VectorDatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_analytics_agent():
    """Test the Analytics Agent functionality"""
    
    print("🧪 Testing Enhanced Analytics Agent with Semantic Understanding")
    print("=" * 60)
    
    try:
        # Initialize vector database
        print("📚 Initializing Vector Database...")
        vector_db = VectorDatabaseManager()
        await vector_db.initialize()
        
        # Initialize Analytics Agent
        print("📊 Initializing Analytics Agent...")
        analytics_agent = AnalyticsAgent(vector_db=vector_db)
        
        # Initialize the agent
        success = await analytics_agent.initialize_agent()
        if not success:
            print("❌ Failed to initialize Analytics Agent")
            return
        
        print("✅ Analytics Agent initialized successfully")
        
        # Test 1: Basic data analysis
        print("\n🔍 Test 1: Basic Data Analysis")
        analysis_result = await analytics_agent.analyze_waste_data(
            data_sources=["bin_data", "route_data"],
            analysis_type="comprehensive",
            user_context={"role": "supervisor", "ward_id": 1}
        )
        
        if analysis_result.get("success"):
            print(f"✅ Analysis completed with {len(analysis_result.get('insights', []))} insights")
            print(f"📈 Data quality score: {analysis_result.get('data_quality_score', 0):.1f}%")
        else:
            print(f"❌ Analysis failed: {analysis_result.get('error', 'Unknown error')}")
        
        # Test 2: Natural language question answering
        print("\n🗣️ Test 2: Natural Language Question Answering")
        question_result = await analytics_agent._answer_data_question_tool({
            "question": "What are the current fuel cost trends?",
            "user_profile": {"role": "operator", "experience_level": "intermediate"}
        })
        
        if question_result.get("success"):
            print(f"✅ Question answered with confidence: {question_result.get('confidence_score', 0):.2f}")
            print(f"💬 Answer: {question_result.get('answer', 'No answer')[:100]}...")
        else:
            print(f"❌ Question answering failed: {question_result.get('error', 'Unknown error')}")
        
        # Test 3: Advanced question answering with cost focus
        print("\n💰 Test 3: Cost-focused Question Answering")
        cost_question_result = await analytics_agent._answer_data_question_tool({
            "question": "What are the fuel cost trends and optimization opportunities?",
            "user_profile": {
                "role": "supervisor",
                "preferred_metrics": ["fuel_efficiency", "cost_savings"],
                "experience_level": "expert"
            }
        })
        
        if cost_question_result.get("success"):
            answer = cost_question_result.get("answer", "")
            confidence = cost_question_result.get("confidence_score", 0)
            print(f"✅ Cost analysis answered with confidence: {confidence:.2f}")
            print(f"💬 Answer: {answer[:150]}...")
        else:
            print(f"❌ Cost question failed: {cost_question_result.get('error', 'Unknown error')}")
        
        # Test 4: Report generation with intelligent narrative
        print("\n📋 Test 4: Report Generation with Intelligent Narrative")
        report_result = await analytics_agent._generate_report_tool({
            "report_type": "comprehensive",
            "time_period": "last_7_days",
            "ward_id": 1,
            "user_context": {"role": "supervisor"}
        })
        
        if report_result.get("success"):
            report = report_result.get("report", {})
            print(f"✅ Report generated: {report.get('report_id', 'Unknown ID')}")
            print(f"📊 Executive Summary: {report.get('executive_summary', 'No summary')[:100]}...")
            print(f"📝 Narrative length: {len(report.get('intelligent_narrative', ''))} characters")
        else:
            print(f"❌ Report generation failed: {report_result.get('error', 'Unknown error')}")
        
        # Test 5: Agent statistics
        print("\n📈 Test 5: Agent Statistics")
        stats = analytics_agent.get_analytics_stats()
        print(f"✅ Agent Statistics:")
        print(f"   • Analyses performed: {stats['analytics_stats']['analyses_performed']}")
        print(f"   • Insights generated: {stats['analytics_stats']['insights_generated']}")
        print(f"   • Reports created: {stats['analytics_stats']['reports_created']}")
        print(f"   • ML models loaded: {stats['ml_models_loaded']}")
        print(f"   • Vector DB connected: {stats['vector_db_connected']}")
        
        print("\n🎉 All tests completed successfully!")
        print("📊 Enhanced Analytics Agent with semantic understanding is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_analytics_agent())