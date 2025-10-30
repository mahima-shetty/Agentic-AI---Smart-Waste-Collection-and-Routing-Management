#!/usr/bin/env python3
"""
Test script to verify the enhanced dashboard works correctly
"""

import sys
import os
from pathlib import Path

# Add project directories to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ai_agents"))
sys.path.insert(0, str(project_root / "backend"))
sys.path.insert(0, str(project_root / "dashboard"))

def test_enhanced_dashboard_import():
    """Test if enhanced dashboard can be imported"""
    print("🧪 Testing Enhanced Dashboard Import...")
    
    try:
        from dashboard.enhanced_dashboard import run_enhanced_dashboard, EnhancedDashboard
        print("✅ Enhanced dashboard imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Enhanced dashboard import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_system_initializer():
    """Test if SystemInitializer can be created"""
    print("\n🧪 Testing System Initializer...")
    
    try:
        from app import SystemInitializer
        
        # Create system initializer
        system_init = SystemInitializer()
        print("✅ SystemInitializer created successfully")
        
        # Test basic properties
        print(f"   - Redis client: {system_init.redis_client is not None}")
        print(f"   - ChromaDB client: {system_init.chroma_client is not None}")
        print(f"   - Components initialized: {system_init.components_initialized}")
        
        return True
        
    except Exception as e:
        print(f"❌ SystemInitializer test failed: {e}")
        return False

def test_dashboard_creation():
    """Test if EnhancedDashboard can be created"""
    print("\n🧪 Testing Dashboard Creation...")
    
    try:
        from app import SystemInitializer
        from dashboard.enhanced_dashboard import EnhancedDashboard
        
        # Create system initializer
        system_init = SystemInitializer()
        
        # Create dashboard
        dashboard = EnhancedDashboard(system_init)
        print("✅ EnhancedDashboard created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Dashboard creation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Enhanced Dashboard Test Suite")
    print("=" * 40)
    
    tests = [
        ("Enhanced Dashboard Import", test_enhanced_dashboard_import),
        ("System Initializer", test_system_initializer),
        ("Dashboard Creation", test_dashboard_creation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 Test Results:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced dashboard should work.")
        print("\n🚀 Try running: python -m streamlit run app.py")
    else:
        print("❌ Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)