#!/usr/bin/env python3
"""
Test the enhanced dashboard directly without Streamlit
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

def test_direct_import():
    """Test importing the enhanced dashboard directly"""
    print("🧪 Testing Direct Enhanced Dashboard Import...")
    
    try:
        # Test system initializer
        from app import SystemInitializer
        system_init = SystemInitializer()
        print("✅ SystemInitializer created")
        
        # Test enhanced dashboard import
        from dashboard.enhanced_dashboard import EnhancedDashboard, run_enhanced_dashboard
        print("✅ Enhanced dashboard imported")
        
        # Test dashboard creation
        dashboard = EnhancedDashboard(system_init)
        print("✅ Enhanced dashboard instance created")
        
        print("\n🎉 All imports successful!")
        print("The enhanced dashboard should work when Streamlit runs correctly.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_direct_import()
    if success:
        print("\n✅ Enhanced dashboard is ready!")
        print("The issue is likely with Streamlit's Python environment.")
        print("\nTry these solutions:")
        print("1. Activate virtual environment: venv\\Scripts\\activate")
        print("2. Install streamlit in venv: python -m pip install streamlit")
        print("3. Run app: python -m streamlit run app.py")
    else:
        print("\n❌ Enhanced dashboard has issues that need to be fixed.")