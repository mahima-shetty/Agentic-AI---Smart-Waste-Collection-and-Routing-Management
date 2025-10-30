#!/usr/bin/env python3
"""
Simple health check script for Smart Waste Management System
Tests basic functionality without running the full Streamlit app
"""

import sys
import os
import sqlite3
from pathlib import Path

# Add project directories to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ai_agents"))
sys.path.insert(0, str(project_root / "backend"))
sys.path.insert(0, str(project_root / "dashboard"))

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas imported successfully")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import plotly.express as px
        print("✅ Plotly imported successfully")
    except ImportError as e:
        print(f"❌ Plotly import failed: {e}")
        return False
    
    try:
        import chromadb
        print("✅ ChromaDB imported successfully")
    except ImportError as e:
        print(f"❌ ChromaDB import failed: {e}")
        return False
    
    try:
        import redis
        print("✅ Redis imported successfully")
    except ImportError as e:
        print("⚠️ Redis import failed (optional): {e}")
    
    return True

def test_database():
    """Test database connectivity"""
    print("\n🗄️ Testing database...")
    
    db_path = "backend/db/operators.db"
    
    if not os.path.exists(db_path):
        print(f"❌ Database file not found: {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Test operators table
        cursor.execute("SELECT COUNT(*) FROM operators")
        operator_count = cursor.fetchone()[0]
        print(f"✅ Database connected - {operator_count} operators found")
        
        # Test a sample query
        cursor.execute("SELECT name, email, ward_id FROM operators LIMIT 3")
        operators = cursor.fetchall()
        
        print("📋 Sample operators:")
        for op in operators:
            print(f"   - {op[0]} ({op[1]}) - Ward {op[2]}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_dashboard_imports():
    """Test dashboard module imports"""
    print("\n🎛️ Testing dashboard modules...")
    
    try:
        import dashboard.enhanced_dashboard
        print("✅ Enhanced dashboard imported successfully")
    except ImportError as e:
        print(f"❌ Enhanced dashboard import failed: {e}")
        return False
    
    try:
        import dashboard.auth
        print("✅ Authentication module imported successfully")
    except ImportError as e:
        print(f"❌ Authentication import failed: {e}")
        return False
    
    return True

def test_ai_agents():
    """Test AI agent imports"""
    print("\n🤖 Testing AI agent modules...")
    
    try:
        from ai_agents.master_agent import MasterCoordinationAgent
        print("✅ Master agent imported successfully")
    except ImportError as e:
        print(f"❌ Master agent import failed: {e}")
        return False
    
    try:
        from ai_agents.bin_simulator import BinSimulatorAgent
        print("✅ Bin simulator imported successfully")
    except ImportError as e:
        print(f"❌ Bin simulator import failed: {e}")
        return False
    
    return True

def test_chromadb():
    """Test ChromaDB functionality"""
    print("\n📚 Testing ChromaDB...")
    
    try:
        import chromadb
        
        # Create a test client
        client = chromadb.Client()
        
        # Try to create a test collection
        test_collection = client.create_collection("test_collection")
        print("✅ ChromaDB test collection created")
        
        # Clean up
        client.delete_collection("test_collection")
        print("✅ ChromaDB test collection deleted")
        
        return True
        
    except Exception as e:
        print(f"❌ ChromaDB test failed: {e}")
        return False

def test_config_files():
    """Test configuration files"""
    print("\n⚙️ Testing configuration files...")
    
    config_files = [
        "data/config.json",
        "dashboard/requirements.txt"
    ]
    
    all_good = True
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✅ Found: {config_file}")
        else:
            print(f"⚠️ Missing: {config_file}")
            all_good = False
    
    return all_good

def main():
    """Run all health checks"""
    print("🏥 Smart Waste Management System - Health Check")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_imports),
        ("Database", test_database),
        ("Dashboard Modules", test_dashboard_imports),
        ("AI Agents", test_ai_agents),
        ("ChromaDB", test_chromadb),
        ("Configuration Files", test_config_files)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Health Check Summary:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        print("🎉 All systems operational! The app should work correctly.")
        print("\n🚀 To start the app, run: streamlit run app.py")
    elif passed >= total * 0.7:
        print("⚠️ Most systems operational. App may work with limited functionality.")
        print("\n🚀 To start the app, run: streamlit run app.py")
    else:
        print("❌ Critical issues detected. Please fix the failing components.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)