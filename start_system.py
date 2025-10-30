#!/usr/bin/env python3
"""
Smart Waste Management System - Startup Script
Simple script to launch the system with proper initialization
"""

import sys
import subprocess
import os
from pathlib import Path

def check_redis():
    """Check if Redis is running"""
    try:
        import redis
        client = redis.Redis(host='localhost', port=6379, db=0, socket_connect_timeout=2)
        client.ping()
        print("✅ Redis is running")
        return True
    except:
        print("⚠️  Redis is not running - starting without Redis (limited functionality)")
        return False

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def main():
    """Main startup function"""
    print("🚀 Starting Smart Waste Management System...")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("app.py").exists():
        print("❌ app.py not found. Please run this script from the project root directory.")
        sys.exit(1)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies if needed
    if "--install-deps" in sys.argv:
        if not install_dependencies():
            sys.exit(1)
    
    # Check Redis status
    redis_available = check_redis()
    
    # Create necessary directories
    directories = [
        "data/chromadb",
        "backend/db",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    print("=" * 50)
    
    # Launch the application
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("🧪 Running system tests...")
        subprocess.run([sys.executable, "app.py", "test"])
    else:
        print("🌐 Launching Streamlit dashboard...")
        print("📝 The dashboard will open in your default web browser")
        print("🔗 URL: http://localhost:8501")
        print("=" * 50)
        
        # Set environment variables for Streamlit
        env = os.environ.copy()
        env["STREAMLIT_SERVER_HEADLESS"] = "true"
        env["STREAMLIT_SERVER_PORT"] = "8501"
        
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.headless", "true",
            "--server.port", "8501",
            "--server.address", "localhost"
        ], env=env)

if __name__ == "__main__":
    main()