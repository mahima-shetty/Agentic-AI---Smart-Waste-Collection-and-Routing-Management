#!/usr/bin/env python3
"""
Smart Waste Management System - Quick Start Version
Simplified launcher that skips full AI agent initialization for faster startup
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add project directories to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "dashboard"))

# Configure Streamlit page
st.set_page_config(
    page_title="Smart Waste Management System",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Quick start main function"""
    
    # Create a simple system initializer mock
    class QuickSystemInitializer:
        def __init__(self):
            self.redis_client = None
            self.chroma_client = None
            self.master_agent = None
            self.components_initialized = True  # Mock as initialized
    
    system_initializer = QuickSystemInitializer()
    
    # Try to import and run enhanced dashboard
    try:
        # Direct import method
        dashboard_path = str(project_root / "dashboard")
        if dashboard_path not in sys.path:
            sys.path.insert(0, dashboard_path)
        
        import enhanced_dashboard
        enhanced_dashboard.run_enhanced_dashboard(system_initializer)
        
    except Exception as e:
        st.error(f"❌ Enhanced dashboard failed: {e}")
        
        # Fallback to basic interface
        st.title("🗑️ Smart Waste Management System")
        st.subheader("Quick Start Mode")
        
        st.info("""
        **Quick Start Mode Active**
        
        This is a simplified version for faster loading. 
        
        **Demo Login Credentials:**
        - Email: amit.sharma.a@bmc.gov.in
        - Password: amitA@123
        
        **To access full features:**
        1. Wait for the main app to finish initialization
        2. Or restart with: `venv\\Scripts\\python.exe -m streamlit run app.py`
        """)
        
        # Simple login form
        with st.form("quick_login"):
            st.subheader("🔐 Quick Login")
            email = st.text_input("Email", value="amit.sharma.a@bmc.gov.in")
            password = st.text_input("Password", type="password")
            
            if st.form_submit_button("Login"):
                if email and password:
                    st.success("✅ Login successful! (Demo mode)")
                    
                    # Show demo dashboard
                    st.subheader("📊 Demo Dashboard")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("🗑️ Total Bins", "100", delta="24 wards")
                    
                    with col2:
                        st.metric("🚨 Active Alerts", "5", delta="-2")
                    
                    with col3:
                        st.metric("📧 Emails Sent", "28", delta="+8")
                    
                    with col4:
                        st.metric("✅ Collection Rate", "94.2%", delta="2.1%")
                    
                    st.info("🚧 This is demo mode. For full functionality, use the main app.")
                else:
                    st.error("Please enter email and password")

if __name__ == "__main__":
    main()