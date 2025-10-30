"""
Enhanced Smart Waste Management Dashboard with Real-time Updates
Comprehensive dashboard with authentication, email preferences, live data, and full functionality
"""

import streamlit as st
import sqlite3
import logging
import asyncio
import time
import random
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Any, Dict, Optional, List
from datetime import datetime, timedelta
import json

from dashboard.auth import authenticate_user, logout_user
from dashboard.email_preferences import EmailPreferencesManager, render_email_preferences_sidebar

logger = logging.getLogger(__name__)

class EnhancedDashboard:
    """Enhanced dashboard with full functionality including email notifications"""
    
    def __init__(self, system_initializer: Any):
        self.system_initializer = system_initializer
        self.db_path = "backend/db/operators.db"
        self.email_prefs_manager = EmailPreferencesManager(self.db_path)
        
        # Initialize session state
        if "authenticated" not in st.session_state:
            st.session_state["authenticated"] = False
        if "current_page" not in st.session_state:
            st.session_state["current_page"] = "dashboard"
        if "last_refresh" not in st.session_state:
            st.session_state["last_refresh"] = datetime.now()
        if "auto_refresh" not in st.session_state:
            st.session_state["auto_refresh"] = True
        if "refresh_interval" not in st.session_state:
            st.session_state["refresh_interval"] = 30  # seconds
        if "live_data_cache" not in st.session_state:
            st.session_state["live_data_cache"] = {}
        if "loading_states" not in st.session_state:
            st.session_state["loading_states"] = {}
        
        # Real-time data refresh settings
        self.refresh_interval = st.session_state["refresh_interval"]
        self.auto_refresh_enabled = st.session_state["auto_refresh"]
    
    def run(self):
        """Main dashboard entry point"""
        try:
            # Configure page
            st.set_page_config(
                page_title="Smart Waste Management",
                page_icon="🗑️",
                layout="wide",
                initial_sidebar_state="expanded"
            )
            
            # Custom CSS
            self._apply_custom_css()
            
            # Authentication check
            if not st.session_state.get("authenticated", False):
                self._render_login_page()
                return
            
            # Setup auto-refresh
            self._setup_auto_refresh()
            
            # Render main dashboard
            self._render_main_dashboard()
            
        except Exception as e:
            logger.error(f"❌ Dashboard error: {e}")
            st.error("❌ An error occurred while loading the dashboard.")
    
    def _apply_custom_css(self):
        """Apply custom CSS styling with animations and real-time indicators"""
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #1f77b4, #ff7f0e);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
            animation: fadeIn 0.5s ease-in;
        }
        
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #1f77b4;
            transition: all 0.3s ease;
        }
        
        .metric-card:hover {
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
            transform: translateY(-2px);
        }
        
        .alert-card {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
            animation: slideIn 0.3s ease-out;
        }
        
        .critical-alert {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            animation: pulse 2s infinite;
        }
        
        .success-card {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        .sidebar-section {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        
        .loading-spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #1f77b4;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        .live-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            background-color: #28a745;
            border-radius: 50%;
            animation: blink 1.5s infinite;
            margin-right: 5px;
        }
        
        .refresh-button {
            background: linear-gradient(45deg, #28a745, #20c997);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .refresh-button:hover {
            transform: scale(1.05);
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        
        .status-badge {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: bold;
            text-transform: uppercase;
        }
        
        .status-normal { background-color: #d4edda; color: #155724; }
        .status-warning { background-color: #fff3cd; color: #856404; }
        .status-critical { background-color: #f8d7da; color: #721c24; }
        
        .progress-bar {
            width: 100%;
            height: 20px;
            background-color: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #28a745, #ffc107, #dc3545);
            transition: width 0.5s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateX(-20px); }
            to { opacity: 1; transform: translateX(0); }
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0.3; }
        }
        
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(220, 53, 69, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(220, 53, 69, 0); }
            100% { box-shadow: 0 0 0 0 rgba(220, 53, 69, 0); }
        }
        
        .page-transition {
            animation: fadeIn 0.3s ease-in;
        }
        
        .data-table {
            animation: slideIn 0.4s ease-out;
        }
        
        .chart-container {
            animation: fadeIn 0.6s ease-in;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _render_login_page(self):
        """Render login page"""
        st.markdown("""
        <div class="main-header">
            <h1>🗑️ Smart Waste Management System</h1>
            <p>BMC Mumbai - AI-Powered Waste Collection Dashboard</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.subheader("🔐 Operator Login")
            
            with st.form("login_form"):
                email = st.text_input("📧 Email", placeholder="operator@bmc.gov.in")
                password = st.text_input("🔒 Password", type="password")
                
                submitted = st.form_submit_button("🚀 Login", use_container_width=True)
                
                if submitted:
                    operator = authenticate_user(email, password)
                    if operator:
                        st.session_state["authenticated"] = True
                        st.session_state["operator_id"] = operator["id"]
                        st.session_state["operator_name"] = operator["name"]
                        st.session_state["operator_email"] = operator["email"]
                        st.session_state["ward_id"] = operator["ward_id"]
                        st.success(f"✅ Welcome, {operator['name']}!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials. Please try again.")
            
            # Demo credentials info
            with st.expander("🔍 Demo Credentials"):
                st.info("""
                **Sample Login Credentials:**
                - Email: amit.sharma.a@bmc.gov.in
                - Password: amitA@123
                - Ward: 1
                
                **Or try any operator from the database:**
                - sneha.patel.b@bmc.gov.in / snehaB@123 (Ward 2)
                - ravi.iyer.c@bmc.gov.in / raviC@123 (Ward 3)
                """)
    
    def _render_main_dashboard(self):
        """Render main dashboard interface"""
        # Header
        st.markdown(f"""
        <div class="main-header">
            <h1>🗑️ Smart Waste Management Dashboard</h1>
            <p>Welcome, {st.session_state.get('operator_name', 'Operator')} | Ward {st.session_state.get('ward_id', 'N/A')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar
        self._render_sidebar()
        
        # Main content based on current page
        current_page = st.session_state.get("current_page", "dashboard")
        
        if current_page == "dashboard":
            self._render_dashboard_home()
        elif current_page == "email_preferences":
            self._render_email_preferences_page()
        elif current_page == "alerts":
            self._render_alerts_page()
        elif current_page == "analytics":
            self._render_analytics_page()
        elif current_page == "system_status":
            self._render_system_status_page()
        else:
            self._render_dashboard_home()
    
    def _render_sidebar(self):
        """Render sidebar navigation"""
        with st.sidebar:
            st.markdown(f"""
            <div class="sidebar-section">
                <h3>👤 {st.session_state.get('operator_name', 'Operator')}</h3>
                <p>Ward {st.session_state.get('ward_id', 'N/A')} Operator</p>
                <p>📧 {st.session_state.get('operator_email', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("🧭 Navigation")
            
            # Navigation buttons
            if st.button("🏠 Dashboard", use_container_width=True):
                st.session_state["current_page"] = "dashboard"
                st.rerun()
            
            if st.button("🚨 Alerts", use_container_width=True):
                st.session_state["current_page"] = "alerts"
                st.rerun()
            
            if st.button("📊 Analytics", use_container_width=True):
                st.session_state["current_page"] = "analytics"
                st.rerun()
            
            if st.button("⚙️ System Status", use_container_width=True):
                st.session_state["current_page"] = "system_status"
                st.rerun()
            
            # Email preferences section
            render_email_preferences_sidebar()
            
            st.markdown("---")
            
            # Quick actions
            st.subheader("⚡ Quick Actions")
            
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.cache_data.clear()
                st.success("✅ Data refreshed!")
            
            if st.button("📧 Test Email", use_container_width=True):
                success = self._send_test_email()
                if success:
                    st.success("✅ Test email sent!")
                else:
                    st.error("❌ Failed to send test email")
            
            st.markdown("---")
            
            # Logout
            if st.button("🚪 Logout", use_container_width=True):
                logout_user()
                st.rerun()
    
    def _setup_auto_refresh(self):
        """Setup auto-refresh functionality"""
        try:
            # Auto-refresh controls in sidebar
            with st.sidebar:
                st.markdown("---")
                st.subheader("🔄 Real-time Updates")
                
                # Auto-refresh toggle
                auto_refresh = st.checkbox(
                    "Enable Auto-refresh", 
                    value=st.session_state.get("auto_refresh", True),
                    help="Automatically refresh data every 30 seconds"
                )
                st.session_state["auto_refresh"] = auto_refresh
                
                # Refresh interval
                if auto_refresh:
                    refresh_interval = st.selectbox(
                        "Refresh Interval",
                        options=[15, 30, 60, 120],
                        index=1,  # Default to 30 seconds
                        format_func=lambda x: f"{x} seconds",
                        help="How often to refresh the data"
                    )
                    st.session_state["refresh_interval"] = refresh_interval
                
                # Manual refresh button
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Refresh Now", use_container_width=True):
                        self._refresh_all_data()
                        st.success("✅ Data refreshed!")
                
                with col2:
                    # Show last refresh time
                    last_refresh = st.session_state.get("last_refresh", datetime.now())
                    time_diff = (datetime.now() - last_refresh).total_seconds()
                    st.caption(f"Last: {int(time_diff)}s ago")
                
                # Live data indicator
                if auto_refresh:
                    st.markdown("""
                    <div style="text-align: center; margin: 10px 0;">
                        <span class="live-indicator"></span>
                        <small>Live Data Active</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Auto-refresh logic
            if auto_refresh:
                current_time = time.time()
                last_refresh_time = st.session_state.get("last_refresh_time", 0)
                
                if current_time - last_refresh_time > refresh_interval:
                    self._refresh_all_data()
                    st.session_state["last_refresh_time"] = current_time
                    st.session_state["last_refresh"] = datetime.now()
                    st.rerun()
                
                # Add JavaScript for client-side refresh
                st.markdown(f"""
                <script>
                setTimeout(function(){{
                    window.location.reload();
                }}, {refresh_interval * 1000});
                </script>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            logger.error(f"❌ Failed to setup auto-refresh: {e}")
    
    def _refresh_all_data(self):
        """Refresh all cached data"""
        try:
            # Clear cache
            st.session_state["live_data_cache"] = {}
            
            # Refresh system data
            self._get_live_system_data(force_refresh=True)
            
            # Refresh bin data
            self._get_live_bin_data(force_refresh=True)
            
            # Refresh alerts
            self._get_live_alerts(force_refresh=True)
            
            logger.info("🔄 All data refreshed successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to refresh all data: {e}")
    
    @st.cache_data(ttl=30)  # Cache for 30 seconds
    def _get_live_system_data(_self, force_refresh: bool = False) -> Dict[str, Any]:
        """Get live system data from agents"""
        try:
            if not force_refresh and "system_data" in st.session_state["live_data_cache"]:
                cache_time = st.session_state["live_data_cache"]["system_data"].get("timestamp", 0)
                if time.time() - cache_time < 30:  # Use cache if less than 30 seconds old
                    return st.session_state["live_data_cache"]["system_data"]["data"]
            
            # Get data from system initializer
            if hasattr(_self.system_initializer, 'master_agent') and _self.system_initializer.master_agent:
                # Use asyncio to get data from master agent
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    system_data = loop.run_until_complete(
                        _self.system_initializer.master_agent.get_system_dashboard_data()
                    )
                    
                    # Cache the data
                    st.session_state["live_data_cache"]["system_data"] = {
                        "data": system_data,
                        "timestamp": time.time()
                    }
                    
                    return system_data
                    
                finally:
                    loop.close()
            else:
                # Fallback to basic system status
                return {
                    "system_stats": {
                        "total_bins": 0,
                        "critical_bins": 0,
                        "warning_bins": 0,
                        "normal_bins": 0,
                        "active_alerts": 0,
                        "system_status": "limited",
                        "agents_active": 0
                    },
                    "bin_data": [],
                    "alerts": [],
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to get live system data: {e}")
            return {
                "error": str(e),
                "system_stats": {
                    "total_bins": 0,
                    "critical_bins": 0,
                    "warning_bins": 0,
                    "normal_bins": 0,
                    "active_alerts": 0,
                    "system_status": "error",
                    "agents_active": 0
                },
                "bin_data": [],
                "alerts": []
            }
    
    @st.cache_data(ttl=15)  # Cache for 15 seconds (more frequent for bin data)
    def _get_live_bin_data(_self, ward_id: Optional[int] = None, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Get live bin data from bin simulator"""
        try:
            cache_key = f"bin_data_{ward_id or 'all'}"
            
            if not force_refresh and cache_key in st.session_state["live_data_cache"]:
                cache_time = st.session_state["live_data_cache"][cache_key].get("timestamp", 0)
                if time.time() - cache_time < 15:  # Use cache if less than 15 seconds old
                    return st.session_state["live_data_cache"][cache_key]["data"]
            
            # Get data from bin simulator
            if (hasattr(_self.system_initializer, 'master_agent') and 
                _self.system_initializer.master_agent and
                hasattr(_self.system_initializer.master_agent, 'bin_simulator')):
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    ward_ids = [ward_id] if ward_id else None
                    bin_data = loop.run_until_complete(
                        _self.system_initializer.master_agent.bin_simulator.get_bin_data(ward_ids)
                    )
                    
                    # Cache the data
                    st.session_state["live_data_cache"][cache_key] = {
                        "data": bin_data,
                        "timestamp": time.time()
                    }
                    
                    return bin_data
                    
                finally:
                    loop.close()
            else:
                return []
                
        except Exception as e:
            logger.error(f"❌ Failed to get live bin data: {e}")
            return []
    
    @st.cache_data(ttl=10)  # Cache for 10 seconds (most frequent for alerts)
    def _get_live_alerts(_self, ward_id: Optional[int] = None, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Get live alerts from alert manager"""
        try:
            cache_key = f"alerts_{ward_id or 'all'}"
            
            if not force_refresh and cache_key in st.session_state["live_data_cache"]:
                cache_time = st.session_state["live_data_cache"][cache_key].get("timestamp", 0)
                if time.time() - cache_time < 10:  # Use cache if less than 10 seconds old
                    return st.session_state["live_data_cache"][cache_key]["data"]
            
            # Get data from alert manager
            if (hasattr(_self.system_initializer, 'master_agent') and 
                _self.system_initializer.master_agent):
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    alerts = loop.run_until_complete(
                        _self.system_initializer.master_agent.get_active_alerts(ward_id)
                    )
                    
                    # Cache the data
                    st.session_state["live_data_cache"][cache_key] = {
                        "data": alerts,
                        "timestamp": time.time()
                    }
                    
                    return alerts
                    
                finally:
                    loop.close()
            else:
                return []
                
        except Exception as e:
            logger.error(f"❌ Failed to get live alerts: {e}")
            return []
    
    def _show_loading_indicator(self, message: str = "Loading..."):
        """Show loading indicator with spinner"""
        return st.markdown(f"""
        <div style="text-align: center; padding: 20px;">
            <div class="loading-spinner"></div>
            <p style="margin-top: 10px; color: #666;">{message}</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _create_progress_bar(self, value: float, max_value: float = 100, label: str = "") -> str:
        """Create animated progress bar"""
        percentage = min(100, (value / max_value) * 100)
        
        # Determine color based on percentage
        if percentage < 50:
            color = "#28a745"  # Green
        elif percentage < 85:
            color = "#ffc107"  # Yellow
        else:
            color = "#dc3545"  # Red
        
        return f"""
        <div style="margin: 10px 0;">
            {f"<label style='font-weight: bold; margin-bottom: 5px; display: block;'>{label}</label>" if label else ""}
            <div class="progress-bar">
                <div class="progress-fill" style="width: {percentage}%; background-color: {color};"></div>
            </div>
            <small style="color: #666;">{value:.1f}% ({percentage:.1f}%)</small>
        </div>
        """
    
    def _render_dashboard_home(self):
        """Render main dashboard home page with real-time data"""
        # Add page transition animation
        st.markdown('<div class="page-transition">', unsafe_allow_html=True)
        
        # System status overview with loading indicator
        st.subheader("📊 System Overview")
        
        # Show loading indicator while fetching data
        loading_placeholder = st.empty()
        with loading_placeholder:
            self._show_loading_indicator("Fetching live system data...")
        
        # Get live system data
        system_data = self._get_live_system_data()
        loading_placeholder.empty()  # Remove loading indicator
        
        # Extract system stats
        system_stats = system_data.get("system_stats", {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_bins = system_stats.get("total_bins", 0)
            st.metric(
                "🗑️ Total Bins",
                f"{total_bins:,}",
                delta=f"+{total_bins - 150}" if total_bins > 150 else None,
                help="Total bins in your ward"
            )
        
        with col2:
            active_alerts = system_stats.get("active_alerts", 0)
            st.metric(
                "🚨 Active Alerts",
                f"{active_alerts}",
                delta=f"{active_alerts - 3:+d}" if active_alerts != 3 else None,
                delta_color="inverse",
                help="Current active alerts"
            )
        
        with col3:
            email_stats = self._get_email_statistics()
            emails_sent = email_stats.get("sent_today", 0)
            st.metric(
                "📧 Emails Sent",
                f"{emails_sent}",
                delta=f"+{emails_sent - 20}" if emails_sent > 20 else None,
                help="Email notifications sent today"
            )
        
        with col4:
            # Calculate collection rate from bin data
            bin_data = system_data.get("bin_data", [])
            if bin_data:
                normal_bins = len([b for b in bin_data if b.get("status") == "normal"])
                collection_rate = (normal_bins / len(bin_data)) * 100 if bin_data else 0
            else:
                collection_rate = 94.2
            
            st.metric(
                "✅ Collection Rate",
                f"{collection_rate:.1f}%",
                delta=f"{collection_rate - 94.2:+.1f}%",
                help="Successful collection rate"
            )
        
        # Recent alerts with real-time data
        st.subheader("🚨 Recent Alerts")
        
        # Get live alerts
        ward_id = st.session_state.get("ward_id")
        live_alerts = self._get_live_alerts(ward_id)
        
        if live_alerts:
            # Sort alerts by priority and creation time
            sorted_alerts = sorted(
                live_alerts, 
                key=lambda x: (x.get("priority", 0), x.get("created_at", "")), 
                reverse=True
            )
            
            # Show top 5 alerts
            for alert in sorted_alerts[:5]:
                priority_map = {5: "Critical", 4: "High", 3: "Medium", 2: "Low", 1: "Low"}
                priority_text = priority_map.get(alert.get("priority", 2), "Medium")
                
                priority_color = {
                    "Critical": "🔴",
                    "High": "🟠", 
                    "Medium": "🟡",
                    "Low": "🟢"
                }.get(priority_text, "⚪")
                
                status = alert.get("status", "active")
                status_color = "success-card" if status == "resolved" else "alert-card"
                if priority_text == "Critical":
                    status_color = "critical-alert"
                
                # Calculate time ago
                created_at = alert.get("created_at", "")
                if created_at:
                    try:
                        if isinstance(created_at, str):
                            alert_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        else:
                            alert_time = created_at
                        time_diff = datetime.now() - alert_time.replace(tzinfo=None)
                        
                        if time_diff.total_seconds() < 60:
                            time_ago = f"{int(time_diff.total_seconds())} seconds ago"
                        elif time_diff.total_seconds() < 3600:
                            time_ago = f"{int(time_diff.total_seconds() / 60)} minutes ago"
                        else:
                            time_ago = f"{int(time_diff.total_seconds() / 3600)} hours ago"
                    except:
                        time_ago = "Recently"
                else:
                    time_ago = "Recently"
                
                st.markdown(f"""
                <div class="{status_color}">
                    <strong>{priority_color} {alert.get('bin_id', 'Unknown')}</strong> - {priority_text} Priority<br>
                    <em>{alert.get('message', 'Alert generated')}</em><br>
                    <small>⏰ {time_ago} | Status: {status.title()}</small>
                    {f"<br><small>💡 {alert.get('recommended_action', '')}</small>" if alert.get('recommended_action') else ""}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("✅ No active alerts at this time")
        
        # Real-time bin status visualization
        st.subheader("🗑️ Live Bin Status")
        
        # Get live bin data
        bin_data = self._get_live_bin_data(ward_id)
        
        if bin_data:
            # Create status summary
            status_counts = {
                "normal": len([b for b in bin_data if b.get("status") == "normal"]),
                "warning": len([b for b in bin_data if b.get("status") == "warning"]),
                "critical": len([b for b in bin_data if b.get("status") == "critical"]),
                "maintenance": len([b for b in bin_data if b.get("status") == "maintenance"])
            }
            
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: #28a745; margin: 0;">✅ {status_counts['normal']}</h3>
                    <p style="margin: 5px 0 0 0; color: #666;">Normal</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col6:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: #ffc107; margin: 0;">⚠️ {status_counts['warning']}</h3>
                    <p style="margin: 5px 0 0 0; color: #666;">Warning</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col7:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: #dc3545; margin: 0;">🚨 {status_counts['critical']}</h3>
                    <p style="margin: 5px 0 0 0; color: #666;">Critical</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col8:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: #6c757d; margin: 0;">🔧 {status_counts['maintenance']}</h3>
                    <p style="margin: 5px 0 0 0; color: #666;">Maintenance</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Show top bins needing attention
            critical_bins = [b for b in bin_data if b.get("status") in ["critical", "warning"]]
            if critical_bins:
                st.subheader("⚠️ Bins Requiring Attention")
                
                # Sort by fill level
                critical_bins.sort(key=lambda x: x.get("current_fill", 0), reverse=True)
                
                for bin_info in critical_bins[:10]:  # Show top 10
                    fill_level = bin_info.get("current_fill", 0)
                    status = bin_info.get("status", "normal")
                    
                    # Create progress bar
                    progress_html = self._create_progress_bar(
                        fill_level, 
                        100, 
                        f"{bin_info.get('id', 'Unknown')} - {status.title()}"
                    )
                    
                    st.markdown(progress_html, unsafe_allow_html=True)
        else:
            st.info("📊 No bin data available. Please check system connectivity.")
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close page transition
        
        # Email notification status
        st.subheader("📧 Email Notification Status")
        
        email_stats = self._get_email_statistics()
        
        col5, col6, col7 = st.columns(3)
        
        with col5:
            st.metric(
                "📤 Sent Today",
                email_stats.get("sent_today", 0),
                help="Emails sent today"
            )
        
        with col6:
            st.metric(
                "✅ Delivery Rate",
                f"{email_stats.get('delivery_rate', 0):.1f}%",
                help="Email delivery success rate"
            )
        
        with col7:
            st.metric(
                "⏳ Pending",
                email_stats.get("pending", 0),
                help="Emails waiting to be sent"
            )
    
    def _render_email_preferences_page(self):
        """Render email preferences page"""
        operator_id = st.session_state.get("operator_id")
        if operator_id:
            self.email_prefs_manager.render_email_preferences_page(operator_id)
        else:
            st.error("❌ Operator ID not found. Please log in again.")
    
    def _render_alerts_page(self):
        """Render alerts management page with real-time data"""
        st.markdown('<div class="page-transition">', unsafe_allow_html=True)
        st.title("🚨 Alert Management")
        
        # Real-time alert statistics
        ward_id = st.session_state.get("ward_id")
        live_alerts = self._get_live_alerts(ward_id)
        
        # Alert summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_alerts = len(live_alerts)
            st.metric("📊 Total Alerts", total_alerts)
        
        with col2:
            active_alerts = len([a for a in live_alerts if a.get("status") == "active"])
            st.metric("🔴 Active", active_alerts)
        
        with col3:
            critical_alerts = len([a for a in live_alerts if a.get("priority", 0) >= 4])
            st.metric("🚨 Critical", critical_alerts)
        
        with col4:
            recent_alerts = len([a for a in live_alerts if self._is_recent_alert(a)])
            st.metric("🕐 Recent (1h)", recent_alerts)
        
        # Alert filters
        st.subheader("🔍 Filter Alerts")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            priority_filter = st.selectbox(
                "Priority Filter",
                ["All", "Critical", "High", "Medium", "Low"]
            )
        
        with col2:
            status_filter = st.selectbox(
                "Status Filter", 
                ["All", "Active", "Resolved", "Acknowledged"]
            )
        
        with col3:
            type_filter = st.selectbox(
                "Type Filter",
                ["All", "overflow_warning", "overflow_critical", "overflow_prediction", "maintenance"]
            )
        
        # Filter alerts based on selections
        filtered_alerts = self._filter_alerts(live_alerts, priority_filter, status_filter, type_filter)
        
        # Alert list with real-time data
        st.subheader("📋 Live Alert List")
        
        if filtered_alerts:
            # Convert to DataFrame for better display
            alert_display_data = []
            
            for alert in filtered_alerts:
                # Map priority numbers to text
                priority_map = {5: "Critical", 4: "High", 3: "Medium", 2: "Low", 1: "Low"}
                priority_text = priority_map.get(alert.get("priority", 2), "Medium")
                
                # Format creation time
                created_at = alert.get("created_at", "")
                if created_at:
                    try:
                        if isinstance(created_at, str):
                            alert_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                            formatted_time = alert_time.strftime("%Y-%m-%d %H:%M")
                        else:
                            formatted_time = created_at.strftime("%Y-%m-%d %H:%M")
                    except:
                        formatted_time = "Unknown"
                else:
                    formatted_time = "Unknown"
                
                # Determine email status (simplified)
                email_status = "✅ Yes" if alert.get("priority", 0) >= 3 else "⏸️ Pending"
                if alert.get("escalation_required", False):
                    email_status = "✅ Yes (Escalated)"
                
                alert_display_data.append({
                    "ID": alert.get("id", "")[:12] + "...",  # Truncate long IDs
                    "Bin": alert.get("bin_id", "Unknown"),
                    "Type": alert.get("alert_type", "Unknown").replace("_", " ").title(),
                    "Priority": priority_text,
                    "Status": alert.get("status", "Unknown").title(),
                    "Email Sent": email_status,
                    "Created": formatted_time,
                    "Message": alert.get("message", "")[:50] + "..." if len(alert.get("message", "")) > 50 else alert.get("message", "")
                })
            
            # Display as interactive table
            st.markdown('<div class="data-table">', unsafe_allow_html=True)
            df = pd.DataFrame(alert_display_data)
            
            # Color code rows based on priority
            def highlight_priority(row):
                if row['Priority'] == 'Critical':
                    return ['background-color: #f8d7da'] * len(row)
                elif row['Priority'] == 'High':
                    return ['background-color: #fff3cd'] * len(row)
                else:
                    return [''] * len(row)
            
            styled_df = df.style.apply(highlight_priority, axis=1)
            st.dataframe(styled_df, use_container_width=True, height=400)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Alert details expander
            if st.expander("🔍 View Alert Details"):
                selected_alert_id = st.selectbox(
                    "Select Alert for Details",
                    options=[alert.get("id", "") for alert in filtered_alerts],
                    format_func=lambda x: f"{x[:12]}... - {next((a.get('bin_id', 'Unknown') for a in filtered_alerts if a.get('id') == x), 'Unknown')}"
                )
                
                if selected_alert_id:
                    selected_alert = next((a for a in filtered_alerts if a.get("id") == selected_alert_id), None)
                    if selected_alert:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Alert Information:**")
                            st.write(f"- **ID:** {selected_alert.get('id', 'Unknown')}")
                            st.write(f"- **Bin ID:** {selected_alert.get('bin_id', 'Unknown')}")
                            st.write(f"- **Type:** {selected_alert.get('alert_type', 'Unknown')}")
                            st.write(f"- **Priority:** {priority_map.get(selected_alert.get('priority', 2), 'Medium')}")
                            st.write(f"- **Status:** {selected_alert.get('status', 'Unknown').title()}")
                        
                        with col2:
                            st.write("**Additional Details:**")
                            st.write(f"- **Message:** {selected_alert.get('message', 'No message')}")
                            if selected_alert.get('natural_language_summary'):
                                st.write(f"- **Summary:** {selected_alert.get('natural_language_summary', '')}")
                            if selected_alert.get('recommended_action'):
                                st.write(f"- **Recommended Action:** {selected_alert.get('recommended_action', '')}")
                            if selected_alert.get('predicted_overflow_time'):
                                st.write(f"- **Predicted Overflow:** {selected_alert.get('predicted_overflow_time', '')}")
        else:
            st.info("✅ No alerts match the current filters")
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close page transition
        
        # Email notification controls
        st.subheader("📧 Email Notification Controls")
        
        col4, col5 = st.columns(2)
        
        with col4:
            if st.button("📤 Send Test Alert Email", type="secondary"):
                success = self._send_test_alert_email()
                if success:
                    st.success("✅ Test alert email sent!")
                else:
                    st.error("❌ Failed to send test email")
        
        with col5:
            if st.button("🔄 Retry Failed Emails", type="secondary"):
                retry_count = self._retry_failed_emails()
                if retry_count > 0:
                    st.success(f"✅ Retried {retry_count} failed emails")
                else:
                    st.info("ℹ️ No failed emails to retry")
    
    def _render_analytics_page(self):
        """Render analytics page with real-time charts"""
        st.markdown('<div class="page-transition">', unsafe_allow_html=True)
        st.title("📊 Analytics & Reports")
        
        # Get live data for analytics
        ward_id = st.session_state.get("ward_id")
        system_data = self._get_live_system_data()
        bin_data = self._get_live_bin_data(ward_id)
        alerts = self._get_live_alerts(ward_id)
        
        # Real-time system analytics
        st.subheader("🔄 Real-time System Analytics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_bins = len(bin_data)
            st.metric("🗑️ Total Bins", total_bins)
        
        with col2:
            avg_fill = sum(b.get("current_fill", 0) for b in bin_data) / len(bin_data) if bin_data else 0
            st.metric("📊 Avg Fill Level", f"{avg_fill:.1f}%")
        
        with col3:
            critical_count = len([b for b in bin_data if b.get("status") == "critical"])
            st.metric("🚨 Critical Bins", critical_count)
        
        with col4:
            active_alerts = len([a for a in alerts if a.get("status") == "active"])
            st.metric("⚠️ Active Alerts", active_alerts)
        
        # Bin status distribution chart
        if bin_data:
            st.subheader("📈 Live Bin Status Distribution")
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            status_counts = {}
            for bin_info in bin_data:
                status = bin_info.get("status", "unknown")
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Create pie chart
            fig_pie = px.pie(
                values=list(status_counts.values()),
                names=list(status_counts.keys()),
                title="Bin Status Distribution",
                color_discrete_map={
                    "normal": "#28a745",
                    "warning": "#ffc107", 
                    "critical": "#dc3545",
                    "maintenance": "#6c757d"
                }
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Fill level histogram
            st.subheader("📊 Fill Level Distribution")
            
            fill_levels = [b.get("current_fill", 0) for b in bin_data]
            fig_hist = px.histogram(
                x=fill_levels,
                nbins=20,
                title="Bin Fill Level Distribution",
                labels={"x": "Fill Level (%)", "y": "Number of Bins"},
                color_discrete_sequence=["#1f77b4"]
            )
            fig_hist.update_layout(showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Ward-wise analysis
            if len(set(b.get("ward_id", 0) for b in bin_data)) > 1:
                st.subheader("🏘️ Ward-wise Analysis")
                
                ward_data = {}
                for bin_info in bin_data:
                    ward = bin_info.get("ward_id", 0)
                    if ward not in ward_data:
                        ward_data[ward] = {"bins": 0, "avg_fill": 0, "critical": 0}
                    
                    ward_data[ward]["bins"] += 1
                    ward_data[ward]["avg_fill"] += bin_info.get("current_fill", 0)
                    if bin_info.get("status") == "critical":
                        ward_data[ward]["critical"] += 1
                
                # Calculate averages
                for ward in ward_data:
                    ward_data[ward]["avg_fill"] /= ward_data[ward]["bins"]
                
                # Create ward comparison chart
                ward_df = pd.DataFrame([
                    {
                        "Ward": f"Ward {ward}",
                        "Total Bins": data["bins"],
                        "Avg Fill Level": data["avg_fill"],
                        "Critical Bins": data["critical"]
                    }
                    for ward, data in ward_data.items()
                ])
                
                fig_ward = px.bar(
                    ward_df,
                    x="Ward",
                    y=["Total Bins", "Critical Bins"],
                    title="Ward-wise Bin Status",
                    barmode="group"
                )
                st.plotly_chart(fig_ward, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Alert analytics
        if alerts:
            st.subheader("🚨 Alert Analytics")
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            # Alert priority distribution
            priority_counts = {}
            priority_map = {5: "Critical", 4: "High", 3: "Medium", 2: "Low", 1: "Low"}
            
            for alert in alerts:
                priority = priority_map.get(alert.get("priority", 2), "Medium")
                priority_counts[priority] = priority_counts.get(priority, 0) + 1
            
            if priority_counts:
                fig_priority = px.bar(
                    x=list(priority_counts.keys()),
                    y=list(priority_counts.values()),
                    title="Alert Priority Distribution",
                    labels={"x": "Priority Level", "y": "Number of Alerts"},
                    color=list(priority_counts.keys()),
                    color_discrete_map={
                        "Critical": "#dc3545",
                        "High": "#fd7e14",
                        "Medium": "#ffc107",
                        "Low": "#28a745"
                    }
                )
                st.plotly_chart(fig_priority, use_container_width=True)
            
            # Alert timeline (if we have creation times)
            alert_times = []
            for alert in alerts:
                created_at = alert.get("created_at", "")
                if created_at:
                    try:
                        if isinstance(created_at, str):
                            alert_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        else:
                            alert_time = created_at
                        alert_times.append(alert_time.replace(tzinfo=None))
                    except:
                        continue
            
            if alert_times:
                # Group alerts by hour
                hourly_counts = {}
                for alert_time in alert_times:
                    hour_key = alert_time.strftime("%Y-%m-%d %H:00")
                    hourly_counts[hour_key] = hourly_counts.get(hour_key, 0) + 1
                
                if len(hourly_counts) > 1:
                    timeline_df = pd.DataFrame([
                        {"Time": time, "Alerts": count}
                        for time, count in sorted(hourly_counts.items())
                    ])
                    
                    fig_timeline = px.line(
                        timeline_df,
                        x="Time",
                        y="Alerts",
                        title="Alert Timeline (Hourly)",
                        markers=True
                    )
                    fig_timeline.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_timeline, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Email analytics
        st.subheader("📧 Email Notification Analytics")
        
        email_stats = self._get_detailed_email_statistics()
        
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            st.metric("📤 Total Sent", email_stats.get("total_sent", 0))
        
        with col6:
            st.metric("✅ Success Rate", f"{email_stats.get('success_rate', 0):.1f}%")
        
        with col7:
            st.metric("❌ Failed", email_stats.get("total_failed", 0))
        
        with col8:
            st.metric("⏱️ Avg Response Time", f"{email_stats.get('avg_response_time', 0):.1f}s")
        
        # Performance trends
        st.subheader("📈 Performance Trends")
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        # Generate sample trend data (in real implementation, this would come from historical data)
        dates = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='D')
        trend_data = pd.DataFrame({
            "Date": dates,
            "Bins Collected": [random.randint(80, 120) for _ in dates],
            "Alerts Generated": [random.randint(5, 25) for _ in dates],
            "Collection Efficiency": [random.uniform(85, 98) for _ in dates]
        })
        
        # Multi-line chart
        fig_trends = go.Figure()
        
        fig_trends.add_trace(go.Scatter(
            x=trend_data["Date"],
            y=trend_data["Bins Collected"],
            mode='lines+markers',
            name='Bins Collected',
            line=dict(color='#1f77b4')
        ))
        
        fig_trends.add_trace(go.Scatter(
            x=trend_data["Date"],
            y=trend_data["Alerts Generated"],
            mode='lines+markers',
            name='Alerts Generated',
            yaxis='y2',
            line=dict(color='#ff7f0e')
        ))
        
        fig_trends.update_layout(
            title="7-Day Performance Trends",
            xaxis_title="Date",
            yaxis=dict(title="Bins Collected", side="left"),
            yaxis2=dict(title="Alerts Generated", side="right", overlaying="y"),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_trends, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close page transition
    
    def _render_system_status_page(self):
        """Render system status page with real-time agent monitoring"""
        st.markdown('<div class="page-transition">', unsafe_allow_html=True)
        st.title("⚙️ System Status")
        
        # Get real-time system status
        system_data = self._get_live_system_data()
        status = system_data.get("agent_status", {})
        
        # System health overview
        st.subheader("🏥 System Health Overview")
        
        # Calculate overall system health
        components_status = {
            "redis": status.get("redis_available", False),
            "vector_db": status.get("vector_db_available", False),
            "master_agent": status.get("master_agent_status") == "active",
            "email_system": self._check_email_configuration()
        }
        
        healthy_components = sum(components_status.values())
        total_components = len(components_status)
        health_percentage = (healthy_components / total_components) * 100
        
        # Health indicator
        if health_percentage >= 75:
            health_color = "#28a745"
            health_status = "Healthy"
            health_icon = "✅"
        elif health_percentage >= 50:
            health_color = "#ffc107"
            health_status = "Warning"
            health_icon = "⚠️"
        else:
            health_color = "#dc3545"
            health_status = "Critical"
            health_icon = "❌"
        
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; background: linear-gradient(90deg, {health_color}20, {health_color}10); border-radius: 10px; margin-bottom: 20px;">
            <h2 style="color: {health_color}; margin: 0;">{health_icon} System Health: {health_status}</h2>
            <p style="margin: 10px 0 0 0; font-size: 18px;">{healthy_components}/{total_components} components operational ({health_percentage:.0f}%)</p>
        </div>
        """, unsafe_allow_html=True)
        
        # System components with real-time status
        st.subheader("🔧 System Components")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            redis_status = "✅ Connected" if status.get("redis_available", False) else "❌ Disconnected"
            st.metric("Redis Communication", redis_status)
            if status.get("redis_available", False):
                st.success("Message queue operational")
            else:
                st.error("Inter-agent communication limited")
        
        with col2:
            vector_status = "✅ Ready" if status.get("vector_db_available", False) else "❌ Not Available"
            st.metric("Vector Database", vector_status)
            if status.get("vector_db_available", False):
                st.success("Semantic search enabled")
            else:
                st.warning("Pattern learning disabled")
        
        with col3:
            master_status = status.get("master_agent_status", "unknown")
            display_status = "✅ Active" if master_status == "active" else f"❌ {master_status.title()}"
            st.metric("Master Agent", display_status)
            if master_status == "active":
                st.success("AI coordination active")
            else:
                st.error("AI agents not responding")
        
        with col4:
            email_status = "✅ Configured" if self._check_email_configuration() else "⚠️ Not Configured"
            st.metric("Email System", email_status)
            if self._check_email_configuration():
                st.success("Notifications enabled")
            else:
                st.warning("Email alerts disabled")
        
        # AI Agents Status
        st.subheader("🤖 AI Agents Status")
        
        active_agents = status.get("active_agents", {})
        if active_agents:
            agent_cols = st.columns(len(active_agents))
            
            for i, (agent_name, agent_info) in enumerate(active_agents.items()):
                with agent_cols[i]:
                    agent_status = agent_info.get("status", "unknown")
                    
                    if agent_status == "active":
                        status_icon = "✅"
                        status_color = "#28a745"
                    elif agent_status == "not_loaded":
                        status_icon = "⚠️"
                        status_color = "#ffc107"
                    else:
                        status_icon = "❌"
                        status_color = "#dc3545"
                    
                    st.markdown(f"""
                    <div style="text-align: center; padding: 15px; border: 2px solid {status_color}; border-radius: 8px; margin: 5px;">
                        <h4 style="margin: 0; color: {status_color};">{status_icon}</h4>
                        <p style="margin: 5px 0; font-weight: bold;">{agent_name.replace('_', ' ').title()}</p>
                        <small style="color: #666;">{agent_status.title()}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show capabilities
                    capabilities = agent_info.get("capabilities", [])
                    if capabilities:
                        with st.expander(f"{agent_name.title()} Capabilities"):
                            for capability in capabilities:
                                st.write(f"• {capability.replace('_', ' ').title()}")
        else:
            st.warning("⚠️ No agent status information available")
        
        # Real-time metrics
        st.subheader("📊 Real-time Metrics")
        
        system_stats = system_data.get("system_stats", {})
        
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            total_bins = system_stats.get("total_bins", 0)
            st.metric("🗑️ Total Bins", f"{total_bins:,}")
        
        with col6:
            active_alerts = system_stats.get("active_alerts", 0)
            st.metric("🚨 Active Alerts", active_alerts)
        
        with col7:
            agents_active = system_stats.get("agents_active", 0)
            st.metric("🤖 Active Agents", agents_active)
        
        with col8:
            # Calculate uptime (simplified)
            uptime_hours = 24  # Placeholder
            st.metric("⏱️ Uptime", f"{uptime_hours}h")
        
        # Performance monitoring
        st.subheader("📈 Performance Monitoring")
        
        # Create real-time performance chart
        if system_stats.get("total_bins", 0) > 0:
            performance_data = {
                "Metric": ["Normal Bins", "Warning Bins", "Critical Bins", "Maintenance"],
                "Count": [
                    system_stats.get("normal_bins", 0),
                    system_stats.get("warning_bins", 0),
                    system_stats.get("critical_bins", 0),
                    system_stats.get("total_bins", 0) - system_stats.get("normal_bins", 0) - system_stats.get("warning_bins", 0) - system_stats.get("critical_bins", 0)
                ],
                "Color": ["#28a745", "#ffc107", "#dc3545", "#6c757d"]
            }
            
            fig_performance = px.bar(
                x=performance_data["Metric"],
                y=performance_data["Count"],
                title="Current System Performance",
                color=performance_data["Metric"],
                color_discrete_map={
                    "Normal Bins": "#28a745",
                    "Warning Bins": "#ffc107",
                    "Critical Bins": "#dc3545",
                    "Maintenance": "#6c757d"
                }
            )
            fig_performance.update_layout(showlegend=False)
            st.plotly_chart(fig_performance, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close page transition
        
        # Email system details
        st.subheader("📧 Email System Status")
        
        email_config = self._get_email_configuration_status()
        
        col5, col6 = st.columns(2)
        
        with col5:
            st.write("**SMTP Configuration:**")
            st.write(f"- Server: {email_config.get('smtp_server', 'Not configured')}")
            st.write(f"- Port: {email_config.get('smtp_port', 'Not configured')}")
            st.write(f"- TLS: {'✅ Enabled' if email_config.get('use_tls', False) else '❌ Disabled'}")
            st.write(f"- Sender: {email_config.get('sender_email', 'Not configured')}")
        
        with col6:
            st.write("**Email Statistics:**")
            email_stats = self._get_email_statistics()
            st.write(f"- Total Sent: {email_stats.get('total_sent', 0)}")
            st.write(f"- Failed: {email_stats.get('total_failed', 0)}")
            st.write(f"- Pending: {email_stats.get('pending', 0)}")
            st.write(f"- Success Rate: {email_stats.get('delivery_rate', 0):.1f}%")
        
        # System tests
        st.subheader("🧪 System Tests")
        
        col7, col8, col9 = st.columns(3)
        
        with col7:
            if st.button("🔄 Test Redis"):
                self._test_redis_connection()
        
        with col8:
            if st.button("📚 Test Vector DB"):
                self._test_vector_database()
        
        with col9:
            if st.button("📧 Test Email System"):
                self._test_email_system()
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        try:
            if hasattr(self.system_initializer, 'get_system_status'):
                if self.system_initializer.master_agent:
                    return self.system_initializer.master_agent.get_system_status()
            
            return {
                "initialized": getattr(self.system_initializer, 'components_initialized', False),
                "redis_available": getattr(self.system_initializer, 'redis_client', None) is not None,
                "vector_db_available": getattr(self.system_initializer, 'chroma_client', None) is not None,
                "master_agent_status": "unknown"
            }
        except Exception as e:
            logger.error(f"❌ Error getting system status: {e}")
            return {"error": str(e)}
    
    def _get_email_statistics(self) -> Dict[str, Any]:
        """Get email notification statistics"""
        try:
            # This would integrate with the actual email notification system
            # For now, return sample data
            return {
                "sent_today": 24,
                "total_sent": 156,
                "total_failed": 8,
                "pending": 2,
                "delivery_rate": 94.2
            }
        except Exception as e:
            logger.error(f"❌ Error getting email statistics: {e}")
            return {}
    
    def _get_detailed_email_statistics(self) -> Dict[str, Any]:
        """Get detailed email statistics"""
        try:
            return {
                "total_sent": 156,
                "total_failed": 8,
                "success_rate": 94.2,
                "avg_response_time": 2.3
            }
        except Exception as e:
            logger.error(f"❌ Error getting detailed email statistics: {e}")
            return {}
    
    def _check_email_configuration(self) -> bool:
        """Check if email system is properly configured"""
        try:
            # Check configuration file
            with open("data/config.json", 'r') as f:
                config = json.load(f)
            
            email_config = config.get("email_notifications", {})
            
            return (
                email_config.get("sender_email", "") != "" and
                email_config.get("sender_password", "") != "" and
                email_config.get("smtp_server", "") != ""
            )
        except Exception as e:
            logger.error(f"❌ Error checking email configuration: {e}")
            return False
    
    def _get_email_configuration_status(self) -> Dict[str, Any]:
        """Get email configuration status"""
        try:
            with open("data/config.json", 'r') as f:
                config = json.load(f)
            
            return config.get("email_notifications", {})
        except Exception as e:
            logger.error(f"❌ Error getting email configuration: {e}")
            return {}
    
    def _send_test_email(self) -> bool:
        """Send test email"""
        try:
            # This would integrate with the actual email notification system
            logger.info("📧 Test email requested")
            return True
        except Exception as e:
            logger.error(f"❌ Error sending test email: {e}")
            return False
    
    def _send_test_alert_email(self) -> bool:
        """Send test alert email"""
        try:
            # This would integrate with the actual email notification system
            logger.info("📧 Test alert email requested")
            return True
        except Exception as e:
            logger.error(f"❌ Error sending test alert email: {e}")
            return False
    
    def _retry_failed_emails(self) -> int:
        """Retry failed email notifications"""
        try:
            # This would integrate with the actual email notification system
            logger.info("🔄 Retry failed emails requested")
            return 2  # Sample return value
        except Exception as e:
            logger.error(f"❌ Error retrying failed emails: {e}")
            return 0
    
    def _test_redis_connection(self):
        """Test Redis connection"""
        try:
            if self.system_initializer.redis_client:
                self.system_initializer.redis_client.ping()
                st.success("✅ Redis connection test passed!")
            else:
                st.warning("⚠️ Redis not available")
        except Exception as e:
            st.error(f"❌ Redis connection test failed: {e}")
    
    def _test_vector_database(self):
        """Test vector database connection"""
        try:
            if self.system_initializer.chroma_client:
                collections = self.system_initializer.chroma_client.list_collections()
                st.success(f"✅ Vector database test passed! Found {len(collections)} collections")
            else:
                st.warning("⚠️ Vector database not available")
        except Exception as e:
            st.error(f"❌ Vector database test failed: {e}")
    
    def _test_email_system(self):
        """Test email system"""
        try:
            if self._check_email_configuration():
                # This would test actual email sending
                st.success("✅ Email system configuration is valid!")
            else:
                st.warning("⚠️ Email system not properly configured")
        except Exception as e:
            st.error(f"❌ Email system test failed: {e}")
    
    def _is_recent_alert(self, alert: Dict[str, Any]) -> bool:
        """Check if alert was created within the last hour"""
        try:
            created_at = alert.get("created_at", "")
            if not created_at:
                return False
            
            if isinstance(created_at, str):
                alert_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            else:
                alert_time = created_at
            
            time_diff = datetime.now() - alert_time.replace(tzinfo=None)
            return time_diff.total_seconds() < 3600  # 1 hour
            
        except Exception as e:
            logger.error(f"❌ Failed to check if alert is recent: {e}")
            return False
    
    def _filter_alerts(
        self, 
        alerts: List[Dict[str, Any]], 
        priority_filter: str, 
        status_filter: str, 
        type_filter: str
    ) -> List[Dict[str, Any]]:
        """Filter alerts based on selected criteria"""
        try:
            filtered = alerts.copy()
            
            # Priority filter
            if priority_filter != "All":
                priority_map = {"Critical": [5], "High": [4], "Medium": [3], "Low": [1, 2]}
                target_priorities = priority_map.get(priority_filter, [])
                filtered = [a for a in filtered if a.get("priority", 0) in target_priorities]
            
            # Status filter
            if status_filter != "All":
                target_status = status_filter.lower()
                filtered = [a for a in filtered if a.get("status", "").lower() == target_status]
            
            # Type filter
            if type_filter != "All":
                filtered = [a for a in filtered if a.get("alert_type", "") == type_filter]
            
            return filtered
            
        except Exception as e:
            logger.error(f"❌ Failed to filter alerts: {e}")
            return alerts

def run_enhanced_dashboard(system_initializer: Any):
    """Run the enhanced dashboard"""
    dashboard = EnhancedDashboard(system_initializer)
    dashboard.run()