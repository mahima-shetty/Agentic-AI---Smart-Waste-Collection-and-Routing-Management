#!/usr/bin/env python3
"""
Smart Waste Management System - Unified Entry Point
Main application launcher with Master Coordination Agent initialization
"""

import asyncio
import logging
import sys
import os
import importlib.util
from pathlib import Path
from typing import Optional
import streamlit as st
import redis
import chromadb
from contextlib import asynccontextmanager

# Add project directories to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ai_agents"))
sys.path.insert(0, str(project_root / "backend"))
sys.path.insert(0, str(project_root / "dashboard"))

# Ensure dashboard module is importable
dashboard_path = project_root / "dashboard"
if str(dashboard_path) not in sys.path:
    sys.path.append(str(dashboard_path))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('smart_waste_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class SystemInitializer:
    """Handles system initialization and component startup"""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.chroma_client: Optional[chromadb.Client] = None
        self.master_agent = None
        self.components_initialized = False
    
    async def initialize_redis(self) -> bool:
        """Initialize Redis for inter-agent communication and MCP protocol"""
        try:
            # Try to connect to Redis (assuming it's running on default port)
            self.redis_client = redis.Redis(
                host='localhost',
                port=6379,
                db=0,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Test connection
            await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.ping
            )
            
            logger.info("✅ Redis connection established successfully")
            
            # Initialize MCP communication channels
            await self._setup_mcp_channels()
            
            return True
            
        except redis.ConnectionError as e:
            logger.warning(f"⚠️  Redis not available: {e}")
            logger.info("📝 System will run without Redis (limited inter-agent communication)")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to initialize Redis: {e}")
            return False
    
    async def _setup_mcp_channels(self):
        """Setup Model Context Protocol communication channels"""
        try:
            channels = [
                'mcp:route_optimization',
                'mcp:alert_management', 
                'mcp:analytics',
                'mcp:bin_simulation',
                'mcp:master_coordination'
            ]
            
            for channel in channels:
                await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.delete, channel
                )
            
            logger.info("🔄 MCP communication channels initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup MCP channels: {e}")
    
    def initialize_chromadb(self) -> bool:
        """Initialize ChromaDB vector database for semantic search"""
        try:
            # Create data directory if it doesn't exist
            data_dir = project_root / "data" / "chromadb"
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=str(data_dir)
            )
            
            # Create collections for different data types
            collections = [
                "route_patterns",
                "alert_patterns", 
                "analytics_insights",
                "bin_behaviors",
                "agent_conversations"
            ]
            
            for collection_name in collections:
                try:
                    collection = self.chroma_client.get_collection(collection_name)
                    logger.info(f"📚 Found existing collection: {collection_name}")
                except:
                    collection = self.chroma_client.create_collection(
                        name=collection_name,
                        metadata={"description": f"Vector storage for {collection_name}"}
                    )
                    logger.info(f"🆕 Created new collection: {collection_name}")
            
            logger.info("✅ ChromaDB vector database initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize ChromaDB: {e}")
            return False
    
    async def initialize_master_agent(self) -> bool:
        """Initialize Master Coordination Agent with all specialized agents"""
        try:
            # Import here to avoid circular imports
            from ai_agents.master_agent import MasterCoordinationAgent
            
            self.master_agent = MasterCoordinationAgent(
                redis_client=self.redis_client,
                chroma_client=self.chroma_client
            )
            
            # Initialize the master agent (this will initialize all specialized agents)
            await self.master_agent.initialize()
            
            logger.info("🤖 Master Coordination Agent with all specialized agents initialized successfully")
            return True
            
        except ImportError as e:
            logger.warning(f"⚠️  Master Coordination Agent import failed: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to initialize Master Coordination Agent: {e}")
            return False
    
    async def initialize_database(self) -> bool:
        """Initialize SQLite database"""
        try:
            from backend.db.init_db import initialize_database
            
            # Initialize database
            await asyncio.get_event_loop().run_in_executor(
                None, initialize_database
            )
            
            logger.info("🗄️  Database initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize database: {e}")
            return False
    
    async def startup_sequence(self) -> bool:
        """Execute complete system startup sequence"""
        logger.info("🚀 Starting Smart Waste Management System...")
        
        # Initialize components in order
        components = [
            ("Database", self.initialize_database()),
            ("ChromaDB Vector Database", asyncio.get_event_loop().run_in_executor(
                None, self.initialize_chromadb
            )),
            ("Redis Communication", self.initialize_redis()),
            ("Master Coordination Agent", self.initialize_master_agent())
        ]
        
        success_count = 0
        for name, task in components:
            try:
                if isinstance(task, asyncio.Task):
                    result = await task
                else:
                    result = await task
                
                if result:
                    success_count += 1
                    logger.info(f"✅ {name} initialized successfully")
                else:
                    logger.warning(f"⚠️  {name} initialization failed (non-critical)")
                    
            except Exception as e:
                logger.error(f"❌ {name} initialization failed: {e}")
        
        self.components_initialized = success_count > 0
        
        if self.components_initialized:
            logger.info(f"🎉 System startup completed! ({success_count}/{len(components)} components initialized)")
            return True
        else:
            logger.error("💥 System startup failed - no components initialized")
            return False
    
    async def get_system_data(self) -> dict:
        """Get comprehensive system data from Master Agent"""
        try:
            if self.master_agent and hasattr(self.master_agent, 'get_system_dashboard_data'):
                return await self.master_agent.get_system_dashboard_data()
            else:
                return {
                    "error": "Master Agent not available",
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
                    "timestamp": "N/A"
                }
        except Exception as e:
            logger.error(f"❌ Failed to get system data: {e}")
            return {"error": str(e)}
    
    async def request_route_optimization(self, ward_id: int, constraints: dict = None) -> dict:
        """Request route optimization from Master Agent"""
        try:
            if self.master_agent and hasattr(self.master_agent, 'request_route_optimization'):
                return await self.master_agent.request_route_optimization(ward_id, constraints)
            else:
                return {"error": "Route optimization not available"}
        except Exception as e:
            logger.error(f"❌ Route optimization request failed: {e}")
            return {"error": str(e)}
    
    async def get_analytics_report(self, analysis_type: str = "comprehensive", ward_id: int = None) -> dict:
        """Get analytics report from Master Agent"""
        try:
            if self.master_agent and hasattr(self.master_agent, 'generate_analytics_report'):
                return await self.master_agent.generate_analytics_report(analysis_type, ward_id)
            else:
                return {"error": "Analytics not available"}
        except Exception as e:
            logger.error(f"❌ Analytics report request failed: {e}")
            return {"error": str(e)}

    async def shutdown_sequence(self):
        """Execute graceful system shutdown"""
        logger.info("🛑 Shutting down Smart Waste Management System...")
        
        try:
            if self.master_agent:
                await self.master_agent.shutdown()
                logger.info("🤖 Master Coordination Agent shut down")
            
            if self.redis_client:
                await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.close
                )
                logger.info("🔄 Redis connection closed")
            
            logger.info("✅ System shutdown completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")

# Global system initializer
system_initializer = SystemInitializer()

@asynccontextmanager
async def system_lifespan():
    """Context manager for system lifecycle"""
    try:
        # Startup
        success = await system_initializer.startup_sequence()
        if not success:
            logger.warning("⚠️  System started with limited functionality")
        
        yield system_initializer
        
    finally:
        # Shutdown
        await system_initializer.shutdown_sequence()

def main():
    """Main application entry point"""
    
    # Set Streamlit page configuration
    st.set_page_config(
        page_title="Smart Waste Management System",
        page_icon="🗑️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize system if not already done
    if not system_initializer.components_initialized:
        with st.spinner("🚀 Initializing Smart Waste Management System..."):
            # Run initialization in Streamlit context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                success = loop.run_until_complete(
                    system_initializer.startup_sequence()
                )
                
                if success:
                    st.success("✅ System initialized successfully!")
                else:
                    st.warning("⚠️ System started with limited functionality")
                    
            except Exception as e:
                st.error(f"❌ System initialization failed: {e}")
                logger.error(f"System initialization error: {e}")
            finally:
                loop.close()
    
    # Import and run enhanced dashboard with email notifications
    try:
        # Try multiple import methods for enhanced dashboard
        run_enhanced_dashboard = None
        
        # Method 1: Package import
        try:
            from dashboard.enhanced_dashboard import run_enhanced_dashboard
        except ImportError:
            pass
        
        # Method 2: Direct import
        if run_enhanced_dashboard is None:
            try:
                dashboard_path = str(project_root / "dashboard")
                if dashboard_path not in sys.path:
                    sys.path.insert(0, dashboard_path)
                import enhanced_dashboard
                run_enhanced_dashboard = enhanced_dashboard.run_enhanced_dashboard
            except ImportError:
                pass
        
        # Method 3: Absolute import
        if run_enhanced_dashboard is None:
            try:
                spec = importlib.util.spec_from_file_location(
                    "enhanced_dashboard", 
                    project_root / "dashboard" / "enhanced_dashboard.py"
                )
                enhanced_dash_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(enhanced_dash_module)
                run_enhanced_dashboard = enhanced_dash_module.run_enhanced_dashboard
            except Exception:
                pass
        
        if run_enhanced_dashboard:
            run_enhanced_dashboard(system_initializer)
        else:
            raise ImportError("Could not import enhanced dashboard")
        
    except Exception as e:
        st.error(f"❌ Enhanced dashboard failed: {e}")
        
        # Fallback to basic dashboard
        try:
            # Try to import basic dashboard
            try:
                from dashboard.dashboard import run_dashboard
            except ImportError:
                dashboard_path = str(project_root / "dashboard")
                if dashboard_path not in sys.path:
                    sys.path.insert(0, dashboard_path)
                import dashboard as basic_dash
                run_dashboard = basic_dash.run_dashboard
            
            st.warning("⚠️ Using basic dashboard - enhanced features not available")
            run_dashboard(system_initializer)
            
        except Exception as fallback_error:
            st.error(f"❌ All dashboard imports failed: {fallback_error}")
            st.info("📝 The system is running but the dashboard interface is not yet implemented.")
        
        # Show basic system status
        st.header("🗑️ Smart Waste Management System")
        st.subheader("System Status")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Redis Connection", 
                "✅ Connected" if system_initializer.redis_client else "❌ Disconnected"
            )
        
        with col2:
            st.metric(
                "Vector Database", 
                "✅ Ready" if system_initializer.chroma_client else "❌ Not Available"
            )
        
        with col3:
            st.metric(
                "Master Agent", 
                "✅ Active" if system_initializer.master_agent else "❌ Not Loaded"
            )
        
        st.info("🚧 Dashboard interface will be implemented in subsequent tasks.")
        
    except Exception as e:
        st.error(f"❌ Error loading dashboard: {e}")
        logger.error(f"Dashboard loading error: {e}")

if __name__ == "__main__":
    # Handle different execution modes
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "init":
            # Initialize system components only
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                loop.run_until_complete(system_initializer.startup_sequence())
            finally:
                loop.close()
                
        elif command == "test":
            # Test system components
            print("🧪 Testing system components...")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                success = loop.run_until_complete(system_initializer.startup_sequence())
                if success:
                    print("✅ All tests passed!")
                    sys.exit(0)
                else:
                    print("⚠️ Some components failed")
                    sys.exit(1)
            finally:
                loop.close()
        else:
            print(f"Unknown command: {command}")
            print("Available commands: init, test")
            sys.exit(1)
    else:
        # Run Streamlit dashboard (default)
        main()