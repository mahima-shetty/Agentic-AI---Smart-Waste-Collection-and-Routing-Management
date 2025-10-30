# Smart Waste Management System - Setup Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the System
```bash
# Option 1: Using the startup script (recommended)
python start_system.py

# Option 2: Direct Streamlit launch
streamlit run app.py

# Option 3: Test system components
python app.py test
```

### 3. Access the Dashboard
Open your web browser and navigate to: `http://localhost:8501`

## System Components

### Core Dependencies Installed
- **Streamlit**: Web dashboard interface
- **FastAPI**: API services (for future tasks)
- **LangChain & LangGraph**: AI agent framework
- **ChromaDB**: Vector database for semantic search
- **Redis**: Inter-agent communication (optional)
- **OR-Tools**: Route optimization
- **Scikit-learn**: Machine learning
- **Pandas & Plotly**: Data processing and visualization
- **TensorFlow**: Deep learning capabilities

### Optional Components
- **Redis Server**: For enhanced inter-agent communication
  - Install Redis locally for full functionality
  - System works without Redis but with limited agent communication

## System Architecture

```
app.py (Unified Entry Point)
├── Master Coordination Agent
├── Redis Communication (optional)
├── ChromaDB Vector Database
├── Streamlit Dashboard
└── SQLite Database
```

## Current Implementation Status

✅ **Completed in Task 1:**
- Unified app.py entry point
- Master Coordination Agent framework
- Redis configuration for MCP protocol
- ChromaDB vector database initialization
- Basic dashboard interface
- Complete dependency management

🚧 **Coming in Future Tasks:**
- Route Optimization Agent
- Alert Management Agent
- Analytics Agent
- Bin Simulator Agent
- Authentication system
- Interactive map visualization
- Advanced AI capabilities

## Testing the System

### Basic System Test
```bash
python app.py test
```

### Component Tests
1. **Redis Connection**: Tests inter-agent communication
2. **Vector Database**: Tests ChromaDB functionality
3. **Master Agent**: Tests agent coordination
4. **Dashboard**: Tests web interface

## Troubleshooting

### Common Issues

1. **Redis Connection Failed**
   - System continues without Redis (limited functionality)
   - Install Redis locally for full features

2. **Import Errors**
   - Run: `pip install -r requirements.txt`
   - Ensure Python 3.8+ is installed

3. **Port Already in Use**
   - Change Streamlit port: `streamlit run app.py --server.port 8502`

4. **ChromaDB Issues**
   - Delete `data/chromadb` folder and restart
   - System will recreate vector database

### System Logs
- Check `smart_waste_system.log` for detailed system information
- Dashboard shows real-time system status

## Development Notes

### File Structure
```
├── app.py                          # Unified entry point
├── requirements.txt                # All dependencies
├── start_system.py                # Startup script
├── ai_agents/
│   └── master_agent.py            # Master Coordination Agent
├── dashboard/
│   └── dashboard.py               # Basic dashboard
├── data/
│   ├── config.json                # System configuration
│   └── chromadb/                  # Vector database storage
└── backend/db/                    # Database files
```

### Configuration
- System settings: `data/config.json`
- Agent parameters, UI themes, alert rules
- MCP protocol configuration
- Database and logging settings

## Next Steps

After successful setup, the system is ready for implementing:
1. Advanced AI agents (Route Optimization, Alert Management, Analytics)
2. Authentication and ward-specific access
3. Interactive map visualization
4. Real-time data processing
5. Natural language interface

The foundation is now in place for building the complete Smart Waste Management System!