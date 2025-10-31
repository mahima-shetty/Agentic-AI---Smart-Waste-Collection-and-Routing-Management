# Smart Waste Management System - Docker Image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p backend/db data chroma_db

# Expose ports
EXPOSE 8501 8509

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8509/_stcore/health || exit 1

# Start command
CMD ["python", "-m", "streamlit", "run", "judges_demo.py", "--server.port=8509", "--server.address=0.0.0.0"]