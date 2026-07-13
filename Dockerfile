FROM python:3.10-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    # Whisper model cache location
    WHISPER_CACHE_DIR="/home/streamlit/.cache/whisper"

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user and directories
RUN groupadd --system --gid 1001 streamlit && \
    useradd --system --uid 1001 --gid streamlit --no-create-home streamlit && \
    mkdir -p /home/streamlit/.cache/whisper && \
    mkdir -p /app/logs /tmp/yt-dlp && \
    chown -R streamlit:streamlit /home/streamlit

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY --chown=streamlit:streamlit . .

# Create startup script
RUN echo '#!/bin/bash\n\
    set -e\n\
    \n\
    echo "========================================="\n\
    echo "🚀 YouTube RAG Assistant Starting..."\n\
    echo "========================================="\n\
    echo "Python version: $(python --version)"\n\
    echo "Configured port: ${PORT:-8501}"\n\
    echo "========================================="\n\
    \n\
    mkdir -p /app/logs /tmp/yt-dlp\n\
    \n\
    exec streamlit run YouTube_Video_Q_N_A/app.py \\\n\
    --server.port=${PORT:-8501} \\\n\
    --server.address=0.0.0.0 \\\n\
    --server.headless=true \\\n\
    --server.enableCORS=false \\\n\
    --server.enableXsrfProtection=true \\\n\
    --browser.serverAddress=0.0.0.0' > /app/start.sh && \
    chmod +x /app/start.sh

# Set permissions
RUN chmod -R 755 /app && \
    chown -R streamlit:streamlit /app && \
    chown -R streamlit:streamlit /tmp/yt-dlp

USER streamlit

EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:${PORT:-8501}/_stcore/health || exit 1

# Use startup script
CMD ["/app/start.sh"]