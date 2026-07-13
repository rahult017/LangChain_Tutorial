FROM python:3.12-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true \
    STREAMLIT_SERVER_HEADLESS=true \
    # Render specific - use PORT env variable
    PORT=8501

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

# Create non-root user
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

# Set permissions
RUN chmod -R 755 /app && \
    chown -R streamlit:streamlit /app && \
    chown -R streamlit:streamlit /tmp/yt-dlp

USER streamlit

EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Start Streamlit
CMD ["sh", "-c", "mkdir -p /app/logs && \
    streamlit run YouTube_Video_Q_N_A/app.py \
    --server.port=${PORT:-8501} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.serverAddress=0.0.0.0 \
    --server.enableCORS=false \
    --server.enableXsrfProtection=true"]