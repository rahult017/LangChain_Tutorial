# ---- Stage 1: Build ----
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install only essential build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip setuptools wheel && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ---- Stage 2: Production ----
FROM python:3.12-slim AS production

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Install only absolutely necessary system packages
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

COPY --from=builder /opt/venv /opt/venv
COPY --chown=streamlit:streamlit . .

RUN chmod -R 755 /app && \
    chown -R streamlit:streamlit /app && \
    chown -R streamlit:streamlit /tmp/yt-dlp

USER streamlit

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["sh", "-c", "mkdir -p /app/logs && streamlit run YouTube_Video_Q_N_A/app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true"]