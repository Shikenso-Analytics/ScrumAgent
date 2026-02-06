# ScrumAgent Webhook Server Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for webhook server
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    pydantic \
    pyyaml

# Copy application code
COPY scrumagent/ ./scrumagent/
COPY config/ ./config/

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash scrumagent
RUN chown -R scrumagent:scrumagent /app
USER scrumagent

# Expose the webhook port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the webhook server
CMD ["python", "-m", "scrumagent.webhook_server"]
