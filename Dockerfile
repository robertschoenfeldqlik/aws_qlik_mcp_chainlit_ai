FROM python:3.12-slim

WORKDIR /app

# Install Node.js (for npx-based MCP servers) and create a non-root user.
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    useradd --create-home --shell /bin/bash --uid 1000 appuser

# Install Python deps as root (system-wide), then drop privileges below.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code owned by the non-root user so Chainlit can write its
# session/.files directories at runtime.
COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8000

CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]
